#define IN_boundary

#include "boundary_private.h"

// If this is defined particle and mover buffers will not resize dynamically.
// This is the common case for the users.

#define DISABLE_DYNAMIC_RESIZING

// FIXME: ARCHITECTURAL FLAW!  CUSTOM BCS AND SHARED FACES CANNOT
// COEXIST ON THE SAME FACE!  THIS MEANS THAT CUSTOM BOUNDARYS MUST
// REINJECT ALL ABSORBED PARTICLES IN THE SAME DOMAIN!


// Updated by Scott V. Luedtke, XCP-6, December 6, 2018.
// The mover array is now resized along with the particle array.  The mover
// array is filled during advance_p and is most likely to overflow there, not
// here.  Both arrays will now resize down as well.
// 12/20/18: The mover array is no longer resized with the particle array, as
// this actually uses more RAM than having static mover arrays.  The mover will
// still size up if there are too many incoming particles, but I have not
// encountered this.  Some hard-to-understand bit shifts have been replaced with
// cleaner code that the compiler should have no trouble optimizing.
// Spits out lots of warnings. TODO: Remove warnings after testing.

#ifdef V4_ACCELERATION
using namespace v4;
#endif

#ifndef MIN_NP
#define MIN_NP 128 // Default to 4kb (~1 page worth of memory)
//#define MIN_NP 32768 // 32768 particles is 1 MiB of memory.
#endif


enum { MAX_PBC = 32, MAX_SP = 32 };

#if defined(VPIC_USE_AOSOA_P)
void
boundary_p( particle_bc_t       * RESTRICT pbc_list,
            species_t           * RESTRICT sp_list,
            field_array_t       * RESTRICT fa,
            accumulator_array_t * RESTRICT aa )
{
  // Gives the local mp port associated with a local face.
  static const int f2b[6]  = { BOUNDARY(-1, 0, 0),
                               BOUNDARY( 0,-1, 0),
                               BOUNDARY( 0, 0,-1),
                               BOUNDARY( 1, 0, 0),
                               BOUNDARY( 0, 1, 0),
                               BOUNDARY( 0, 0, 1) };

  // Gives the remote mp port associated with a local face.
  static const int f2rb[6] = { BOUNDARY( 1, 0, 0),
                               BOUNDARY( 0, 1, 0),
                               BOUNDARY( 0, 0, 1),
                               BOUNDARY(-1, 0, 0),
                               BOUNDARY( 0,-1, 0),
                               BOUNDARY( 0, 0,-1) };

  // Gives the axis associated with a local face.
  static const int axis[6]  = { 0, 1, 2, 0, 1, 2 };

  // Gives the location of sending face on the receiver.
  static const float dir[6] = { 1, 1, 1, -1, -1, -1 };

  // Temporary store for local particle injectors.
  // FIXME: Ugly static usage.
  static particle_injector_t * RESTRICT ALIGNED(16) ci = NULL;

  static int max_ci = 0;

  int n_send[6], n_recv[6], n_ci;

  species_t * sp;

  int face;

  // Check input args.

  if ( ! sp_list )
  {
    return; // Nothing to do if no species.
  }

  if ( ! fa                ||
       ! aa                ||
       sp_list->g != aa->g ||
       fa->g      != aa->g )
  {
    ERROR( ( "Bad args." ) );
  }

  // Unpack the particle boundary conditions.

  particle_bc_func_t pbc_interact[MAX_PBC];

  void * pbc_params[MAX_PBC];

  const int nb = num_particle_bc( pbc_list );

  if ( nb > MAX_PBC )
  {
    ERROR( ( "Update this to support more particle boundary conditions." ) );
  }

  for( particle_bc_t *pbc = pbc_list; pbc; pbc = pbc->next )
  {
    pbc_interact[ -pbc->id - 3 ] = pbc->interact;
    pbc_params  [ -pbc->id - 3 ] = pbc->params;
  }

  // Unpack fields.

  field_t * RESTRICT ALIGNED(128) f = fa->f;
  grid_t  * RESTRICT              g = fa->g;

  // Unpack accumulator.

  accumulator_t * RESTRICT ALIGNED(128) a0 = aa->a;

  // Unpack the grid.

  const int64_t * RESTRICT ALIGNED(128) neighbor = g->neighbor;
  /**/  mp_t    * RESTRICT              mp       = g->mp;

  const int64_t rangel = g->rangel;
  const int64_t rangeh = g->rangeh;
  const int64_t rangem = g->range[world_size];

  /*const*/ int bc[6], shared[6];
  /*const*/ int64_t range[6];

  for( face = 0; face < 6; face++ )
  {
    bc    [ face ] = g->bc[ f2b[ face ] ];

    shared[ face ] = ( bc[ face ] >= 0          ) &&
                     ( bc[ face ] <  world_size ) &&
                     ( bc[ face ] != world_rank );

    if ( shared[ face ] )
    {
      range[ face ] = g->range[ bc[ face ] ];
    }
  }

  // Begin receiving the particle counts.

  for( face = 0; face < 6; face++ )
  {
    if ( shared[ face ] )
    {
      mp_size_recv_buffer( mp,
			   f2b[ face ],
			   sizeof( int ) );

      mp_begin_recv( mp,
		     f2b[ face ],
		     sizeof( int ),
		     bc[ face ],
		     f2rb[ face ] );
    }
  }

  // Load the particle send and local injection buffers.

  do
  {
    particle_injector_t * RESTRICT ALIGNED(16) pi_send[6];

    // Presize the send and injection buffers.
    //
    // Each buffer is large enough to hold one injector corresponding
    // to every mover in use (worst case, but plausible scenario in
    // beam simulations, is one buffer gets all the movers).
    //
    // FIXME: We could be several times more efficient in our particle
    // injector buffer sizing here.  Namely, we could create on local
    // injector buffer of nm is size.  All injection for all
    // boundaries would be done here.  The local buffer would then be
    // counted to determine the size of each send buffer.  The local
    // buffer would then move all injectors into the approate send
    // buffers (leaving only the local injectors).  This would require
    // some extra data motion though.  (But would give a more robust
    // implementation against variations in MP implementation.)
    //
    // FIXME: This presizing assumes that custom boundary conditions
    // inject at most one particle per incident particle.  Currently,
    // the invocation of pbc_interact[*] insures that assumption will
    // be satisfied (if the handlers conform that it).  We should be
    // more flexible though in the future (especially given above the
    // above overalloc).

    int nm = 0;

    LIST_FOR_EACH( sp, sp_list ) nm += sp->nm;

    for( face = 0; face < 6; face++ )
    {
      if ( shared[ face ] )
      {
        mp_size_send_buffer( mp,
			     f2b[ face ],
			     16 + nm * sizeof( particle_injector_t ) );

        pi_send[ face ] = (particle_injector_t *) ( ( (char *) mp_send_buffer( mp,
                                                                               f2b[ face ] )
						    ) + 16 );

        n_send[face] = 0;
      }
    }

    if ( max_ci < nm )
    {
      particle_injector_t * new_ci = ci;

      FREE_ALIGNED( new_ci );

      MALLOC_ALIGNED( new_ci, nm, 16 );

      ci     = new_ci;
      max_ci = nm;
    }

    n_ci = 0;

    // For each species, load the movers.

    LIST_FOR_EACH( sp, sp_list )
    {
      const float   sp_q  = sp->q;
      const int32_t sp_id = sp->id;

      particle_block_t * RESTRICT ALIGNED(128) pb0 = sp->pb;
      int np = sp->np;

      particle_mover_t * RESTRICT ALIGNED(16)  pm = sp->pm + sp->nm - 1;
      nm = sp->nm;

      particle_injector_t * RESTRICT ALIGNED(16) pi;
      int i, voxel;
      int64_t nn;

      int ib = 0;
      int ip = 0;

      // Note that particle movers for each species are processed in
      // reverse order.  This allows us to backfill holes in the
      // particle list created by boundary conditions and/or
      // communication.  This assumes particles on the mover list are
      // monotonically increasing.  That is: pm[n].i > pm[n-1].i for
      // n=1...nm-1.  advance_p and inject_particle create movers with
      // property if all aged particle injection occurs after
      // advance_p and before this.

      for( ; nm; pm--, nm-- )
      {
        i             = pm->i;
        ib            = i / PARTICLE_BLOCK_SIZE;      // Index of particle block.
        ip            = i - PARTICLE_BLOCK_SIZE * ib; // Index of next particle in block.
        voxel         = pb0[ib].i[ip];
        face          = voxel & 7;
        voxel       >>= 3;
        pb0[ib].i[ip] = voxel;
        nn            = neighbor[ 6*voxel + face ];

        // Absorb.

        if ( nn == absorb_particles )
	{
          // Ideally, we would batch all rhob accumulations together
          // for efficiency.
          ERROR( ( "Need AoSoA implementation of accumulate_rhob." ) );
          // accumulate_rhob( f, p0 + i, g, sp_q );

          goto backfill;
        }

        // Send to a neighboring node.

        if ( ( ( nn >= 0      ) & ( nn <  rangel ) ) |
	     ( ( nn >  rangeh ) & ( nn <= rangem ) ) )
	{
          pi = &pi_send[ face ] [ n_send[ face ]++ ];

          // #ifdef V4_ACCELERATION

          // copy_4x1( &pi->dx,    &p0[i].dx  );
          // copy_4x1( &pi->ux,    &p0[i].ux  );
          // copy_4x1( &pi->dispx, &pm->dispx );

          // #else

          pi->dx    = pb0[ib].dx[ip];
	  pi->dy    = pb0[ib].dy[ip];
	  pi->dz    = pb0[ib].dz[ip];
          pi->i     = nn - range[face];

          pi->ux    = pb0[ib].ux[ip];
	  pi->uy    = pb0[ib].uy[ip];
	  pi->uz    = pb0[ib].uz[ip];
	  pi->w     = pb0[ib].w [ip];

          pi->dispx = pm->dispx;
	  pi->dispy = pm->dispy;
	  pi->dispz = pm->dispz;
          pi->sp_id = sp_id;

          // #endif

          ( &pi->dx )[ axis[ face ] ] = dir[ face ];
          pi->i                       = nn - range[ face ];
          pi->sp_id                   = sp_id;

          goto backfill;
        }

        // User-defined handling.

        // After a particle interacts with a boundary it is removed
        // from the local particle list.  Thus, if a boundary handler
        // does not want a particle destroyed,  it is the boundary
        // handler's job to append the destroyed particle to the list
        // of particles to inject.
        //
        // Note that these destruction and creation processes do _not_
        // adjust rhob by default.  Thus, a boundary handler is
        // responsible for insuring that the rhob is updated
        // appropriate for the incident particle it destroys and for
        // any particles it injects as a result too.
        //
        // Since most boundary handlers do local reinjection and are
        // charge neutral, this means most boundary handlers do
        // nothing to rhob.

        nn = -nn - 3; // Assumes reflective/absorbing are -1, -2

        if ( ( nn >= 0  ) &
             ( nn <  nb ) )
        {
          ERROR( ( "Need AoSoA implementation for pbc_interact called in boundary_p." ) );

          // n_ci += pbc_interact[ nn ]( pbc_params[ nn ],
          //   	                         sp,
          //                             p0 + i,
          //                             pm,
          //                             ci + n_ci,
          //                             1,
          //                             face );
        
          goto backfill;
        }

        // Uh-oh: We fell through.

        WARNING( ( "Unknown boundary interaction ... dropping particle "
		   "(species=%s)",
		   sp->name ) );

      backfill:

        np--;

        // #ifdef V4_ACCELERATION

        // copy_4x1( &p0[i].dx, &p0[np].dx );
        // copy_4x1( &p0[i].ux, &p0[np].ux );

        // #else

        int jb = np / PARTICLE_BLOCK_SIZE;      // Index of particle block.
        int jp = np - PARTICLE_BLOCK_SIZE * jb; // Index of next particle in block.

        pb0[ib].dx[ip] = pb0[jb].dx[jp];
        pb0[ib].dy[ip] = pb0[jb].dy[jp];
        pb0[ib].dz[ip] = pb0[jb].dz[jp];
        pb0[ib].i [ip] = pb0[jb].i [jp];

        pb0[ib].ux[ip] = pb0[jb].ux[jp];
        pb0[ib].uy[ip] = pb0[jb].uy[jp];
        pb0[ib].uz[ip] = pb0[jb].uz[jp];
        pb0[ib].w [ip] = pb0[jb].w [jp];

        // #endif
      }

      sp->np = np;
      sp->nm = 0;
    }
  } while( 0 );

  // Finish exchanging particle counts and start exchanging actual
  // particles.

  // Note: This is wasteful of communications.  A better protocol
  // would fuse the exchange of the counts with the exchange of the
  // messages, in a slightly more complex protocol.  However, the MP
  // API prohibits such a model.  Unfortuantely, refining MP is not
  // much help here.  Under the hood on Roadrunner, the DaCS API also
  // prohibits such (specifically, in both, you can't do the
  // equilvanet of a MPI_Getcount to determine how much data you
  // actually received.

  for( face = 0; face < 6; face++ )
  {
    if ( shared[ face ] )
    {
      *( (int *) mp_send_buffer( mp,
				 f2b[face] ) ) = n_send[ face ];

      mp_begin_send( mp,
		     f2b[ face ],
		     sizeof( int ),
		     bc[ face ],
		     f2b[ face ] );
    }
  }

  for( face = 0; face < 6; face++ )
  {
    if ( shared[ face ] )
    {
      mp_end_recv( mp,
		   f2b[ face ] );

      n_recv[ face ] = *( (int *) mp_recv_buffer( mp,
						  f2b[ face ] ) );

      mp_size_recv_buffer( mp,
			   f2b[ face ],
                           16 + n_recv[ face ] * sizeof( particle_injector_t ) );

      mp_begin_recv( mp,
		     f2b[ face ],
		     16 + n_recv[ face ] * sizeof( particle_injector_t ),
                     bc[ face ],
		     f2rb[ face ] );
    }
  }

  for( face = 0; face < 6; face++ )
  {
    if ( shared[ face ] )
    {
      mp_end_send( mp,
		   f2b[ face ] );

      // FIXME: ASSUMES MP WON'T MUCK WITH REST OF SEND BUFFER. IF WE
      // DID MORE EFFICIENT MOVER ALLOCATION ABOVE, THIS WOULD BE
      // ROBUSTED AGAINST MP IMPLEMENTATION VAGARIES.

      mp_begin_send( mp,
		     f2b[ face ],
		     16 + n_send[ face ] * sizeof( particle_injector_t ),
                     bc[ face ],
		     f2b[ face ] );
    }
  }

  #ifndef DISABLE_DYNAMIC_RESIZING
  ERROR( ( "Need AoSoA implementation for boundary_p w/ dynamic resizing." ) );
  // Resize particle storage to accomodate worst case inject.

  do
  {
    int n;

    // Resize each species's particle and mover storage to be large
    // enough to guarantee successful injection.  If we broke down
    // the n_recv[face] by species before sending it, we could be
    // tighter on memory footprint here.

    int max_inj = n_ci;

    for( face = 0; face < 6; face++ )
    {
      if ( shared[ face ] )
      {
	max_inj += n_recv[ face ];
      }
    }

    LIST_FOR_EACH( sp, sp_list )
    {
      particle_mover_t * new_pm;
      particle_t       * new_p;

      n = sp->np + max_inj;

      if ( n > sp->max_np )
      {
        n = n + (n>>2) + (n>>4); // Increase by 31.25% (~<"silver
        /**/                     // ratio") to minimize resizes (max
        /**/                     // rate that avoids excessive heap
        /**/                     // fragmentation)

        WARNING( ( "Resizing local %s particle storage from %i to %i",
                   sp->name,
		   sp->max_np,
		   n ) );

        MALLOC_ALIGNED( new_p, n, 128 );

        COPY( new_p, sp->p, sp->np );

        FREE_ALIGNED( sp->p );

        sp->p      = new_p;
	sp->max_np = n;
      }

      n = sp->nm + max_inj;

      if ( n > sp->max_nm )
      {
        n = n + (n>>2) + (n>>4); // See note above

        WARNING( ( "Resizing local %s mover storage from %i to %i",
                   sp->name,
		   sp->max_nm,
		   n ) );

        MALLOC_ALIGNED( new_pm, n, 128 );

        COPY( new_pm, sp->pm, sp->nm );

        FREE_ALIGNED( sp->pm );

        sp->pm = new_pm;

        sp->max_nm = n;
      }
    }
  } while( 0 );
  #endif

  do
  {
    // Unpack the species list for random acesss.

    particle_block_t * RESTRICT ALIGNED(32) sp_pb[ MAX_SP ];
    particle_mover_t * RESTRICT ALIGNED(32) sp_pm[ MAX_SP ];

    float sp_q [ MAX_SP ];
    int   sp_np[ MAX_SP ];
    int   sp_nm[ MAX_SP ];

    #ifdef DISABLE_DYNAMIC_RESIZING
    int sp_max_np[64], n_dropped_particles[64];
    int sp_max_nm[64], n_dropped_movers   [64];
    #endif

    if ( num_species( sp_list ) > MAX_SP )
    {
      ERROR( ( "Update this to support more species." ) );
    }

    LIST_FOR_EACH( sp, sp_list )
    {
      sp_pb[ sp->id ] = sp->pb;
      sp_pm[ sp->id ] = sp->pm;
      sp_q [ sp->id ] = sp->q;
      sp_np[ sp->id ] = sp->np;
      sp_nm[ sp->id ] = sp->nm;

      #ifdef DISABLE_DYNAMIC_RESIZING
      sp_max_np[ sp->id ] = sp->max_np;
      sp_max_nm[ sp->id ] = sp->max_nm;

      n_dropped_particles[ sp->id ] = 0;
      n_dropped_movers   [ sp->id ] = 0;
      #endif
    }

    // Inject particles.  We do custom local injection first to
    // increase message overlap opportunities.

    face = 5;

    do
    {
      /**/  particle_block_t    * RESTRICT ALIGNED(32) pb;
      /**/  particle_mover_t    * RESTRICT ALIGNED(16) pm;
      const particle_injector_t * RESTRICT ALIGNED(16) pi;

      int np, nm, n, id;

      face++;

      if ( face == 7 )
      {
	face = 0;
      }

      if ( face == 6 )
      {
	pi = ci;
	n  = n_ci;
      }

      else if ( shared[ face ] )
      {
        mp_end_recv( mp,
		     f2b[ face ] );

        pi = (const particle_injector_t *)
             ( ( (char *) mp_recv_buffer( mp,
					  f2b[ face ] ) ) + 16 );

        n  = n_recv[ face ];
      }

      else
      {
	continue;
      }

      // Reverse order injection is done to reduce thrashing of the
      // particle list.  Particles are removed in reverse order so the
      // overall impact of removal + injection is to keep injected
      // particles in order.
      //
      // WARNING: THIS TRUSTS THAT THE INJECTORS, INCLUDING THOSE
      // RECEIVED FROM OTHER NODES, HAVE VALID PARTICLE IDS.

      pi += n - 1;

      for( ; n; pi--, n-- )
      {
        id = pi->sp_id;

        pb = sp_pb[id];
	np = sp_np[id];

        pm = sp_pm[id];
	nm = sp_nm[id];

        #ifdef DISABLE_DYNAMIC_RESIZING
        if ( np >= sp_max_np[ id ] )
	{
	  n_dropped_particles[ id ]++;

	  continue;
	}
        #endif

        // #ifdef V4_ACCELERATION

        // copy_4x1( &p[np].dx, &pi->dx );
        // copy_4x1( &p[np].ux, &pi->ux );

        // #else

        int ib_np = np / PARTICLE_BLOCK_SIZE;         // Index of particle block.
        int ip_np = np - PARTICLE_BLOCK_SIZE * ib_np; // Index of next particle in block.

        pb[ib_np].dx[ip_np] = pi->dx;
	pb[ib_np].dy[ip_np] = pi->dy;
	pb[ib_np].dz[ip_np] = pi->dz;
	pb[ib_np].i [ip_np] = pi->i;

        pb[ib_np].ux[ip_np] = pi->ux;
	pb[ib_np].uy[ip_np] = pi->uy;
	pb[ib_np].uz[ip_np] = pi->uz;
	pb[ib_np].w [ip_np] = pi->w;

        // #endif

        sp_np[id] = np + 1;

        #ifdef DISABLE_DYNAMIC_RESIZING
        if ( nm >= sp_max_nm[ id ] )
	{
	  n_dropped_movers[ id ]++;

	  continue;
	}
        #endif

        #ifdef V4_ACCELERATION

        copy_4x1( &pm[nm].dispx, &pi->dispx );

        pm[nm].i = np;

        #else

        pm[nm].dispx = pi->dispx;
	pm[nm].dispy = pi->dispy;
	pm[nm].dispz = pi->dispz;
        pm[nm].i     = np;

        #endif

        sp_nm[ id ] = nm + move_p( pb,
				   pm + nm,
				   a0,
				   g,
				   sp_q[ id ] );
      }
    } while( face != 5 );

    LIST_FOR_EACH( sp, sp_list )
    {
      #ifdef DISABLE_DYNAMIC_RESIZING
      if ( n_dropped_particles[ sp->id ] )
      {
        WARNING( ( "Dropped %i particles from species \"%s\".  Use a larger "
                   "local particle allocation in your simulation setup for "
                   "this species on this node.",
                   n_dropped_particles[ sp->id ],
		   sp->name ) );
      }

      if ( n_dropped_movers[ sp->id ] )
      {
        WARNING( ( "%i particles were not completed moved to their final "
                   "location this timestep for species \"%s\".  Use a larger "
                   "local particle mover buffer in your simulation setup "
                   "for this species on this node.",
                   n_dropped_movers[ sp->id ],
		   sp->name ) );
      }
      #endif

      sp->np = sp_np[ sp->id ];
      sp->nm = sp_nm[ sp->id ];
    }
  } while( 0 );

  for( face = 0; face < 6; face++ )
  {
    if ( shared[ face ] )
    {
      mp_end_send( mp,
		   f2b[ face ] );
    }
  }
}
#else // VPIC_USE_AOSOA_P not defined, VPIC_USE_AOS_P case.
void
boundary_p( particle_bc_t       * RESTRICT pbc_list,
            species_t           * RESTRICT sp_list,
            field_array_t       * RESTRICT fa,
            accumulator_array_t * RESTRICT aa )
{
  // Gives the local mp port associated with a local face.
  static const int f2b[6]  = { BOUNDARY(-1, 0, 0),
                               BOUNDARY( 0,-1, 0),
                               BOUNDARY( 0, 0,-1),
                               BOUNDARY( 1, 0, 0),
                               BOUNDARY( 0, 1, 0),
                               BOUNDARY( 0, 0, 1) };

  // Gives the remote mp port associated with a local face.
  static const int f2rb[6] = { BOUNDARY( 1, 0, 0),
                               BOUNDARY( 0, 1, 0),
                               BOUNDARY( 0, 0, 1),
                               BOUNDARY(-1, 0, 0),
                               BOUNDARY( 0,-1, 0),
                               BOUNDARY( 0, 0,-1) };

  // Gives the axis associated with a local face.
  static const int axis[6]  = { 0, 1, 2,  0,  1,  2 };

  // Gives the location of sending face on the receiver.
  static const float dir[6] = { 1, 1, 1, -1, -1, -1 };

  // Temporary store for local particle injectors.
  // FIXME: Ugly static usage.
  static particle_injector_t * RESTRICT ALIGNED(16) ci = NULL;

  static int max_ci = 0;

  int n_send[6], n_recv[6], n_ci;

  species_t * sp;

  int face;

  particle_t * ALIGNED(32) p_src;
  int s_vox;
  int p_loc;

  // Check input args.

  if ( ! sp_list )
  {
    return; // Nothing to do if no species.
  }

  if ( ! fa                ||
       ! aa                ||
       sp_list->g != aa->g ||
       fa->g      != aa->g )
  {
    ERROR( ( "Bad args." ) );
  }

  // Unpack the particle boundary conditions.

  particle_bc_func_t pbc_interact[MAX_PBC];

  void * pbc_params[MAX_PBC];

  const int nb = num_particle_bc( pbc_list );

  if ( nb > MAX_PBC )
  {
    ERROR( ( "Update this to support more particle boundary conditions." ) );
  }

  for( particle_bc_t * pbc = pbc_list; pbc; pbc = pbc->next )
  {
    pbc_interact[ -pbc->id - 3 ] = pbc->interact;
    pbc_params  [ -pbc->id - 3 ] = pbc->params;
  }

  // Unpack fields.

  field_t * RESTRICT ALIGNED(128) f = fa->f;
  grid_t  * RESTRICT              g = fa->g;

  // Unpack accumulator.

  accumulator_t * RESTRICT ALIGNED(128) a0 = aa->a;

  // Unpack the grid.

  const int64_t * RESTRICT ALIGNED(128) neighbor = g->neighbor;
  /**/  mp_t    * RESTRICT              mp       = g->mp;

  const int64_t rangel = g->rangel;
  const int64_t rangeh = g->rangeh;
  const int64_t rangem = g->range[world_size];

  /*const*/ int bc[6], shared[6];
  /*const*/ int64_t range[6];

  for( face = 0; face < 6; face++ )
  {
    bc    [ face ] = g->bc[ f2b[ face ] ];

    shared[ face ] = ( bc[ face ] >= 0          ) &&
                     ( bc[ face ] <  world_size ) &&
                     ( bc[ face ] != world_rank );

    if ( shared[ face ] )
    {
      range[ face ] = g->range[ bc[ face ] ];
    }
  }

  // Begin receiving the particle counts.

  for( face = 0; face < 6; face++ )
  {
    if ( shared[ face ] )
    {
      mp_size_recv_buffer( mp,
			   f2b[ face ],
			   sizeof( int ) );

      mp_begin_recv( mp,
		     f2b[ face ],
		     sizeof( int ),
		     bc[ face ],
		     f2rb[ face ] );
    }
  }

  // Load the particle send and local injection buffers.

  do
  {
    particle_injector_t * RESTRICT ALIGNED(16) pi_send[6];

    // Presize the send and injection buffers.
    //
    // Each buffer is large enough to hold one injector corresponding
    // to every mover in use (worst case, but plausible scenario in
    // beam simulations, is one buffer gets all the movers).
    //
    // FIXME: We could be several times more efficient in our particle
    // injector buffer sizing here.  Namely, we could create one local
    // injector buffer of nm in size.  All injection for all boundaries
    // would be done here.  The local buffer would then be counted to
    // determine the size of each send buffer.  The local buffer would
    // then move all injectors into the approate send buffers, leaving
    // only the local injectors.  This would require some extra data
    // motion though, but would give a more robust implementation against
    // variations in MP implementation.
    //
    // FIXME: This presizing assumes that custom boundary conditions
    // inject at most one particle per incident particle.  Currently,
    // the invocation of pbc_interact[*] insures that assumption will
    // be satisfied, if the handlers conform that it.  We should be
    // more flexible though in the future, especially given the above
    // overalloc.

    int nm = 0;

    LIST_FOR_EACH( sp, sp_list ) nm += sp->nm;

    for( face = 0; face < 6; face++ )
    {
      if ( shared[ face ] )
      {
        mp_size_send_buffer( mp,
			     f2b[ face ],
			     16 + nm * sizeof( particle_injector_t ) );

        pi_send[ face ] = (particle_injector_t *) ( ( (char *) mp_send_buffer( mp,
									       f2b[ face ] )
						    ) + 16 );

        n_send[ face ] = 0;
      }
    }

    if ( max_ci < nm )
    {
      particle_injector_t * new_ci = ci;

      FREE_ALIGNED( new_ci );

      MALLOC_ALIGNED( new_ci, nm, 16 );

      ci     = new_ci;
      max_ci = nm;
    }

    n_ci = 0;

    // For each species, load the movers.

    LIST_FOR_EACH( sp, sp_list )
    {
      const float   sp_q  = sp->q;
      const int32_t sp_id = sp->id;

      particle_t * RESTRICT ALIGNED(128) p0 = sp->p;
      int np = sp->np;

      particle_mover_t * RESTRICT ALIGNED(16)  pm = sp->pm + sp->nm - 1;
      nm = sp->nm;

      particle_injector_t * RESTRICT ALIGNED(16) pi;
      int i, voxel;
      int64_t nn;

      // Note that particle movers for each species are processed in
      // reverse order.  This allows us to backfill holes in the
      // particle list created by boundary conditions and/or
      // communication.  This assumes particles on the mover list are
      // monotonically increasing.  That is: pm[n].i > pm[n-1].i for
      // n=1...nm-1.  advance_p and inject_particle create movers with
      // property if all aged particle injection occurs after
      // advance_p and before this.

      for( ; nm; pm--, nm-- )
      {
        i       = pm->i;
        voxel   = p0[i].i;
        face    = voxel & 7;
        voxel >>= 3;
        p0[i].i = voxel;
        nn      = neighbor[ 6*voxel + face ];

        // Absorb.

        if ( nn == absorb_particles )
	{
          // Ideally, we would batch all rhob accumulations together
          // for efficiency.
          accumulate_rhob( f, p0 + i, g, sp_q );

          goto backfill;
        }

        // Send to a neighboring node.

        if ( ( ( nn >= 0      ) & ( nn <  rangel ) ) |
	     ( ( nn >  rangeh ) & ( nn <= rangem ) ) )
	{
          pi = &pi_send[ face ] [n_send[ face ]++ ];

          #ifdef V4_ACCELERATION

          copy_4x1( &pi->dx,    &p0[i].dx  );
          copy_4x1( &pi->ux,    &p0[i].ux  );
          copy_4x1( &pi->dispx, &pm->dispx );

          #else

          pi->dx    = p0[i].dx;
	  pi->dy    = p0[i].dy;
	  pi->dz    = p0[i].dz;

          pi->ux    = p0[i].ux;
	  pi->uy    = p0[i].uy;
	  pi->uz    = p0[i].uz;
	  pi->w     = p0[i].w;

          pi->dispx = pm->dispx;
	  pi->dispy = pm->dispy;
	  pi->dispz = pm->dispz;

          #endif

          ( &pi->dx )[ axis[ face ] ] = dir[ face ];
          pi->i                       = nn - range[ face ];
          pi->sp_id                   = sp_id;

          goto backfill;
        }

        // User-defined handling.

        // After a particle interacts with a boundary it is removed
        // from the local particle list.  Thus, if a boundary handler
        // does not want a particle destroyed,  it is the boundary
        // handler's job to append the destroyed particle to the list
        // of particles to inject.
        //
        // Note that these destruction and creation processes do _not_
        // adjust rhob by default.  Thus, a boundary handler is
        // responsible for insuring that the rhob is updated
        // appropriate for the incident particle it destroys and for
        // any particles it injects as a result too.
        //
        // Since most boundary handlers do local reinjection and are
        // charge neutral, this means most boundary handlers do
        // nothing to rhob.

        nn = -nn - 3; // Assumes reflective/absorbing are -1, -2

        if ( ( nn >= 0  ) &
	     ( nn <  nb ) )
	{
          n_ci += pbc_interact[ nn ]( pbc_params[ nn ],
                                      sp,
                                      p0 + i,
                                      pm,
                                      ci + n_ci,
                                      1,
                                      face );

          goto backfill;
        }

        // Uh-oh: We fell through.

        WARNING( ( "Unknown boundary interaction ... dropping particle "
                   "(species=%s)",
		   sp->name ) );

      backfill:

        s_vox = p0[i].i;

        sp->counts[s_vox]--;

        p_loc = sp->partition[s_vox] + sp->counts[s_vox];

        p_src = &p0[p_loc];

        np--;

        #if defined(V8_ACCELERATION)

        copy_8x1( &p0[i].dx, &p0[p_loc].dx );

        clear_8x1( &p0[p_loc].dx );

        #elif defined(V4_ACCELERATION)

        // #ifdef V4_ACCELERATION

        // copy_4x1( &p0[i].dx, &p0[np].dx );
        // copy_4x1( &p0[i].ux, &p0[np].ux );

        copy_4x1( &p0[i].dx, &p0[p_loc].dx );
        copy_4x1( &p0[i].ux, &p0[p_loc].ux );

        clear_4x1( &p0[p_loc].dx );
        clear_4x1( &p0[p_loc].ux );

        #else

        // Is this the best way to do this?

        // p0[i] = p0[np];

        p0[i] = p0[p_loc];

        // Clear the memory for the particle used to fill the hole. Is this the
        // best way to do this?
        CLEAR( p_src, 1 );

        #endif
      }

      sp->np = np;
      sp->nm = 0;
    }

  } while(0);

  // Finish exchanging particle counts and start exchanging actual
  // particles.

  // Note: This is wasteful of communications.  A better protocol
  // would fuse the exchange of the counts with the exchange of the
  // messages.  in a slightly more complex protocol.  However, the MP
  // API prohibits such a model.  Unfortuantely, refining MP is not
  // much help here.  Under the hood on Roadrunner, the DaCS API also
  // prohibits such (specifically, in both, you can't do the
  // equilvanet of a MPI_Getcount to determine how much data you
  // actually received.

  for( face = 0; face < 6; face++ )
  {
    if ( shared[ face ] )
    {
      *( (int *) mp_send_buffer( mp,
				 f2b[ face ] ) ) = n_send[ face ];

      mp_begin_send( mp,
		     f2b[ face ],
		     sizeof( int ),
		     bc[ face ],
		     f2b[ face ] );
    }
  }

  for( face = 0; face < 6; face++ )
  {
    if ( shared[ face ] )
    {
      mp_end_recv( mp,
		   f2b[ face ] );

      n_recv[ face ] = *( (int *) mp_recv_buffer( mp,
						  f2b[ face ] ) );

      mp_size_recv_buffer( mp,
			   f2b[ face ],
                           16 + n_recv[ face ] * sizeof( particle_injector_t ) );

      mp_begin_recv( mp,
		     f2b[ face ],
		     16 + n_recv[ face ] * sizeof( particle_injector_t ),
                     bc[ face ],
		     f2rb[ face ] );
    }
  }

  for( face = 0; face < 6; face++ )
  {
    if ( shared[ face ] )
    {
      mp_end_send( mp,
		   f2b[ face ] );

      // FIXME: ASSUMES MP WON'T MUCK WITH REST OF SEND BUFFER. IF WE
      // DID MORE EFFICIENT MOVER ALLOCATION ABOVE, THIS WOULD BE
      // ROBUSTED AGAINST MP IMPLEMENTATION VAGARIES.

      mp_begin_send( mp,
		     f2b[ face ],
		     16 + n_send[ face ] * sizeof( particle_injector_t ),
                     bc[ face ],
		     f2b[ face ] );
    }
  }

  #ifndef DISABLE_DYNAMIC_RESIZING
  // Resize particle storage to accomodate worst case inject.

  do
  {
    int n, nm;

    // Resize each species's particle and mover storage to be large
    // enough to guarantee successful injection.  If we broke down
    // the n_recv[face] by species before sending it, we could be
    // tighter on memory footprint here.

    int max_inj = n_ci;

    for( face = 0; face < 6; face++ )
    {
      if ( shared[ face ] )
      {
	max_inj += n_recv[ face ];
      }
    }

    LIST_FOR_EACH( sp, sp_list )
    {
      particle_mover_t * new_pm;
      particle_t       * new_p;

      n = sp->np + max_inj;

      if ( n > sp->max_np )
      {
        n += 0.3125 * n; // Increase by 31.25% (~<"silver
        /**/             // ratio") to minimize resizes (max
        /**/             // rate that avoids excessive heap
        /**/             // fragmentation)

        // float resize_ratio = (float)n/sp->max_np;

        WARNING( ( "Resizing local %s particle storage from %i to %i",
                   sp->name,
		   sp->max_np,
		   n ) );

        MALLOC_ALIGNED( new_p, n, 128 );

        COPY( new_p, sp->p, sp->np );

        FREE_ALIGNED( sp->p );

        sp->p      = new_p;
	sp->max_np = n;

        /*nm = sp->max_nm * resize_ratio;
        WARNING(( "Resizing local %s mover storage from %i to %i",
                  sp->name, sp->max_nm, nm ));
        MALLOC_ALIGNED( new_pm, nm, 128 );
        COPY( new_pm, sp->pm, sp->nm );
        FREE_ALIGNED( sp->pm );
        sp->pm = new_pm;
        sp->max_nm = nm;*/
      }

      else if( sp->max_np > MIN_NP          &&
	       n          < sp->max_np >> 1 )
      {
        n += 0.125 * n; // Overallocate by less since this rank is decreasing

        if ( n < MIN_NP )
	{
	  n = MIN_NP;
	}

        // float resize_ratio = (float)n/sp->max_np;

        WARNING( ( "Resizing (shrinking) local %s particle storage from "
                   "%i to %i",
		   sp->name,
		   sp->max_np,
		   n ) );

        MALLOC_ALIGNED( new_p, n, 128 );

        COPY( new_p, sp->p, sp->np );

        FREE_ALIGNED( sp->p );

        sp->p      = new_p;
	sp->max_np = n;

        /*
        nm = sp->max_nm * resize_ratio;

        WARNING(( "Resizing (shrinking) local %s mover storage from "
                    "%i to %i", sp->name, sp->max_nm, nm));
        MALLOC_ALIGNED( new_pm, nm, 128 );
        COPY( new_pm, sp->pm, sp->nm );
        FREE_ALIGNED( sp->pm );
        sp->pm = new_pm, sp->max_nm = nm;
        */
      }

      // Feasibly, a vacuum-filled rank may receive a shock and need more movers
      // than available from MIN_NP.

      nm = sp->nm + max_inj;

      if ( nm > sp->max_nm )
      {
        nm += 0.3125 * nm; // See note above

        // float resize_ratio = (float)nm/sp->max_nm;

        WARNING( ( "This happened.  Resizing local %s mover storage from "
                   "%i to %i based on not enough movers",
                   sp->name,
		   sp->max_nm,
		   nm ) );

        MALLOC_ALIGNED( new_pm, nm, 128 );

        COPY( new_pm, sp->pm, sp->nm );

        FREE_ALIGNED( sp->pm );

        sp->pm     = new_pm;
        sp->max_nm = nm;

        /*
        n = sp->max_np * resize_ratio;
        WARNING(( "Resizing local %s particle storage from %i to %i",
                  sp->name, sp->max_np, n ));
        MALLOC_ALIGNED( new_p, n, 128 );
        COPY( new_p, sp->p, sp->np );
        FREE_ALIGNED( sp->p );
        sp->p = new_p, sp->max_np = n;
        */
      }
    }
  } while(0);
  #endif

  do
  {
    // Unpack the species list for random acesss.

    particle_t       * RESTRICT ALIGNED(32) sp_p [ MAX_SP ];
    particle_mover_t * RESTRICT ALIGNED(32) sp_pm[ MAX_SP ];

    float sp_q [ MAX_SP ];
    int   sp_np[ MAX_SP ];
    int   sp_nm[ MAX_SP ];

    #ifdef DISABLE_DYNAMIC_RESIZING
    int sp_max_np[64], n_dropped_particles[64];
    int sp_max_nm[64], n_dropped_movers   [64];
    #endif

    if ( num_species( sp_list ) > MAX_SP )
    {
      ERROR( ( "Update this to support more species." ) );
    }

    LIST_FOR_EACH( sp, sp_list )
    {
      sp_p [ sp->id ] = sp->p;
      sp_pm[ sp->id ] = sp->pm;
      sp_q [ sp->id ] = sp->q;
      sp_np[ sp->id ] = sp->np;
      sp_nm[ sp->id ] = sp->nm;

      #ifdef DISABLE_DYNAMIC_RESIZING
      sp_max_np[ sp->id ] = sp->max_np;
      sp_max_nm[ sp->id ] = sp->max_nm;

      n_dropped_particles[ sp->id ] = 0;
      n_dropped_movers   [ sp->id ] = 0;
      #endif
    }

    // Inject particles.  We do custom local injection first to
    // increase message overlap opportunities.

    face = 5;

    do
    {
      /**/  particle_t          * RESTRICT ALIGNED(32) p;
      /**/  particle_mover_t    * RESTRICT ALIGNED(16) pm;
      const particle_injector_t * RESTRICT ALIGNED(16) pi;

      int np, nm, n, id;

      face++;

      if ( face == 7 )
      {
	face = 0;
      }

      if ( face == 6 )
      {
	pi = ci;
	n  = n_ci;
      }

      else if ( shared[ face ] )
      {
        mp_end_recv( mp,
		     f2b[ face ] );

        pi = (const particle_injector_t *)
             ( ( (char *) mp_recv_buffer( mp,
					  f2b[ face ] ) ) + 16 );

        n  = n_recv[ face ];
      }

      else
      {
	continue;
      }

      // Reverse order injection is done to reduce thrashing of the
      // particle list Particles are removed in reverse order so the
      // overall impact of removal + injection is to keep injected
      // particles in order.
      //
      // WARNING: THIS TRUSTS THAT THE INJECTORS, INCLUDING THOSE
      // RECEIVED FROM OTHER NODES, HAVE VALID PARTICLE IDS.

      pi += n - 1;

      for( ; n; pi--, n-- )
      {
        id = pi->sp_id;

	// Get a species pointer for this injector.
	sp = find_species_id( id, sp_list );

	// Get the voxel for the particle in this injector.
	s_vox = pi->i;

	// Get the location in the particle array for the next particle
	// location at the end of the target voxel particles.
        p_loc = sp->partition[s_vox] + sp->counts[s_vox];

        p  = sp_p [id];
	np = sp_np[id];

        pm = sp_pm[id];
	nm = sp_nm[id];

        #ifdef DISABLE_DYNAMIC_RESIZING
	// This needs to be changed to use the voxel specific max.
        if ( np >= sp_max_np[ id ] )
	{
	  n_dropped_particles[id]++;

	  continue;
	}
        #endif

        #ifdef V4_ACCELERATION

        copy_4x1( &p[p_loc].dx, &pi->dx );
        copy_4x1( &p[p_loc].ux, &pi->ux );

        #else

        p[p_loc].dx = pi->dx;
	p[p_loc].dy = pi->dy;
	p[p_loc].dz = pi->dz;
	p[p_loc].i  = pi->i;

        p[p_loc].ux = pi->ux;
	p[p_loc].uy = pi->uy;
	p[p_loc].uz = pi->uz;
	p[p_loc].w  = pi->w;

        #endif

        sp_np[id] = np + 1;

        #ifdef DISABLE_DYNAMIC_RESIZING
        if ( nm >= sp_max_nm[ id ] )
	{
	  n_dropped_movers[id]++;

	  continue;
	}
        #endif

        #ifdef V4_ACCELERATION

        copy_4x1( &pm[nm].dispx, &pi->dispx );

        pm[nm].i = p_loc;

        #else

        pm[nm].dispx = pi->dispx;
	pm[nm].dispy = pi->dispy;
	pm[nm].dispz = pi->dispz;
        pm[nm].i     = p_loc;

        #endif

        sp_nm[id] = nm + move_p( p, pm + nm, a0, g, sp_q[id], sp );
      }
    } while( face != 5 );

    LIST_FOR_EACH( sp, sp_list )
    {
      #ifdef DISABLE_DYNAMIC_RESIZING
      if ( n_dropped_particles[ sp->id ] )
      {
        WARNING( ( "Dropped %i particles from species \"%s\".  Use a larger "
                   "local particle allocation in your simulation setup for "
                   "this species on this node.",
                   n_dropped_particles[ sp->id ],
		   sp->name ) );
      }

      if ( n_dropped_movers[ sp->id ] )
      {
        WARNING( ( "%i particles were not completed moved to their final "
                   "location this timestep for species \"%s\".  Use a larger "
                   "local particle mover buffer in your simulation setup "
                   "for this species on this node.",
                   n_dropped_movers[ sp->id ],
		   sp->name ) );
      }
      #endif

      sp->np = sp_np[ sp->id ];
      sp->nm = sp_nm[ sp->id ];
    }

  } while(0);

  for( face = 0; face < 6; face++ )
  {
    if ( shared[ face ] )
    {
      mp_end_send( mp,
		   f2b[ face ] );
    }
  }
}
#endif // End of VPIC_USE_AOSOA_P, VPIC_USE_AOS_P selection.
