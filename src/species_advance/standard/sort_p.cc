//============================================================================//
// Written by:
//   Kevin J. Bowers, Ph.D.
//   Plasma Physics Group (X-1)
//   Applied Physics Division
//   Los Alamos National Lab
// March/April 2004 - Revised and extened from earlier V4PIC versions.
//============================================================================//

#define IN_spa

#include "../species_advance.h"

//----------------------------------------------------------------------------//
// This is the legacy thread serial version of the particle sort.
//----------------------------------------------------------------------------//

#if defined(VPIC_USE_LEGACY_SORT) 

//----------------------------------------------------------------------------//
// 
//----------------------------------------------------------------------------//

#if defined(VPIC_USE_AOSOA_P)
// void
// sort_p( species_t * sp )
// {
//   ERROR(("Need AoSoA implementation."));
// }

void
sort_p( species_t * sp )
{
  if ( !sp )
    ERROR( ( "Bad args." ) );

  sp->last_sorted = sp->g->step;

  particle_block_t * ALIGNED(128) pb = sp->pb;

  const int np                       = sp->np; 
  const int nc                       = sp->g->nv;
  const int nc1                      = nc + 1;

  int * RESTRICT ALIGNED(128) partition = sp->partition;

  static int * RESTRICT ALIGNED(128) next = NULL;

  static int max_nc1 = 0;

  int i, j;

  int ib = 0;
  int ip = 0;

  // Do not need to sort.
  if ( np == 0 )
    return;

  // Allocate the sorting intermediate. Making this into a static is done to
  // avoid heap shredding.
 
  if ( max_nc1 < nc1 )
  {
    // Hack around RESTRICT issues.
    int *tmp = next;

    FREE_ALIGNED( tmp );

    MALLOC_ALIGNED( tmp, nc1, 128 );

    next    = tmp;
    max_nc1 = nc1;
  }

  // Count particles in each cell.
  CLEAR( next, nc1 );

  for( i = 0; i < np; i++ )
  {
    ib = i / PARTICLE_BLOCK_SIZE;          // Index of particle block.
    ip = i - PARTICLE_BLOCK_SIZE * ib;     // Index of next particle in block.

    next[ pb[ib].i[ip] ]++;
  }

  // Convert the count to a partitioning and save a copy in next.
  j = 0;
  for( i = 0; i < nc1; i++ )
  {
    partition[i]  = j;
    j            += next[i];
    next[i]       = partition[i];
  }

  if ( sp->sort_out_of_place )
  {
    //-----------------------------------------------------------------------------------------
    // Original version.
    //-----------------------------------------------------------------------------------------
    // Throw down the particle array in order.

    // /**/  particle_t *          ALIGNED(128) new_p;
    // const particle_t * RESTRICT ALIGNED( 32)  in_p;
    // /**/  particle_t * RESTRICT ALIGNED( 32) out_p;

    // MALLOC_ALIGNED( new_p, sp->max_np, 128 );

    // in_p  = sp->p;
    // out_p = new_p;

    // for( i = 0; i < np; i++ )
    // {
    //   out_p[ next[ in_p[i].i ]++ ] = in_p[i];
    // }

    // FREE_ALIGNED( sp->p );

    // sp->p = new_p;

    //-----------------------------------------------------------------------------------------
    // New particle block version.
    //-----------------------------------------------------------------------------------------
    // Throw down the particle array in order.

    /**/  particle_block_t *          ALIGNED(128) new_pb;
    const particle_block_t * RESTRICT ALIGNED( 32)  in_pb;
    /**/  particle_block_t * RESTRICT ALIGNED( 32) out_pb;

    int max_pblocks = sp->max_np / PARTICLE_BLOCK_SIZE + 1;

    int num_pblocks = np / PARTICLE_BLOCK_SIZE;

    if ( np > num_pblocks * PARTICLE_BLOCK_SIZE )
    {
      num_pblocks++;
    }

    MALLOC_ALIGNED( new_pb, max_pblocks, 128 );

    in_pb  = sp->pb;
    out_pb = new_pb;

    // for( i = 0; i < np; i++ )
    // {
    //   out_p[ next[ in_p[i].i ]++ ] = in_p[i];
    // }

    for( int jb = 0; jb < num_pblocks; jb++ )
    {
      for( int jp = 0; jp < PARTICLE_BLOCK_SIZE; jp++ )
      {
	i  = next[ in_pb[jb].i[jp] ]++;        // Index of particle.

        if ( i >= np ) break;                  // Is this correct logic?

        ib = i / PARTICLE_BLOCK_SIZE;          // Index of particle block.
        ip = i - PARTICLE_BLOCK_SIZE * ib;     // Index of next particle in block.

	out_pb[ib].dx[ip] = in_pb[jb].dx[jp];
	out_pb[ib].dy[ip] = in_pb[jb].dy[jp];
	out_pb[ib].dz[ip] = in_pb[jb].dz[jp];
	out_pb[ib].i [ip] = in_pb[jb].i [jp];

	out_pb[ib].ux[ip] = in_pb[jb].ux[jp];
	out_pb[ib].uy[ip] = in_pb[jb].uy[jp];
	out_pb[ib].uz[ip] = in_pb[jb].uz[jp];
	out_pb[ib].w [ip] = in_pb[jb].w [jp];
      }
    }

    FREE_ALIGNED( sp->pb );

    sp->pb = new_pb;
  }

  else
  {
    // Run sort cycles until the list is sorted.

    // particle_t               save_p;
    // particle_t * ALIGNED(32) src;
    // particle_t * ALIGNED(32) dest;

    int ii_src = 0;
    int ib_src = 0;
    int ip_src = 0;

    int ii_dst = 0;
    int ib_dst = 0;
    int ip_dst = 0;

    float   save_dx;
    float   save_dy;
    float   save_dz;
    int32_t save_i;
    float   save_ux;
    float   save_uy;
    float   save_uz;
    float   save_w;

    i = 0;
    while( i < nc )
    {
      if ( next[i] >= partition[i+1] )
      {
        i++;
      }

      else
      {
        ii_src = next[i];
        ib_src = ii_src / PARTICLE_BLOCK_SIZE;
        ip_src = ii_src - PARTICLE_BLOCK_SIZE * ib_src;

        // src = &p[ next[i] ];

        for( ; ; )
        {
          ii_dst = next[ sp->pb[ ib_src ].i[ ip_src ] ]++;
          ib_dst = ii_dst / PARTICLE_BLOCK_SIZE;
          ip_dst = ii_dst - PARTICLE_BLOCK_SIZE * ib_dst;

          // dest = &p[ next[ src->i ]++ ];

          if ( ( ib_src == ib_dst ) &&
	       ( ip_src == ip_dst ) ) break;

          // if ( src == dest ) break;

          save_dx = sp->pb[ ib_dst ].dx[ ip_dst ];
          save_dy = sp->pb[ ib_dst ].dy[ ip_dst ];
          save_dz = sp->pb[ ib_dst ].dz[ ip_dst ];
          save_i  = sp->pb[ ib_dst ].i [ ip_dst ];
          save_ux = sp->pb[ ib_dst ].ux[ ip_dst ];
          save_uy = sp->pb[ ib_dst ].uy[ ip_dst ];
          save_uz = sp->pb[ ib_dst ].uz[ ip_dst ];
          save_w  = sp->pb[ ib_dst ].w [ ip_dst ];

          sp->pb[ ib_dst ].dx[ ip_dst ] = sp->pb[ ib_src ].dx[ ip_src ];
          sp->pb[ ib_dst ].dy[ ip_dst ] = sp->pb[ ib_src ].dy[ ip_src ];
          sp->pb[ ib_dst ].dz[ ip_dst ] = sp->pb[ ib_src ].dz[ ip_src ];
          sp->pb[ ib_dst ].i [ ip_dst ] = sp->pb[ ib_src ].i [ ip_src ];
          sp->pb[ ib_dst ].ux[ ip_dst ] = sp->pb[ ib_src ].ux[ ip_src ];
          sp->pb[ ib_dst ].uy[ ip_dst ] = sp->pb[ ib_src ].uy[ ip_src ];
          sp->pb[ ib_dst ].uz[ ip_dst ] = sp->pb[ ib_src ].uz[ ip_src ];
          sp->pb[ ib_dst ].w [ ip_dst ] = sp->pb[ ib_src ].w [ ip_src ];

          sp->pb[ ib_src ].dx[ ip_src ] = save_dx;
          sp->pb[ ib_src ].dy[ ip_src ] = save_dy;
          sp->pb[ ib_src ].dz[ ip_src ] = save_dz;
          sp->pb[ ib_src ].i [ ip_src ] = save_i;
          sp->pb[ ib_src ].ux[ ip_src ] = save_ux;
          sp->pb[ ib_src ].uy[ ip_src ] = save_uy;
          sp->pb[ ib_src ].uz[ ip_src ] = save_uz;
          sp->pb[ ib_src ].w [ ip_src ] = save_w;

          // save_p = *dest;
          // *dest  = *src;
          // *src   = save_p;
        }
      }
    }
  }
}
#else
void
sort_p( species_t * sp )
{
  if ( !sp )
    ERROR( ( "Bad args" ) );

  sp->last_sorted = sp->g->step;

  particle_t * ALIGNED(128) p = sp->p;

  const int np                = sp->np; 
  const int nc                = sp->g->nv;
  const int nc1               = nc + 1;

  int * RESTRICT ALIGNED(128) partition = sp->partition;

  static int * RESTRICT ALIGNED(128) next = NULL;

  static int max_nc1 = 0;

  int i, j;

  // Do not need to sort.
  if ( np == 0 )
    return;

  // Allocate the sorting intermediate. Making this into a static is done to
  // avoid heap shredding.
 
  if ( max_nc1 < nc1 )
  {
    // Hack around RESTRICT issues.
    int *tmp = next;

    FREE_ALIGNED( tmp );

    MALLOC_ALIGNED( tmp, nc1, 128 );

    next    = tmp;
    max_nc1 = nc1;
  }

  // Count particles in each cell.
  CLEAR( next, nc1 );

  for( i = 0; i < np; i++ )
  {
    next[ p[i].i ]++;
  }

  // Convert the count to a partitioning and save a copy in next.
  j = 0;
  for( i = 0; i < nc1; i++ )
  {
    partition[i]  = j;
    j            += next[i];
    next[i]       = partition[i];
  }

  if ( sp->sort_out_of_place )
  {
    // Throw down the particle array in order.

    /**/  particle_t *          ALIGNED(128) new_p;
    const particle_t * RESTRICT ALIGNED( 32)  in_p;
    /**/  particle_t * RESTRICT ALIGNED( 32) out_p;

    MALLOC_ALIGNED( new_p, sp->max_np, 128 );

    in_p  = sp->p;
    out_p = new_p;

    for( i = 0; i < np; i++ )
    {
      out_p[ next[ in_p[i].i ]++ ] = in_p[i];
    }

    FREE_ALIGNED( sp->p );

    sp->p = new_p;
  }

  else
  {
    // Run sort cycles until the list is sorted.

    particle_t               save_p;
    particle_t * ALIGNED(32) src;
    particle_t * ALIGNED(32) dest;

    i = 0;
    while( i < nc )
    {
      if ( next[i] >= partition[i+1] )
      {
        i++;
      }

      else
      {
        src = &p[ next[i] ];

        for( ; ; )
        {
          dest = &p[ next[ src->i ]++ ];

          if ( src == dest ) break;

          save_p = *dest;
          *dest  = *src;
          *src   = save_p;
        }
      }
    }
  }
}
#endif

//----------------------------------------------------------------------------//
// This is the new thread parallel version of the particle sort.
//----------------------------------------------------------------------------//

#else

//----------------------------------------------------------------------------//
// Top level function to select and call the proper sort_p function using the
// desired particle sort abstraction.  Currently, the only abstraction
// available is the pipeline abstraction.
//----------------------------------------------------------------------------//

void
sort_p( species_t * sp )
{
  if ( ! sp )
  {
    ERROR( ( "Bad args." ) );
  }

  // Conditionally execute this when more abstractions are available.
  sort_p_pipeline( sp );
}

#endif
