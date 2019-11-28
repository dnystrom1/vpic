// FIXME: PARTICLE MOVERS NEED TO BE OVERALLOCATED IN STRUCTORS TO
// ACCOUNT FOR SPLITTING THE MOVER ARRAY BETWEEN HOST AND PIPELINES

#define IN_spa

#define HAS_V4_PIPELINE
#define HAS_V8_PIPELINE
#define HAS_V16_PIPELINE

#include "spa_private.h"

#include "species_advance_pipeline.h"

#include "../../../util/pipelines/pipelines_exec.h"

#if defined(VPIC_USE_AOSOA_P)

//----------------------------------------------------------------------------//
// Reference implementation for an advance_p pipeline function which does not
// make use of explicit calls to vector intrinsic functions. This is the AoSoA
// version.
//----------------------------------------------------------------------------//

void
test_advance_p_pipeline_scalar( advance_p_pipeline_args_t * args,
                                int pipeline_rank,
                                int n_pipeline )
{
  particle_block_t     * ALIGNED(128) pb0 = args->pb0;
  accumulator_t        * ALIGNED(128) a0  = args->a0;
  const interpolator_t * ALIGNED(128) f0  = args->f0;
  const grid_t         *              g   = args->g;
  /* */ species_t      *              sp  = args->sp;

  particle_block_t     * ALIGNED(32)  pb;
  particle_mover_t     * ALIGNED(16)  pm;
  const interpolator_t * ALIGNED(16)  f;
  float                * ALIGNED(16)  a;

  const float qdt_2mc        = args->qdt_2mc;
  const float cdt_dx         = args->cdt_dx;
  const float cdt_dy         = args->cdt_dy;
  const float cdt_dz         = args->cdt_dz;
  const float qsp            = args->qsp;
  const float one            = 1.0;
  const float one_third      = 1.0/3.0;
  const float two_fifteenths = 2.0/15.0;

  float ex, dexdy, dexdz, d2exdydz;
  float ey, deydz, deydx, d2eydzdx;
  float ez, dezdx, dezdy, d2ezdxdy;

  float cbx, dcbxdx;
  float cby, dcbydy;
  float cbz, dcbzdz;

  float dx, dy, dz;
  float ux, uy, uz;
  float hax, hay, haz;
  float cbxp, cbyp, cbzp;
  float v0, v1, v2, v3, v4, v5;
  float q;

  float ja[12][PARTICLE_BLOCK_SIZE] __attribute__((aligned(64)));

  // float ja[12][PARTICLE_BLOCK_SIZE];

  int itmp, nm, max_nm;

  int first_part; // Index of first particle for this thread.
  int  last_part; // Index of last  particle for this thread.
  int     n_part; // Number of particles for this thread.

  int      n_vox; // Number of voxels for this thread.
  int        vox; // Index of current voxel.

  int first_ix = 0;
  int first_iy = 0;
  int first_iz = 0;

  float wdn_zero, wdn_one;      // Variables used to confuse compiler.
  float ux_old, uy_old, uz_old; // Variables used to confuse compiler.

  DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );

  //--------------------------------------------------------------------------//
  // Compute an equal division of particles across pipeline processes.
  //--------------------------------------------------------------------------//

  DISTRIBUTE( args->np, 1, pipeline_rank, n_pipeline, first_part, n_part );

  last_part = first_part + n_part - 1;

  //--------------------------------------------------------------------------//
  // Determine which movers are reserved for this pipeline.  Movers, 16 bytes,
  // should be reserved for pipelines in at least multiples of 8 such that the
  // set of particle movers reserved for a pipeline is 128-byte aligned and a
  // multiple of 128-byte in size.  The host is guaranteed to get enough movers
  // to process its particles with this allocation.
  //--------------------------------------------------------------------------//

  max_nm = args->max_nm - ( args->np&15 );

  if ( max_nm < 0 ) max_nm = 0;

  DISTRIBUTE( max_nm, 8, pipeline_rank, n_pipeline, itmp, max_nm );

  if ( pipeline_rank == n_pipeline ) max_nm = args->max_nm - itmp;

  pm   = args->pm + itmp;
  nm   = 0;
  itmp = 0;

  //--------------------------------------------------------------------------//
  // Determine which accumulator array to use.  The host gets the first
  // accumulator array.
  //--------------------------------------------------------------------------//

  if ( pipeline_rank != n_pipeline )
  {
    a0 += ( 1 + pipeline_rank ) * POW2_CEIL( ( args->nx + 2 ) *
                                             ( args->ny + 2 ) *
                                             ( args->nz + 2 ), 2 );
  }

  //--------------------------------------------------------------------------//
  // Initialize the particle charge, q. This assumes all particles have the
  // same charge within a species. This needs to be user selectable in case
  // the assumption is not satisfied for some problems.
  //--------------------------------------------------------------------------//
  // See what this does for performance.  In most use cases, w is constant for
  // a given species.

  // q = p->w;

  //--------------------------------------------------------------------------//
  // Set some variables used to confuse compiler into performing stores of
  // data that has not really changed.
  //--------------------------------------------------------------------------//

  wdn_zero = 100.0;
  wdn_one  = 1000.0;

  get_constants( wdn_zero, wdn_one, args->nx );

  //--------------------------------------------------------------------------//
  // Determine the first and last voxel for each pipeline and the number of
  // voxels for each pipeline.
  //--------------------------------------------------------------------------//

  distribute_voxels( first_ix, first_iy, first_iz, n_vox,
		     sp, pipeline_rank, n_pipeline,
		     n_part, first_part, last_part );

  //--------------------------------------------------------------------------//
  // Loop over voxels.
  //--------------------------------------------------------------------------//

  int ix = first_ix;
  int iy = first_iy;
  int iz = first_iz;

  vox = VOXEL( ix, iy, iz,
               sp->g->nx, sp->g->ny, sp->g->nz );

  for( int j = 0; j < n_vox; j++ )
  {
    const int part_start = sp->partition[vox];
    const int part_count = sp->counts[vox];

    // Only do work if there are particles to process in this voxel.
    if ( part_count > 0 )
    {
      // Define the field data.
      ex       = f0[vox].ex;
      dexdy    = f0[vox].dexdy;
      dexdz    = f0[vox].dexdz;
      d2exdydz = f0[vox].d2exdydz;

      ey       = f0[vox].ey;
      deydz    = f0[vox].deydz;
      deydx    = f0[vox].deydx;
      d2eydzdx = f0[vox].d2eydzdx;

      ez       = f0[vox].ez;
      dezdx    = f0[vox].dezdx;
      dezdy    = f0[vox].dezdy;
      d2ezdxdy = f0[vox].d2ezdxdy;

      cbx      = f0[vox].cbx;
      dcbxdx   = f0[vox].dcbxdx;

      cby      = f0[vox].cby;
      dcbydy   = f0[vox].dcbydy;

      cbz      = f0[vox].cbz;
      dcbzdz   = f0[vox].dcbzdz;

      // Initialize cell local accumulator to zero.

      for( int k = 0; k < 12; k++ )
      {
        for( int j = 0; j < PARTICLE_BLOCK_SIZE; j++ )
        {
          ja[k][j] = 0.0f;
        }
      }

      // Initialize particle pointer to first particle in cell. This assumes part_start
      // is always on a PARTICLE_BLOCK_SIZE boundary.
      pb = args->pb0 + part_start / PARTICLE_BLOCK_SIZE;

      int ib = 0;

      // Process the particles in a cell.
      for( int i = 0; i < part_count; i += PARTICLE_BLOCK_SIZE )
      {
        ib = i / PARTICLE_BLOCK_SIZE;          // Index of particle block.

        #ifdef VPIC_SIMD_LEN
        #ifdef ARM_SVE
        #pragma omp simd
        #else
        #pragma omp simd simdlen(VPIC_SIMD_LEN)
        #endif
        #endif
	for( int j = 0; j < PARTICLE_BLOCK_SIZE; j++ )
	{
          // Load position.
          dx   = pb[ib].dx[j];
          dy   = pb[ib].dy[j];
          dz   = pb[ib].dz[j];

          // Interpolate E.
          hax  = qdt_2mc * ( ( ex + dy * dexdy ) + dz * ( dexdz + dy * d2exdydz ) );
          hay  = qdt_2mc * ( ( ey + dz * deydz ) + dx * ( deydx + dz * d2eydzdx ) );
          haz  = qdt_2mc * ( ( ez + dx * dezdx ) + dy * ( dezdy + dx * d2ezdxdy ) );

          // Interpolate B.
          cbxp = cbx + dx * dcbxdx;
          cbyp = cby + dy * dcbydy;
          cbzp = cbz + dz * dcbzdz;

          // Load momentum.
          ux   = pb[ib].ux[j];
          uy   = pb[ib].uy[j];
          uz   = pb[ib].uz[j];
          q    = pb[ib].w [j];  // Hoist this as an optimization.

          // Old momentum + fake.
          ux_old = ux + wdn_zero;
          uy_old = uy + wdn_zero;
          uz_old = uz + wdn_zero;

          // Half advance E.
          ux  += hax;
          uy  += hay;
          uz  += haz;

          // Boris - scalars.
          v0   = qdt_2mc / sqrtf( one + ( ux * ux + ( uy * uy + uz * uz ) ) );
          v1   = cbxp * cbxp + ( cbyp * cbyp + cbzp * cbzp );
          v2   = ( v0 * v0 ) * v1;
          v3   = v0 * ( one + v2 * ( one_third + v2 * two_fifteenths ) );
          v4   = v3 / ( one + v1 * ( v3 * v3 ) );
          v4  += v4;

          // Boris - uprime.
          v0   = ux + v3 * ( uy * cbzp - uz * cbyp );
          v1   = uy + v3 * ( uz * cbxp - ux * cbzp );
          v2   = uz + v3 * ( ux * cbyp - uy * cbxp );

          // Boris - rotation.
          ux  += v4 * ( v1 * cbzp - v2 * cbyp );
          uy  += v4 * ( v2 * cbxp - v0 * cbzp );
          uz  += v4 * ( v0 * cbyp - v1 * cbxp );

          // Half advance E.
          ux  += hax;
          uy  += hay;
          uz  += haz;

          // Store momentum.
          pb[ib].ux[j] = ux_old;
          pb[ib].uy[j] = uy_old;
          pb[ib].uz[j] = uz_old;

          // Get norm displacement.
          v0   = one / sqrtf( one + ( ux * ux + ( uy * uy + uz * uz ) ) );

          ux  *= cdt_dx;
          uy  *= cdt_dy;
          uz  *= cdt_dz;

          ux  *= v0;
          uy  *= v0;
          uz  *= v0;

          // Streak midpoint (inbnds).
          v0   = dx + ux;
          v1   = dy + uy;
          v2   = dz + uz;

          // New position.
          v3   = v0 + ux;
          v4   = v1 + uy;
          v5   = v2 + uz;

          // Old position + fake.
          dx += wdn_zero;
          dy += wdn_zero;
          dz += wdn_zero;

          // FIXME-KJB: COULD SHORT CIRCUIT ACCUMULATION IN THE CASE WHERE QSP==0.
          // Check if inbnds.
          if (  v3 <= one &&  v4 <= one &&  v5 <= one &&
               -v3 <= one && -v4 <= one && -v5 <= one )
          {
            // Common case (inbnds).  Note: accumulator values are 4 times
            // the total physical charge that passed through the appropriate
            // current quadrant in a time-step.

            q *= qsp;

            // Store new position.
            pb[ib].dx[j] = dx;
            pb[ib].dy[j] = dy;
            pb[ib].dz[j] = dz;

            // Streak midpoint.
            dx = v0;
            dy = v1;
            dz = v2;

            // Compute correction.
            v5 = q * ux * uy * uz * one_third;

            // Get accumulator.
            // a  = (float *)( a0 + vox );
            // a  = (float *)( a0 + ii );

            #define ACCUMULATE_J(X,Y,Z,offset)                                \
            v4  = q*u##X;   /* v2 = q ux                            */        \
            v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
            v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
            v1 += v4;       /* v1 = q ux (1+dy)                     */        \
            v4  = one+d##Z; /* v4 = 1+dz                            */        \
            v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
            v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
            v4  = one-d##Z; /* v4 = 1-dz                            */        \
            v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
            v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
            v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
            v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
            v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
            v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */        \
            ja[offset+0][j] += v0;                                            \
            ja[offset+1][j] += v1;                                            \
            ja[offset+2][j] += v2;                                            \
            ja[offset+3][j] += v3

            // a[offset+0] += v0;                                                \
            // a[offset+1] += v1;                                                \
            // a[offset+2] += v2;                                                \
            // a[offset+3] += v3

            ACCUMULATE_J( x, y, z, 0 );
            ACCUMULATE_J( y, z, x, 4 );
            ACCUMULATE_J( z, x, y, 8 );

            #undef ACCUMULATE_J
          }

          // else                                             // Unlikely
          // {
          //   //--------------------------------------------------------------------
          //   // Load and store particle info in the particle mover array to be
          //   // processed by move_p later.
          //   // pm[nm].dispx = ux;
          //   // pm[nm].dispy = uy;
          //   // pm[nm].dispz = uz;
          //   // pm[nm].i     = part_start + i + j;
          //   // // pm[nm].i     = p - p0;
          //   // nm++;
          //   //--------------------------------------------------------------------
          //   local_pm->dispx = ux;
          //   local_pm->dispy = uy;
          //   local_pm->dispz = uz;
	  // 
          //   local_pm->i     = part_start + i + j;
          //   // local_pm->i     = ipart;
	  // 
          //   if ( move_p( pb0, local_pm, a0, g, qsp, sp ) ) // Unlikely
          //   {
          //     if ( nm < max_nm )
          //     {
          //       pm[nm++] = local_pm[0];
          //     }
	  // 
          //     else
          //     {
          //       itmp++;                                    // Unlikely
          //     }
          //   }
          //   //--------------------------------------------------------------------
          // }
	}
      }
    }

    // Add cell local current density to accumulator array. Can this be done with
    // instrinsics? And, would it be faster?

    float * ALIGNED(64) p_a0 = ( float * ALIGNED(64) ) ( a0 + vox );

    for( int j = 0; j < PARTICLE_BLOCK_SIZE; j++ )
    {
      p_a0[ 0] += ja[ 0][j];
      p_a0[ 1] += ja[ 1][j];
      p_a0[ 2] += ja[ 2][j];
      p_a0[ 3] += ja[ 3][j];

      p_a0[ 4] += ja[ 4][j];
      p_a0[ 5] += ja[ 5][j];
      p_a0[ 6] += ja[ 6][j];
      p_a0[ 7] += ja[ 7][j];

      p_a0[ 8] += ja[ 8][j];
      p_a0[ 9] += ja[ 9][j];
      p_a0[10] += ja[10][j];
      p_a0[11] += ja[11][j];
    }

    // Compute next voxel index and its grid indicies.
    NEXT_VOXEL( vox, ix, iy, iz,
                1, sp->g->nx,
                1, sp->g->ny,
                1, sp->g->nz,
                sp->g->nx,
                sp->g->ny,
                sp->g->nz );
  }

  // args->seg[pipeline_rank].pm        = pm;
  // args->seg[pipeline_rank].max_nm    = max_nm;
  // args->seg[pipeline_rank].nm        = nm;
  // args->seg[pipeline_rank].n_ignored = itmp;
}

#else // VPIC_USE_AOSOA_P is not defined i.e. VPIC_USE_AOS_P case.

//----------------------------------------------------------------------------//
// Reference implementation for an advance_p pipeline function which does not
// make use of explicit calls to vector intrinsic functions. This is the AoS
// version.
//----------------------------------------------------------------------------//

void
test_advance_p_pipeline_scalar( advance_p_pipeline_args_t * args,
                                int pipeline_rank,
                                int n_pipeline )
{
  particle_t           * ALIGNED(128) p0 = args->p0;
  accumulator_t        * ALIGNED(128) a0 = args->a0;
  const interpolator_t * ALIGNED(128) f0 = args->f0;
  const grid_t         *              g  = args->g;
  /* */ species_t      *              sp = args->sp;

  particle_t           * ALIGNED(32)  p;
  particle_mover_t     * ALIGNED(16)  pm;
  const interpolator_t * ALIGNED(16)  f;
  float                * ALIGNED(16)  a;

  const float qdt_2mc        = args->qdt_2mc;
  const float cdt_dx         = args->cdt_dx;
  const float cdt_dy         = args->cdt_dy;
  const float cdt_dz         = args->cdt_dz;
  const float qsp            = args->qsp;
  const float one            = 1.0;
  const float one_third      = 1.0/3.0;
  const float two_fifteenths = 2.0/15.0;

  float ex, dexdy, dexdz, d2exdydz;
  float ey, deydz, deydx, d2eydzdx;
  float ez, dezdx, dezdy, d2ezdxdy;

  float cbx, dcbxdx;
  float cby, dcbydy;
  float cbz, dcbzdz;

  float dx, dy, dz;
  float ux, uy, uz;
  float hax, hay, haz;
  float cbxp, cbyp, cbzp;
  float v0, v1, v2, v3, v4, v5;
  float q;
  int   ii;

  int itmp, nm, max_nm;

  int first_part; // Index of first particle for this thread.
  int  last_part; // Index of last  particle for this thread.
  int     n_part; // Number of particles for this thread.

  int      n_vox; // Number of voxels for this thread.
  int        vox; // Index of current voxel.

  int first_ix = 0;
  int first_iy = 0;
  int first_iz = 0;

  float wdn_zero, wdn_one;      // Variables used to confuse compiler.
  float ux_old, uy_old, uz_old; // Variables used to confuse compiler.

  DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );

  //--------------------------------------------------------------------------//
  // Compute an equal division of particles across pipeline processes.
  //--------------------------------------------------------------------------//

  DISTRIBUTE( args->np, 1, pipeline_rank, n_pipeline, first_part, n_part );

  last_part = first_part + n_part - 1;

  // Determine which movers are reserved for this pipeline.
  // Movers (16 bytes) should be reserved for pipelines in at least
  // multiples of 8 such that the set of particle movers reserved for
  // a pipeline is 128-byte aligned and a multiple of 128-byte in
  // size.  The host is guaranteed to get enough movers to process its
  // particles with this allocation.

  max_nm = args->max_nm - ( args->np&15 );

  if ( max_nm < 0 ) max_nm = 0;

  DISTRIBUTE( max_nm, 8, pipeline_rank, n_pipeline, itmp, max_nm );

  if ( pipeline_rank == n_pipeline ) max_nm = args->max_nm - itmp;

  pm   = args->pm + itmp;
  nm   = 0;
  itmp = 0;

  // Determine which accumulator array to use
  // The host gets the first accumulator array.

  if ( pipeline_rank != n_pipeline )
  {
    a0 += ( 1 + pipeline_rank ) * POW2_CEIL( ( args->nx + 2 ) *
                                             ( args->ny + 2 ) *
                                             ( args->nz + 2 ), 2 );
  }

  //--------------------------------------------------------------------------//
  // Set some variables used to confuse compiler into performing stores of
  // data that has not really changed.
  //--------------------------------------------------------------------------//

  wdn_zero = 100.0;
  wdn_one  = 1000.0;

  get_constants( wdn_zero, wdn_one, args->nx );

  //--------------------------------------------------------------------------//
  // Determine the first and last voxel for each pipeline and the number of
  // voxels for each pipeline.
  //--------------------------------------------------------------------------//

  distribute_voxels( first_ix, first_iy, first_iz, n_vox,
		     sp, pipeline_rank, n_pipeline,
		     n_part, first_part, last_part );

  //--------------------------------------------------------------------------//
  // Loop over voxels.
  //--------------------------------------------------------------------------//

  int ix = first_ix;
  int iy = first_iy;
  int iz = first_iz;

  vox = VOXEL( ix, iy, iz,
               sp->g->nx, sp->g->ny, sp->g->nz );

  for( int j = 0; j < n_vox; j++ )
  {
    const int part_start = sp->partition[vox];
    const int part_count = sp->counts[vox];

    // Only do work if there are particles to process in this voxel.
    if ( part_count > 0 )
    {
      // Define the field data.
      ex       = f0[vox].ex;
      dexdy    = f0[vox].dexdy;
      dexdz    = f0[vox].dexdz;
      d2exdydz = f0[vox].d2exdydz;

      ey       = f0[vox].ey;
      deydz    = f0[vox].deydz;
      deydx    = f0[vox].deydx;
      d2eydzdx = f0[vox].d2eydzdx;

      ez       = f0[vox].ez;
      dezdx    = f0[vox].dezdx;
      dezdy    = f0[vox].dezdy;
      d2ezdxdy = f0[vox].d2ezdxdy;

      cbx      = f0[vox].cbx;
      dcbxdx   = f0[vox].dcbxdx;

      cby      = f0[vox].cby;
      dcbydy   = f0[vox].dcbydy;

      cbz      = f0[vox].cbz;
      dcbzdz   = f0[vox].dcbzdz;

      // Initialize particle pointer to first particle in cell.
      p = args->p0 + part_start;

      // Process the particles in a cell.
      // for( int i = 0; i < part_count; i++, p++ )
      #ifdef VPIC_SIMD_LEN
      #ifdef ARM_SVE
      #pragma omp simd
      #else
      #pragma omp simd simdlen(VPIC_SIMD_LEN)
      #endif
      #endif
      for( int i = 0; i < part_count; i++ )
      {
        // Load position.
        dx   = p->dx;
        dy   = p->dy;
        dz   = p->dz;

        // Interpolate E.
        hax  = qdt_2mc * ( ( ex + dy * dexdy ) + dz * ( dexdz + dy * d2exdydz ) );
        hay  = qdt_2mc * ( ( ey + dz * deydz ) + dx * ( deydx + dz * d2eydzdx ) );
        haz  = qdt_2mc * ( ( ez + dx * dezdx ) + dy * ( dezdy + dx * d2ezdxdy ) );

        // Interpolate B.
        cbxp = cbx + dx * dcbxdx;
        cbyp = cby + dy * dcbydy;
        cbzp = cbz + dz * dcbzdz;

        // Load momentum.
        ux   = p->ux;
        uy   = p->uy;
        uz   = p->uz;
        q    = p->w;

        // Old momentum + fake.
        ux_old = ux + wdn_zero;
        uy_old = uy + wdn_zero;
        uz_old = uz + wdn_zero;

        // Half advance E.
        ux  += hax;
        uy  += hay;
        uz  += haz;

        // Boris - scalars.
        v0   = qdt_2mc / sqrtf( one + ( ux * ux + ( uy * uy + uz * uz ) ) );
        v1   = cbxp * cbxp + ( cbyp * cbyp + cbzp * cbzp );
        v2   = ( v0 * v0 ) * v1;
        v3   = v0 * ( one + v2 * ( one_third + v2 * two_fifteenths ) );
        v4   = v3 / ( one + v1 * ( v3 * v3 ) );
        v4  += v4;

        // Boris - uprime.
        v0   = ux + v3 * ( uy * cbzp - uz * cbyp );
        v1   = uy + v3 * ( uz * cbxp - ux * cbzp );
        v2   = uz + v3 * ( ux * cbyp - uy * cbxp );

        // Boris - rotation.
        ux  += v4 * ( v1 * cbzp - v2 * cbyp );
        uy  += v4 * ( v2 * cbxp - v0 * cbzp );
        uz  += v4 * ( v0 * cbyp - v1 * cbxp );

        // Half advance E.
        ux  += hax;
        uy  += hay;
        uz  += haz;

        // Store momentum.
        p->ux = ux_old;
        p->uy = uy_old;
        p->uz = uz_old;

        // Get norm displacement.
        v0   = one / sqrtf( one + ( ux * ux + ( uy * uy + uz * uz ) ) );

        ux  *= cdt_dx;
        uy  *= cdt_dy;
        uz  *= cdt_dz;

        ux  *= v0;
        uy  *= v0;
        uz  *= v0;

        // Streak midpoint (inbnds).
        v0   = dx + ux;
        v1   = dy + uy;
        v2   = dz + uz;

        // New position.
        v3   = v0 + ux;
        v4   = v1 + uy;
        v5   = v2 + uz;

        // Old position + fake.
        dx += wdn_zero;
        dy += wdn_zero;
        dz += wdn_zero;

        // FIXME-KJB: COULD SHORT CIRCUIT ACCUMULATION IN THE CASE WHERE QSP==0.
        // Check if inbnds.
        if (  v3 <= one &&  v4 <= one &&  v5 <= one &&
             -v3 <= one && -v4 <= one && -v5 <= one )
        {
          // Common case (inbnds).  Note: accumulator values are 4 times
          // the total physical charge that passed through the appropriate
          // current quadrant in a time-step.

          q *= qsp;

          // Store new position.
          p->dx = dx;
          p->dy = dy;
          p->dz = dz;

          // Streak midpoint.
          dx = v0;
          dy = v1;
          dz = v2;

          // Compute correction.
          v5 = q * ux * uy * uz * one_third;

          // Get accumulator.
          a  = ( float * ) ( a0 + vox );
          // a  = ( float * ) ( a0 + ii );

          #define ACCUMULATE_J(X,Y,Z,offset)                                \
          v4  = q*u##X;   /* v2 = q ux                            */        \
          v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
          v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
          v1 += v4;       /* v1 = q ux (1+dy)                     */        \
          v4  = one+d##Z; /* v4 = 1+dz                            */        \
          v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
          v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
          v4  = one-d##Z; /* v4 = 1-dz                            */        \
          v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
          v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
          v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
          v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
          v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
          v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */        \
          a[offset+0] += v0;                                                \
          a[offset+1] += v1;                                                \
          a[offset+2] += v2;                                                \
          a[offset+3] += v3

          ACCUMULATE_J( x, y, z, 0 );
          ACCUMULATE_J( y, z, x, 4 );
          ACCUMULATE_J( z, x, y, 8 );

          #undef ACCUMULATE_J
        }

        // else                                            // Unlikely
        // {
        //   // Load and store particle info in the particle mover array to be
        //   // processed by move_p later.
        //   pm[nm].dispx = ux;
        //   pm[nm].dispy = uy;
        //   pm[nm].dispz = uz;
        //   pm[nm].i     = p - p0;
        //   nm++;
        //   //--------------------------------------------------------------------
        //   // local_pm->dispx = ux;
        //   // local_pm->dispy = uy;
        //   // local_pm->dispz = uz;
        //   //
        //   // local_pm->i     = p - p0;
        //   //
        //   // if ( move_p( p0, local_pm, a0, g, qsp, sp ) ) // Unlikely
        //   // {
        //   //   if ( nm < max_nm )
        //   //   {
        //   //     pm[nm++] = local_pm[0];
        //   //   }
        //   //
        //   //   else
        //   //   {
        //   //     itmp++;                                   // Unlikely
        //   //   }
        //   // }
	//   //--------------------------------------------------------------------
        // }

	// Increment particle pointer.
	p++;
      }
    }

    // Compute next voxel index and its grid indicies.
    NEXT_VOXEL( vox, ix, iy, iz,
                1, sp->g->nx,
                1, sp->g->ny,
                1, sp->g->nz,
                sp->g->nx,
                sp->g->ny,
                sp->g->nz );
  }

  // Process the particle mover array with move_p.
  // int nm_move_p = nm;
  // nm = 0;
  // for( ; nm_move_p; nm_move_p-- )
  // {
  //   if ( move_p( p0, &pm[ nm_move_p - 1 ], a0, g, qsp, sp ) ) // Unlikely
  //   {
  //     // Stash these movers at the top of the movers list.
  //     if ( nm < max_nm )
  //     {
  //       nm++;
  //       pm[ max_nm - nm ] = pm[ nm_move_p - 1 ];
  //     }
  // 
  //     else
  //     {
  //       itmp++;                                               // Unlikely
  //     }
  //   }
  // }

  // args->seg[pipeline_rank].pm        = &pm[ max_nm - nm ];
  // args->seg[pipeline_rank].max_nm    = max_nm;
  // args->seg[pipeline_rank].nm        = nm;
  // args->seg[pipeline_rank].n_ignored = itmp;
}

#endif // End of VPIC_USE_AOSOA_P vs VPIC_USE_AOS_P selection.

//----------------------------------------------------------------------------//
// Top level function to select and call the proper advance_p pipeline
// function.
//----------------------------------------------------------------------------//

#if defined(VPIC_USE_AOSOA_P)

void
test_advance_p_pipeline( species_t * RESTRICT sp,
                         accumulator_array_t * RESTRICT aa,
                         const interpolator_array_t * RESTRICT ia )
{
  DECLARE_ALIGNED_ARRAY( advance_p_pipeline_args_t, 128, args, 1 );

  DECLARE_ALIGNED_ARRAY( particle_mover_seg_t, 128, seg, MAX_PIPELINE + 1 );

  int rank;

  if ( ! sp           ||
       ! aa           ||
       ! ia           ||
       sp->g != aa->g ||
       sp->g != ia->g )
  {
    ERROR( ( "Bad args." ) );
  }

  args->pb0     = sp->pb;
  args->pm      = sp->pm;
  args->a0      = aa->a;
  args->f0      = ia->i;
  args->seg     = seg;
  args->g       = sp->g;
  args->sp      = sp;

  args->qdt_2mc = ( sp->q * sp->g->dt ) / ( 2 * sp->m * sp->g->cvac );
  args->cdt_dx  = sp->g->cvac * sp->g->dt * sp->g->rdx;
  args->cdt_dy  = sp->g->cvac * sp->g->dt * sp->g->rdy;
  args->cdt_dz  = sp->g->cvac * sp->g->dt * sp->g->rdz;
  args->qsp     = sp->q;

  args->np      = sp->np;
  args->max_nm  = sp->max_nm;
  args->nx      = sp->g->nx;
  args->ny      = sp->g->ny;
  args->nz      = sp->g->nz;

  // Have the host processor do the last incomplete bundle if necessary.
  // Note: This is overlapped with the pipelined processing.  As such,
  // it uses an entire accumulator.  Reserving an entire accumulator
  // for the host processor to handle at most 15 particles is wasteful
  // of memory.  It is anticipated that it may be useful at some point
  // in the future have pipelines accumulating currents while the host
  // processor is doing other more substantive work (e.g. accumulating
  // currents from particles received from neighboring nodes).
  // However, it is worth reconsidering this at some point in the
  // future.

  EXEC_PIPELINES( test_advance_p, args, 0 );

  WAIT_PIPELINES();

  // FIXME: HIDEOUS HACK UNTIL BETTER PARTICLE MOVER SEMANTICS
  // INSTALLED FOR DEALING WITH PIPELINES.  COMPACT THE PARTICLE
  // MOVERS TO ELIMINATE HOLES FROM THE PIPELINING.

  // sp->nm = 0;
  // for( rank = 0; rank <= N_PIPELINE; rank++ )
  // {
  //   if ( args->seg[rank].n_ignored )
  //   {
  //     WARNING( ( "Pipeline %i ran out of storage for %i movers.",
  //                rank, args->seg[rank].n_ignored ) );
  //   }
  // 
  //   if ( sp->pm + sp->nm != args->seg[rank].pm )
  //   {
  //     MOVE( sp->pm + sp->nm, args->seg[rank].pm, args->seg[rank].nm );
  //   }
  // 
  //   sp->nm += args->seg[rank].nm;
  // }
}

#else

void
test_advance_p_pipeline( species_t * RESTRICT sp,
                         accumulator_array_t * RESTRICT aa,
                         const interpolator_array_t * RESTRICT ia )
{
  DECLARE_ALIGNED_ARRAY( advance_p_pipeline_args_t, 128, args, 1 );

  DECLARE_ALIGNED_ARRAY( particle_mover_seg_t, 128, seg, MAX_PIPELINE + 1 );

  int rank;

  if ( ! sp           ||
       ! aa           ||
       ! ia           ||
       sp->g != aa->g ||
       sp->g != ia->g )
  {
    ERROR( ( "Bad args." ) );
  }

  args->p0      = sp->p;
  args->pm      = sp->pm;
  args->a0      = aa->a;
  args->f0      = ia->i;
  args->seg     = seg;
  args->g       = sp->g;
  args->sp      = sp;

  args->qdt_2mc = ( sp->q * sp->g->dt ) / ( 2 * sp->m * sp->g->cvac );
  args->cdt_dx  = sp->g->cvac * sp->g->dt * sp->g->rdx;
  args->cdt_dy  = sp->g->cvac * sp->g->dt * sp->g->rdy;
  args->cdt_dz  = sp->g->cvac * sp->g->dt * sp->g->rdz;
  args->qsp     = sp->q;

  args->np      = sp->np;
  args->max_nm  = sp->max_nm;
  args->nx      = sp->g->nx;
  args->ny      = sp->g->ny;
  args->nz      = sp->g->nz;

  // Have the host processor do the last incomplete bundle if necessary.
  // Note: This is overlapped with the pipelined processing.  As such,
  // it uses an entire accumulator.  Reserving an entire accumulator
  // for the host processor to handle at most 15 particles is wasteful
  // of memory.  It is anticipated that it may be useful at some point
  // in the future have pipelines accumulating currents while the host
  // processor is doing other more substantive work (e.g. accumulating
  // currents from particles received from neighboring nodes).
  // However, it is worth reconsidering this at some point in the
  // future.

  EXEC_PIPELINES( test_advance_p, args, 0 );

  WAIT_PIPELINES();

  // FIXME: HIDEOUS HACK UNTIL BETTER PARTICLE MOVER SEMANTICS
  // INSTALLED FOR DEALING WITH PIPELINES.  COMPACT THE PARTICLE
  // MOVERS TO ELIMINATE HOLES FROM THE PIPELINING.

  // sp->nm = 0;
  // for( rank = 0; rank <= N_PIPELINE; rank++ )
  // {
  //   if ( args->seg[rank].n_ignored )
  //   {
  //     WARNING( ( "Pipeline %i ran out of storage for %i movers.",
  //                rank, args->seg[rank].n_ignored ) );
  //   }
  // 
  //   if ( sp->pm + sp->nm != args->seg[rank].pm )
  //   {
  //     MOVE( sp->pm + sp->nm, args->seg[rank].pm, args->seg[rank].nm );
  //   }
  // 
  //   sp->nm += args->seg[rank].nm;
  // }
}

#endif
