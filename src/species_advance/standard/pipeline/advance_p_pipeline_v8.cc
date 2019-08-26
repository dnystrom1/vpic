#define IN_spa

#include "spa_private.h"

#if defined(V8_ACCELERATION)

using namespace v8;

#if defined(VPIC_USE_AOSOA_P)

void
advance_p_pipeline_v8( advance_p_pipeline_args_t * args,
                       int pipeline_rank,
                       int n_pipeline )
{
  particle_block_t     * ALIGNED(128) pb0 = args->pb0;
  accumulator_t        * ALIGNED(128) a0  = args->a0;
  const interpolator_t * ALIGNED(128) f0  = args->f0;
  const grid_t         *              g   = args->g;
  /* */ species_t      *              sp  = args->sp;

  particle_block_t     * ALIGNED(128) pb;
  particle_mover_t     * ALIGNED(16)  pm;

  // Basic constants.
  const v8float qdt_2mc(args->qdt_2mc);
  const v8float cdt_dx(args->cdt_dx);
  const v8float cdt_dy(args->cdt_dy);
  const v8float cdt_dz(args->cdt_dz);
  const v8float qsp(args->qsp);
  const v8float one(1.0);
  const v8float one_third(1.0/3.0);
  const v8float two_fifteenths(2.0/15.0);
  const v8float neg_one(-1.0);

  const float _qsp = args->qsp;

  v8float ex, dexdy, dexdz, d2exdydz;
  v8float ey, deydz, deydx, d2eydzdx;
  v8float ez, dezdx, dezdy, d2ezdxdy;

  v8float cbx, dcbxdx;
  v8float cby, dcbydy;
  v8float cbz, dcbzdz;

  v8float jx0, jx1, jx2, jx3;
  v8float jy0, jy1, jy2, jy3;
  v8float jz0, jz1, jz2, jz3;

  v8float dx, dy, dz;
  v8float ux, uy, uz;
  v8float q;
  v8float hax, hay, haz;
  v8float cbxp, cbyp, cbzp;
  v8float v00, v01, v02, v03, v04, v05, v06, v07; // Clean this up.
  v8float v08, v09, v10, v11, v12, v13, v14, v15;
  v8int   outbnd;

  int itmp, nm, max_nm;

  int first_part; // Index of first particle for this thread.
  int  last_part; // Index of last  particle for this thread.
  int     n_part; // Number of particles for this thread.

  int previous_vox; // Index of previous voxel.
  int    first_vox; // Index of first voxel for this thread.
  int     last_vox; // Index of last  voxel for this thread.
  int        n_vox; // Number of voxels for this thread.
  int          vox; // Index of current voxel.

  int sum_part = 0;

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

  // load_8x1( &pb0[0].w[0], q );

  //--------------------------------------------------------------------------//
  // Determine the first and last voxel for each pipeline and the number of
  // voxels for each pipeline.
  //--------------------------------------------------------------------------//

  int ix = 0;
  int iy = 0;
  int iz = 0;

  int n_voxel = 0; // Number of voxels in this MPI domain.

  int first_ix = 0;
  int first_iy = 0;
  int first_iz = 0;

  first_vox = 0;
  last_vox  = 0;
  n_vox     = 0;

  if ( n_part > 0 )
  {
    first_vox = 2*sp->g->nv; // Initialize with invalid values.
    last_vox  = 2*sp->g->nv;

    DISTRIBUTE_VOXELS( 1, sp->g->nx,
                       1, sp->g->ny,
                       1, sp->g->nz,
                       1,
                       0,
                       1,
                       ix, iy, iz, n_voxel );

    vox = VOXEL( ix, iy, iz, sp->g->nx, sp->g->ny, sp->g->nz );

    for( int i = 0; i < n_voxel; i++ )
    {
      sum_part += sp->counts[vox];

      if ( sum_part >= last_part )
      {
        if ( pipeline_rank == n_pipeline - 1 )
        {
          last_vox = vox;

          n_vox++;
        }
        else
        {
          last_vox = previous_vox;
        }

        break;
      }
      else if ( sum_part >= first_part   &&
                first_vox == 2*sp->g->nv )
      {
        first_vox = vox;
        first_ix  = ix;
        first_iy  = iy;
        first_iz  = iz;
      }

      if ( vox >= first_vox )
      {
        n_vox++;
      }

      previous_vox = vox;

      NEXT_VOXEL( vox, ix, iy, iz,
                  1, sp->g->nx,
                  1, sp->g->ny,
                  1, sp->g->nz,
                  sp->g->nx,
                  sp->g->ny,
                  sp->g->nz );
    }
  }

  //--------------------------------------------------------------------------//
  // Loop over voxels.
  //--------------------------------------------------------------------------//

  ix = first_ix;
  iy = first_iy;
  iz = first_iz;

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

      // Initialize cell local accumulator vectors to zero.
      jx0 = 0.0;
      jx1 = 0.0;
      jx2 = 0.0;
      jx3 = 0.0;

      jy0 = 0.0;
      jy1 = 0.0;
      jy2 = 0.0;
      jy3 = 0.0;

      jz0 = 0.0;
      jz1 = 0.0;
      jz2 = 0.0;
      jz3 = 0.0;

      // Initialize particle pointer to first particle in cell.
      pb = args->pb0 + part_start / PARTICLE_BLOCK_SIZE;

      int ib = 0;

      for( int i = 0; i < part_count; i += PARTICLE_BLOCK_SIZE )
      {
        ib = i / PARTICLE_BLOCK_SIZE;          // Index of particle block.

        //--------------------------------------------------------------------------
        // Load particle position and momentum.
        //--------------------------------------------------------------------------

        load_8x1( &pb[ib].dx[0], dx );
        load_8x1( &pb[ib].dy[0], dy );
        load_8x1( &pb[ib].dz[0], dz );

        load_8x1( &pb[ib].ux[0], ux );
        load_8x1( &pb[ib].uy[0], uy );
        load_8x1( &pb[ib].uz[0], uz );

        load_8x1( &pb[ib].w [0], q  );

        //--------------------------------------------------------------------------
        // Interpolate E.
        //--------------------------------------------------------------------------

        hax = qdt_2mc * fma( fma( dy, d2exdydz, dexdz ), dz, fma( dy, dexdy, ex ) );
        hay = qdt_2mc * fma( fma( dz, d2eydzdx, deydx ), dx, fma( dz, deydz, ey ) );
        haz = qdt_2mc * fma( fma( dx, d2ezdxdy, dezdy ), dy, fma( dx, dezdx, ez ) );

        //--------------------------------------------------------------------------
        // Interpolate B.
        //--------------------------------------------------------------------------

        cbxp = fma( dcbxdx, dx, cbx );
        cbyp = fma( dcbydy, dy, cby );
        cbzp = fma( dcbzdz, dz, cbz );

        //--------------------------------------------------------------------------
        // Update momentum.
        //--------------------------------------------------------------------------
        // For a 5-10% performance hit, v00 = qdt_2mc/sqrt(blah) is a few ulps more
        // accurate (but still quite in the noise numerically) for cyclotron
        // frequencies approaching the nyquist frequency.
        //--------------------------------------------------------------------------

        //--------------------------------------------------------------------------
        // Half advance E.
        //--------------------------------------------------------------------------

        ux  += hax;
        uy  += hay;
        uz  += haz;

        //--------------------------------------------------------------------------
        // Boris - scalars.
        //--------------------------------------------------------------------------

        v00  = qdt_2mc * rsqrt( one + fma( ux, ux, fma( uy, uy, uz * uz ) ) );
        v01  = fma( cbxp, cbxp, fma( cbyp, cbyp, cbzp * cbzp ) );
        v02  = ( v00 * v00 ) * v01;
        v03  = v00 * fma( fma( two_fifteenths, v02, one_third ), v02, one );
        v04  = v03 * rcp( fma( v03 * v03, v01, one ) );
        v04 += v04;

        //--------------------------------------------------------------------------
        // Boris - uprime.
        //--------------------------------------------------------------------------

        v00  = fma( fms(  uy, cbzp,  uz * cbyp ), v03, ux );
        v01  = fma( fms(  uz, cbxp,  ux * cbzp ), v03, uy );
        v02  = fma( fms(  ux, cbyp,  uy * cbxp ), v03, uz );

        //--------------------------------------------------------------------------
        // Boris - rotation.
        //--------------------------------------------------------------------------

        ux   = fma( fms( v01, cbzp, v02 * cbyp ), v04, ux );
        uy   = fma( fms( v02, cbxp, v00 * cbzp ), v04, uy );
        uz   = fma( fms( v00, cbyp, v01 * cbxp ), v04, uz );

        //--------------------------------------------------------------------------
        // Half advance E.
        //--------------------------------------------------------------------------

        ux  += hax;
        uy  += hay;
        uz  += haz;

        //--------------------------------------------------------------------------
        // Store particle momentum.
        //--------------------------------------------------------------------------

        store_8x1( ux, &pb[ib].ux[0] );
        store_8x1( uy, &pb[ib].uy[0] );
        store_8x1( uz, &pb[ib].uz[0] );

        //--------------------------------------------------------------------------
        // Update the position of in bound particles.
        //--------------------------------------------------------------------------

        v00 = rsqrt( one + fma( ux, ux, fma( uy, uy, uz * uz ) ) );

        ux *= cdt_dx;
        uy *= cdt_dy;
        uz *= cdt_dz;

        //--------------------------------------------------------------------------
        // ux, uy, and uz are normalized displacement relative to cell size.
        //--------------------------------------------------------------------------

        ux *= v00;
        uy *= v00;
        uz *= v00;

        //--------------------------------------------------------------------------
        // New particle midpoint.
        //--------------------------------------------------------------------------

        v00 =  dx + ux;
        v01 =  dy + uy;
        v02 =  dz + uz;

        //--------------------------------------------------------------------------
        // New particle position.
        //--------------------------------------------------------------------------

        v03 = v00 + ux;
        v04 = v01 + uy;
        v05 = v02 + uz;

        //--------------------------------------------------------------------------
        // Determine which particles are out of bounds.
        //--------------------------------------------------------------------------

        outbnd = ( v03 > one ) | ( v03 < neg_one ) |
                 ( v04 > one ) | ( v04 < neg_one ) |
                 ( v05 > one ) | ( v05 < neg_one );

        //--------------------------------------------------------------------------
        // Do not update outbnd particles.
        //--------------------------------------------------------------------------

        v03 = merge( outbnd, dx, v03 );
        v04 = merge( outbnd, dy, v04 );
        v05 = merge( outbnd, dz, v05 );

        //--------------------------------------------------------------------------
        // Store particle position.
        //--------------------------------------------------------------------------

        store_8x1( v03, &pb[ib].dx[0] );
        store_8x1( v04, &pb[ib].dy[0] );
        store_8x1( v05, &pb[ib].dz[0] );

        //--------------------------------------------------------------------------
        // Accumulate current of inbnd particles.
        // Note: accumulator values are 4 times the total physical charge that
        // passed through the appropriate current quadrant in a time-step.
        // Do not accumulate outbnd particles.
        //--------------------------------------------------------------------------

        q  = czero( outbnd, q * qsp );

        //--------------------------------------------------------------------------
        // Streak midpoint. Valid for inbnd only.
        //--------------------------------------------------------------------------

        dx = v00;
        dy = v01;
        dz = v02;

        //--------------------------------------------------------------------------
        // Charge conservation correction.
        //--------------------------------------------------------------------------

        v15 = q * ux * uy * uz * one_third;

        //--------------------------------------------------------------------------
        // Accumulate current density.
        //--------------------------------------------------------------------------
        // Accumulate Jx for 8 particles into the v0 - v3 vectors.
        //--------------------------------------------------------------------------

        v12  =   q * ux;   // v12 = q ux
        v01  = v12 * dy;   // v01 = q ux dy
        v00  = v12 - v01;  // v00 = q ux (1-dy)
        v01 += v12;        // v01 = q ux (1+dy)

        v13  = one + dz;   // v13 = 1+dz
        v02  = v00 * v13;  // v02 = q ux (1-dy)(1+dz)
        v03  = v01 * v13;  // v03 = q ux (1+dy)(1+dz)

        v14  = one - dz;   // v14 = 1-dz
        v00 *= v14;        // v00 = q ux (1-dy)(1-dz)
        v01 *= v14;        // v01 = q ux (1+dy)(1-dz)

        v00 += v15;        // v00 = q ux [ (1-dy)(1-dz) + uy*uz/3 ]
        v01 -= v15;        // v01 = q ux [ (1+dy)(1-dz) - uy*uz/3 ]
        v02 -= v15;        // v02 = q ux [ (1-dy)(1+dz) - uy*uz/3 ]
        v03 += v15;        // v03 = q ux [ (1+dy)(1+dz) + uy*uz/3 ]

        jx0 += v00;
        jx1 += v01;
        jx2 += v02;
        jx3 += v03;

        //--------------------------------------------------------------------------
        // Accumulate Jy for 8 particles into the v4 - v7 vectors.
        //--------------------------------------------------------------------------

        v12  =   q * uy;   // v12 = q uy
        v05  = v12 * dz;   // v05 = q uy dz
        v04  = v12 - v05;  // v04 = q uy (1-dz)
        v05 += v12;        // v05 = q uy (1+dz)

        v13  = one + dx;   // v13 = 1+dx
        v06  = v04 * v13;  // v06 = q uy (1-dz)(1+dx)
        v07  = v05 * v13;  // v07 = q uy (1+dz)(1+dx)

        v14  = one - dx;   // v14 = 1-dx
        v04 *= v14;        // v04 = q uy (1-dz)(1-dx)
        v05 *= v14;        // v05 = q uy (1+dz)(1-dx)

        v04 += v15;        // v04 = q uy [ (1-dz)(1-dx) + ux*uz/3 ]
        v05 -= v15;        // v05 = q uy [ (1+dz)(1-dx) - ux*uz/3 ]
        v06 -= v15;        // v06 = q uy [ (1-dz)(1+dx) - ux*uz/3 ]
        v07 += v15;        // v07 = q uy [ (1+dz)(1+dx) + ux*uz/3 ]

        jy0 += v04;
        jy1 += v05;
        jy2 += v06;
        jy3 += v07;

        //--------------------------------------------------------------------------
        // Accumulate Jz for 8 particles into the v8 - v11 vectors.
        //--------------------------------------------------------------------------

        v12  =   q * uz;   // v12 = q uz
        v09  = v12 * dx;   // v09 = q uz dx
        v08  = v12 - v09;  // v08 = q uz (1-dx)
        v09 += v12;        // v09 = q uz (1+dx)

        v13  = one + dy;   // v13 = 1+dy
        v10  = v08 * v13;  // v10 = q uz (1-dx)(1+dy)
        v11  = v09 * v13;  // v11 = q uz (1+dx)(1+dy)

        v14  = one - dy;   // v14 = 1-dy
        v08 *= v14;        // v08 = q uz (1-dx)(1-dy)
        v09 *= v14;        // v09 = q uz (1+dx)(1-dy)

        v08 += v15;        // v08 = q uz [ (1-dx)(1-dy) + ux*uy/3 ]
        v09 -= v15;        // v09 = q uz [ (1+dx)(1-dy) - ux*uy/3 ]
        v10 -= v15;        // v10 = q uz [ (1-dx)(1+dy) - ux*uy/3 ]
        v11 += v15;        // v11 = q uz [ (1+dx)(1+dy) + ux*uy/3 ]

        jz0 += v08;
        jz1 += v09;
        jz2 += v10;
        jz3 += v11;

        //--------------------------------------------------------------------------
        // Update position and accumulate current density for out of bounds
        // particles.
        //--------------------------------------------------------------------------

        // #define MOVE_OUTBND(N)                                              \
        // if ( outbnd(N) )                                  /* Unlikely */    \
        // {                                                                   \
        //   local_pm->dispx = ux(N);                                          \
        //   local_pm->dispy = uy(N);                                          \
        //   local_pm->dispz = uz(N);                                          \
        //   local_pm->i     = ipart + N;                                      \
        //   if ( move_p( pb0, local_pm, a0, g, _qsp, sp ) ) /* Unlikely */    \
        //   {                                                                 \
        //     if ( nm < max_nm )                                              \
        //     {                                                               \
        //       v4::copy_4x1( &pm[nm++], local_pm );                          \
        //     }                                                               \
        //     else                                          /* Unlikely */    \
        //     {                                                               \
        //       itmp++;                                                       \
        //     }                                                               \
        //   }                                                                 \
        // }

        #define MOVE_OUTBND(N)                                              \
        if ( outbnd(N) )                                  /* Unlikely */    \
        {                                                                   \
          local_pm->dispx = ux(N);                                          \
          local_pm->dispy = uy(N);                                          \
          local_pm->dispz = uz(N);                                          \
          local_pm->i     = part_start + i + N;                             \
          if ( move_p( pb0, local_pm, a0, g, _qsp, sp ) ) /* Unlikely */    \
          {                                                                 \
            if ( nm < max_nm )                                              \
            {                                                               \
              v4::copy_4x1( &pm[nm++], local_pm );                          \
            }                                                               \
            else                                          /* Unlikely */    \
            {                                                               \
              itmp++;                                                       \
            }                                                               \
          }                                                                 \
        }

        MOVE_OUTBND( 0);
        MOVE_OUTBND( 1);
        MOVE_OUTBND( 2);
        MOVE_OUTBND( 3);
        MOVE_OUTBND( 4);
        MOVE_OUTBND( 5);
        MOVE_OUTBND( 6);
        MOVE_OUTBND( 7);

        #undef MOVE_OUTBND
      }
    }

    // Add cell local current density to accumulator array. Can this be done with
    // instrinsics? And, would it be faster?

    float * ALIGNED(64) p_a0 = ( float * ALIGNED(64) ) ( a0 + vox );

    for( int j = 0; j < PARTICLE_BLOCK_SIZE; j++ )
    {
      p_a0[ 0] += jx0[j];
      p_a0[ 1] += jx1[j];
      p_a0[ 2] += jx2[j];
      p_a0[ 3] += jx3[j];

      p_a0[ 4] += jy0[j];
      p_a0[ 5] += jy1[j];
      p_a0[ 6] += jy2[j];
      p_a0[ 7] += jy3[j];

      p_a0[ 8] += jz0[j];
      p_a0[ 9] += jz1[j];
      p_a0[10] += jz2[j];
      p_a0[11] += jz3[j];
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

  args->seg[pipeline_rank].pm        = pm;
  args->seg[pipeline_rank].max_nm    = max_nm;
  args->seg[pipeline_rank].nm        = nm;
  args->seg[pipeline_rank].n_ignored = itmp;
}

#else // VPIC_USE_AOSOA_P is not defined i.e. VPIC_USE_AOS_P case.

void
advance_p_pipeline_v8( advance_p_pipeline_args_t * args,
                       int pipeline_rank,
                       int n_pipeline )
{
  particle_t           * ALIGNED(128) p0 = args->p0;
  accumulator_t        * ALIGNED(128) a0 = args->a0;
  const interpolator_t * ALIGNED(128) f0 = args->f0;
  const grid_t         *              g  = args->g;
  /* */ species_t      *              sp = args->sp;

  particle_t           * ALIGNED(128) p;
  particle_mover_t     * ALIGNED(16)  pm;

  float                * ALIGNED(32)  vp00;
  float                * ALIGNED(32)  vp01;
  float                * ALIGNED(32)  vp02;
  float                * ALIGNED(32)  vp03;
  float                * ALIGNED(32)  vp04;
  float                * ALIGNED(32)  vp05;
  float                * ALIGNED(32)  vp06;
  float                * ALIGNED(32)  vp07;

  // Basic constants.
  const v8float qdt_2mc(args->qdt_2mc);
  const v8float cdt_dx(args->cdt_dx);
  const v8float cdt_dy(args->cdt_dy);
  const v8float cdt_dz(args->cdt_dz);
  const v8float qsp(args->qsp);
  const v8float one(1.0);
  const v8float one_third(1.0/3.0);
  const v8float two_fifteenths(2.0/15.0);
  const v8float neg_one(-1.0);

  const float _qsp = args->qsp;

  v8float ex, dexdy, dexdz, d2exdydz;
  v8float ey, deydz, deydx, d2eydzdx;
  v8float ez, dezdx, dezdy, d2ezdxdy;

  v8float cbx, dcbxdx;
  v8float cby, dcbydy;
  v8float cbz, dcbzdz;

  v8float jx0, jx1, jx2, jx3;
  v8float jy0, jy1, jy2, jy3;
  v8float jz0, jz1, jz2, jz3;

  v8float dx, dy, dz;
  v8float ux, uy, uz;
  v8float q;
  v8float hax, hay, haz;
  v8float cbxp, cbyp, cbzp;
  v8float v00, v01, v02, v03, v04, v05, v06, v07;
  v8float v08, v09, v10, v11, v12, v13, v14, v15;
  v8int   ii, outbnd;

  int itmp, nm, max_nm;

  int first_part; // Index of first particle for this thread.
  int  last_part; // Index of last  particle for this thread.
  int     n_part; // Number of particles for this thread.

  int previous_vox; // Index of previous voxel.
  int    first_vox; // Index of first voxel for this thread.
  int     last_vox; // Index of last  voxel for this thread.
  int        n_vox; // Number of voxels for this thread.
  int          vox; // Index of current voxel.

  int sum_part = 0;

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
  // Determine the first and last voxel for each pipeline and the number of
  // voxels for each pipeline.
  //--------------------------------------------------------------------------//

  int ix = 0;
  int iy = 0;
  int iz = 0;

  int n_voxel = 0; // Number of voxels in this MPI domain.

  int first_ix = 0;
  int first_iy = 0;
  int first_iz = 0;

  first_vox = 0;
  last_vox  = 0;
  n_vox     = 0;

  if ( n_part > 0 )
  {
    first_vox = 2*sp->g->nv; // Initialize with invalid values.
    last_vox  = 2*sp->g->nv;

    DISTRIBUTE_VOXELS( 1, sp->g->nx,
                       1, sp->g->ny,
                       1, sp->g->nz,
                       1,
                       0,
                       1,
                       ix, iy, iz, n_voxel );

    vox = VOXEL( ix, iy, iz, sp->g->nx, sp->g->ny, sp->g->nz );

    for( int i = 0; i < n_voxel; i++ )
    {
      sum_part += sp->counts[vox];

      if ( sum_part >= last_part )
      {
        if ( pipeline_rank == n_pipeline - 1 )
        {
          last_vox = vox;

          n_vox++;
        }
        else
        {
          last_vox = previous_vox;
        }

        break;
      }
      else if ( sum_part >= first_part   &&
                first_vox == 2*sp->g->nv )
      {
        first_vox = vox;
        first_ix  = ix;
        first_iy  = iy;
        first_iz  = iz;
      }

      if ( vox >= first_vox )
      {
        n_vox++;
      }

      previous_vox = vox;

      NEXT_VOXEL( vox, ix, iy, iz,
                  1, sp->g->nx,
                  1, sp->g->ny,
                  1, sp->g->nz,
                  sp->g->nx,
                  sp->g->ny,
                  sp->g->nz );
    }
  }

  //--------------------------------------------------------------------------//
  // Loop over voxels.
  //--------------------------------------------------------------------------//

  ix = first_ix;
  iy = first_iy;
  iz = first_iz;

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

      // Initialize cell local accumulator vectors to zero.
      jx0 = 0.0;
      jx1 = 0.0;
      jx2 = 0.0;
      jx3 = 0.0;

      jy0 = 0.0;
      jy1 = 0.0;
      jy2 = 0.0;
      jy3 = 0.0;

      jz0 = 0.0;
      jz1 = 0.0;
      jz2 = 0.0;
      jz3 = 0.0;

      // Initialize particle pointer to first particle in cell.
      p = args->p0 + part_start;

      // Process the particles in a cell.
      for( int i = 0; i < part_count; i += 8, p += 8 )
      {
        //--------------------------------------------------------------------------
        // Load particle data.
        //--------------------------------------------------------------------------

        load_8x8_tr( &p[0].dx, &p[1].dx, &p[2].dx, &p[3].dx,
                     &p[4].dx, &p[5].dx, &p[6].dx, &p[7].dx,
                     dx, dy, dz, ii, ux, uy, uz, q );

        //--------------------------------------------------------------------------
        // Interpolate E.
        //--------------------------------------------------------------------------

        hax = qdt_2mc * fma( fma( dy, d2exdydz, dexdz ), dz, fma( dy, dexdy, ex ) );
        hay = qdt_2mc * fma( fma( dz, d2eydzdx, deydx ), dx, fma( dz, deydz, ey ) );
        haz = qdt_2mc * fma( fma( dx, d2ezdxdy, dezdy ), dy, fma( dx, dezdx, ez ) );

        //--------------------------------------------------------------------------
        // Interpolate B.
        //--------------------------------------------------------------------------

        cbxp = fma( dcbxdx, dx, cbx );
        cbyp = fma( dcbydy, dy, cby );
        cbzp = fma( dcbzdz, dz, cbz );

        //--------------------------------------------------------------------------
        // Update momentum.
        //--------------------------------------------------------------------------
        // For a 5-10% performance hit, v00 = qdt_2mc/sqrt(blah) is a few ulps more
        // accurate (but still quite in the noise numerically) for cyclotron
        // frequencies approaching the nyquist frequency.
        //--------------------------------------------------------------------------

        //--------------------------------------------------------------------------
        // Half advance E.
        //--------------------------------------------------------------------------

        ux  += hax;
        uy  += hay;
        uz  += haz;

        //--------------------------------------------------------------------------
        // Boris - scalars.
        //--------------------------------------------------------------------------

        v00  = qdt_2mc * rsqrt( one + fma( ux, ux, fma( uy, uy, uz * uz ) ) );
        v01  = fma( cbxp, cbxp, fma( cbyp, cbyp, cbzp * cbzp ) );
        v02  = ( v00 * v00 ) * v01;
        v03  = v00 * fma( fma( two_fifteenths, v02, one_third ), v02, one );
        v04  = v03 * rcp( fma( v03 * v03, v01, one ) );
        v04 += v04;

        //--------------------------------------------------------------------------
        // Boris - uprime.
        //--------------------------------------------------------------------------

        v00  = fma( fms(  uy, cbzp,  uz * cbyp ), v03, ux );
        v01  = fma( fms(  uz, cbxp,  ux * cbzp ), v03, uy );
        v02  = fma( fms(  ux, cbyp,  uy * cbxp ), v03, uz );

        //--------------------------------------------------------------------------
        // Boris - rotation.
        //--------------------------------------------------------------------------

        ux   = fma( fms( v01, cbzp, v02 * cbyp ), v04, ux );
        uy   = fma( fms( v02, cbxp, v00 * cbzp ), v04, uy );
        uz   = fma( fms( v00, cbyp, v01 * cbxp ), v04, uz );

        //--------------------------------------------------------------------------
        // Half advance E.
        //--------------------------------------------------------------------------

        ux  += hax;
        uy  += hay;
        uz  += haz;

        //--------------------------------------------------------------------------
        // Store ux, uy, uz in v06, v07, v08 so particle velocity store can be done
        // later with the particle positions.
        //--------------------------------------------------------------------------

        v06  = ux;
        v07  = uy;
        v08  = uz;

        //--------------------------------------------------------------------------
        // Update the position of in bound particles.
        //--------------------------------------------------------------------------

        v00 = rsqrt( one + fma( ux, ux, fma( uy, uy, uz * uz ) ) );

        ux *= cdt_dx;
        uy *= cdt_dy;
        uz *= cdt_dz;

        //--------------------------------------------------------------------------
        // ux, uy, and uz are normalized displacement relative to cell size.
        //--------------------------------------------------------------------------

        ux *= v00;
        uy *= v00;
        uz *= v00;

        //--------------------------------------------------------------------------
        // New particle midpoint.
        //--------------------------------------------------------------------------

        v00 =  dx + ux;
        v01 =  dy + uy;
        v02 =  dz + uz;

        //--------------------------------------------------------------------------
        // New particle position.
        //--------------------------------------------------------------------------

        v03 = v00 + ux;
        v04 = v01 + uy;
        v05 = v02 + uz;

        //--------------------------------------------------------------------------
        // Determine which particles are out of bounds.
        //--------------------------------------------------------------------------

        outbnd = ( v03 > one ) | ( v03 < neg_one ) |
                 ( v04 > one ) | ( v04 < neg_one ) |
                 ( v05 > one ) | ( v05 < neg_one );

        //--------------------------------------------------------------------------
        // Do not update outbnd particles.
        //--------------------------------------------------------------------------

        v03 = merge( outbnd, dx, v03 );
        v04 = merge( outbnd, dy, v04 );
        v05 = merge( outbnd, dz, v05 );

        //--------------------------------------------------------------------------
        // Store particle data.
        //--------------------------------------------------------------------------

        store_8x8_tr( v03, v04, v05, ii, v06, v07, v08, q,
                      &p[0].dx, &p[1].dx, &p[2].dx, &p[3].dx,
                      &p[4].dx, &p[5].dx, &p[6].dx, &p[7].dx );

        //--------------------------------------------------------------------------
        // Accumulate current of inbnd particles.
        // Note: accumulator values are 4 times the total physical charge that
        // passed through the appropriate current quadrant in a time-step.
        // Do not accumulate outbnd particles.
        //--------------------------------------------------------------------------

        q  = czero( outbnd, q * qsp );

        //--------------------------------------------------------------------------
        // Streak midpoint. Valid for inbnd only.
        //--------------------------------------------------------------------------

        dx = v00;
        dy = v01;
        dz = v02;

        //--------------------------------------------------------------------------
        // Charge conservation correction.
        //--------------------------------------------------------------------------

        v15 = q * ux * uy * uz * one_third;

        //--------------------------------------------------------------------------
        // Accumulate current density.
        //--------------------------------------------------------------------------
        // Accumulate Jx for 8 particles into the v0 - v3 vectors.
        //--------------------------------------------------------------------------

        v12  =   q * ux;   // v12 = q ux
        v01  = v12 * dy;   // v01 = q ux dy
        v00  = v12 - v01;  // v00 = q ux (1-dy)
        v01 += v12;        // v01 = q ux (1+dy)

        v13  = one + dz;   // v13 = 1+dz
        v02  = v00 * v13;  // v02 = q ux (1-dy)(1+dz)
        v03  = v01 * v13;  // v03 = q ux (1+dy)(1+dz)

        v14  = one - dz;   // v14 = 1-dz
        v00 *= v14;        // v00 = q ux (1-dy)(1-dz)
        v01 *= v14;        // v01 = q ux (1+dy)(1-dz)

        v00 += v15;        // v00 = q ux [ (1-dy)(1-dz) + uy*uz/3 ]
        v01 -= v15;        // v01 = q ux [ (1+dy)(1-dz) - uy*uz/3 ]
        v02 -= v15;        // v02 = q ux [ (1-dy)(1+dz) - uy*uz/3 ]
        v03 += v15;        // v03 = q ux [ (1+dy)(1+dz) + uy*uz/3 ]

        jx0 += v00;
        jx1 += v01;
        jx2 += v02;
        jx3 += v03;

        //--------------------------------------------------------------------------
        // Accumulate Jy for 8 particles into the v4 - v7 vectors.
        //--------------------------------------------------------------------------

        v12  =   q * uy;   // v12 = q uy
        v05  = v12 * dz;   // v05 = q uy dz
        v04  = v12 - v05;  // v04 = q uy (1-dz)
        v05 += v12;        // v05 = q uy (1+dz)

        v13  = one + dx;   // v13 = 1+dx
        v06  = v04 * v13;  // v06 = q uy (1-dz)(1+dx)
        v07  = v05 * v13;  // v07 = q uy (1+dz)(1+dx)

        v14  = one - dx;   // v14 = 1-dx
        v04 *= v14;        // v04 = q uy (1-dz)(1-dx)
        v05 *= v14;        // v05 = q uy (1+dz)(1-dx)

        v04 += v15;        // v04 = q uy [ (1-dz)(1-dx) + ux*uz/3 ]
        v05 -= v15;        // v05 = q uy [ (1+dz)(1-dx) - ux*uz/3 ]
        v06 -= v15;        // v06 = q uy [ (1-dz)(1+dx) - ux*uz/3 ]
        v07 += v15;        // v07 = q uy [ (1+dz)(1+dx) + ux*uz/3 ]

        jy0 += v04;
        jy1 += v05;
        jy2 += v06;
        jy3 += v07;

        //--------------------------------------------------------------------------
        // Accumulate Jz for 8 particles into the v8 - v11 vectors.
        //--------------------------------------------------------------------------

        v12  =   q * uz;   // v12 = q uz
        v09  = v12 * dx;   // v09 = q uz dx
        v08  = v12 - v09;  // v08 = q uz (1-dx)
        v09 += v12;        // v09 = q uz (1+dx)

        v13  = one + dy;   // v13 = 1+dy
        v10  = v08 * v13;  // v10 = q uz (1-dx)(1+dy)
        v11  = v09 * v13;  // v11 = q uz (1+dx)(1+dy)

        v14  = one - dy;   // v14 = 1-dy
        v08 *= v14;        // v08 = q uz (1-dx)(1-dy)
        v09 *= v14;        // v09 = q uz (1+dx)(1-dy)

        v08 += v15;        // v08 = q uz [ (1-dx)(1-dy) + ux*uy/3 ]
        v09 -= v15;        // v09 = q uz [ (1+dx)(1-dy) - ux*uy/3 ]
        v10 -= v15;        // v10 = q uz [ (1-dx)(1+dy) - ux*uy/3 ]
        v11 += v15;        // v11 = q uz [ (1+dx)(1+dy) + ux*uy/3 ]

        jz0 += v08;
        jz1 += v09;
        jz2 += v10;
        jz3 += v11;

        //--------------------------------------------------------------------------
        // Update position and accumulate current density for out of bounds
        // particles.
        //--------------------------------------------------------------------------

        // #define MOVE_OUTBND(N)                                              \
        // if ( outbnd(N) )                                 /* Unlikely */     \
        // {                                                                   \
        //   local_pm->dispx = ux(N);                                          \
        //   local_pm->dispy = uy(N);                                          \
        //   local_pm->dispz = uz(N);                                          \
        //   local_pm->i     = ( p - p0 ) + N;                                 \
        //   if ( move_p( p0, local_pm, a0, g, _qsp, sp ) ) /* Unlikely */     \
        //   {                                                                 \
        //     if ( nm < max_nm )                                              \
        //     {                                                               \
        //       v4::copy_4x1( &pm[nm++], local_pm );                          \
        //     }                                                               \
        //     else                                         /* Unlikely */     \
        //     {                                                               \
        //       itmp++;                                                       \
        //     }                                                               \
        //   }                                                                 \
        // }

        #define MOVE_OUTBND(N)                                              \
        if ( outbnd(N) )                                 /* Unlikely */     \
        {                                                                   \
          local_pm->dispx = ux(N);                                          \
          local_pm->dispy = uy(N);                                          \
          local_pm->dispz = uz(N);                                          \
          local_pm->i     = part_start + i + N;                             \
          if ( move_p( p0, local_pm, a0, g, _qsp, sp ) ) /* Unlikely */     \
          {                                                                 \
            if ( nm < max_nm )                                              \
            {                                                               \
              v4::copy_4x1( &pm[nm++], local_pm );                          \
            }                                                               \
            else                                         /* Unlikely */     \
            {                                                               \
              itmp++;                                                       \
            }                                                               \
          }                                                                 \
        }

        MOVE_OUTBND( 0);
        MOVE_OUTBND( 1);
        MOVE_OUTBND( 2);
        MOVE_OUTBND( 3);
        MOVE_OUTBND( 4);
        MOVE_OUTBND( 5);
        MOVE_OUTBND( 6);
        MOVE_OUTBND( 7);

        #undef MOVE_OUTBND
      }
    }

    // Add cell local current density to accumulator array. Can this be done with
    // instrinsics? And, would it be faster?

    float * ALIGNED(64) p_a0 = ( float * ALIGNED(64) ) ( a0 + vox );

    for( int j = 0; j < 8; j++ )
    {
      p_a0[ 0] += jx0[j];
      p_a0[ 1] += jx1[j];
      p_a0[ 2] += jx2[j];
      p_a0[ 3] += jx3[j];

      p_a0[ 4] += jy0[j];
      p_a0[ 5] += jy1[j];
      p_a0[ 6] += jy2[j];
      p_a0[ 7] += jy3[j];

      p_a0[ 8] += jz0[j];
      p_a0[ 9] += jz1[j];
      p_a0[10] += jz2[j];
      p_a0[11] += jz3[j];
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

  args->seg[pipeline_rank].pm        = pm;
  args->seg[pipeline_rank].max_nm    = max_nm;
  args->seg[pipeline_rank].nm        = nm;
  args->seg[pipeline_rank].n_ignored = itmp;
}

#endif // End of VPIC_USE_AOSOA_P vs VPIC_USE_AOS_P selection.

#else // V8_ACCELERATION is not defined i.e. use stub function.

void
advance_p_pipeline_v8( advance_p_pipeline_args_t * args,
                       int pipeline_rank,
                       int n_pipeline )
{
  // No v8 implementation.
  ERROR( ( "No advance_p_pipeline_v8 implementation." ) );
}

#endif // End of V8_ACCELERATION selection.
