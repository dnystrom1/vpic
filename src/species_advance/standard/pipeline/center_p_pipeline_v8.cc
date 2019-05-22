#define IN_spa

#include "spa_private.h"

#if defined(V8_ACCELERATION)

using namespace v8;

#if defined(VPIC_USE_AOSOA_P)
void
center_p_pipeline_v8( center_p_pipeline_args_t * args,
                      int pipeline_rank,
                      int n_pipeline )
{
  const interpolator_t * ALIGNED(128) f0 = args->f0;

  particle_block_t     * ALIGNED(128) pb;

  const float          * ALIGNED(32)  vp00;
  const float          * ALIGNED(32)  vp01;
  const float          * ALIGNED(32)  vp02;
  const float          * ALIGNED(32)  vp03;
  const float          * ALIGNED(32)  vp04;
  const float          * ALIGNED(32)  vp05;
  const float          * ALIGNED(32)  vp06;
  const float          * ALIGNED(32)  vp07;

  const v8float qdt_2mc(    args->qdt_2mc);
  const v8float qdt_4mc(0.5*args->qdt_2mc); // For half Boris rotate
  const v8float one(1.0);
  const v8float one_third(1.0/3.0);
  const v8float two_fifteenths(2.0/15.0);

  v8float dx, dy, dz, ux, uy, uz, q;
  v8float hax, hay, haz, cbx, cby, cbz;
  v8float v00, v01, v02, v03, v04, v05;
  v8int   ii;

  int itmp, nq;

  // Determine which particle blocks this pipeline processes.

  DISTRIBUTE( args->np, PARTICLE_BLOCK_SIZE, pipeline_rank, n_pipeline, itmp, nq );

  pb = args->pb0 + itmp / PARTICLE_BLOCK_SIZE;

  nq >>= 3;

  // Process the particle blocks for this pipeline.  For the initial AoSoA version,
  // assume that PARTICLE_BLOCK_SIZE is the same as the vector length.  A future
  // version might allow PARTICLE_BLOCK_SIZE to be an integral multiple of the vector
  // length.

  for( ; nq; nq--, pb++ )
  {
    //--------------------------------------------------------------------------
    // Load particle position data.
    //--------------------------------------------------------------------------
    load_8x1( &pb[0].dx[0], dx );
    load_8x1( &pb[0].dy[0], dy );
    load_8x1( &pb[0].dz[0], dz );
    load_8x1( &pb[0].i [0], ii );

    //--------------------------------------------------------------------------
    // Set field interpolation pointers.
    //--------------------------------------------------------------------------
    vp00 = ( const float * ALIGNED(32) ) ( f0 + ii(0) );
    vp01 = ( const float * ALIGNED(32) ) ( f0 + ii(1) );
    vp02 = ( const float * ALIGNED(32) ) ( f0 + ii(2) );
    vp03 = ( const float * ALIGNED(32) ) ( f0 + ii(3) );
    vp04 = ( const float * ALIGNED(32) ) ( f0 + ii(4) );
    vp05 = ( const float * ALIGNED(32) ) ( f0 + ii(5) );
    vp06 = ( const float * ALIGNED(32) ) ( f0 + ii(6) );
    vp07 = ( const float * ALIGNED(32) ) ( f0 + ii(7) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_8x4_tr( vp00, vp01, vp02, vp03,
		 vp04, vp05, vp06, vp07,
		 hax, v00, v01, v02 );

    hax = qdt_2mc*fma( fma( dy, v02, v01 ), dz, fma( dy, v00, hax ) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_8x4_tr( vp00+4, vp01+4, vp02+4, vp03+4,
		 vp04+4, vp05+4, vp06+4, vp07+4,
		 hay, v03, v04, v05 );

    hay = qdt_2mc*fma( fma( dz, v05, v04 ), dx, fma( dz, v03, hay ) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_8x4_tr( vp00+8, vp01+8, vp02+8, vp03+8,
		 vp04+8, vp05+8, vp06+8, vp07+8,
		 haz, v00, v01, v02 );

    haz = qdt_2mc*fma( fma( dx, v02, v01 ), dy, fma( dx, v00, haz ) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_8x4_tr( vp00+12, vp01+12, vp02+12, vp03+12,
		 vp04+12, vp05+12, vp06+12, vp07+12,
		 cbx, v03, cby, v04 );

    cbx = fma( v03, dx, cbx );
    cby = fma( v04, dy, cby );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles, final.
    //--------------------------------------------------------------------------
    load_8x2_tr( vp00+16, vp01+16, vp02+16, vp03+16,
		 vp04+16, vp05+16, vp06+16, vp07+16,
		 cbz, v05 );

    cbz = fma( v05, dz, cbz );

    //--------------------------------------------------------------------------
    // Load particle momentum data.
    //--------------------------------------------------------------------------
    load_8x1( &pb[0].ux[0], ux );
    load_8x1( &pb[0].uy[0], uy );
    load_8x1( &pb[0].uz[0], uz );

    // load_8x4_tr( &p[0].ux, &p[1].ux, &p[2].ux, &p[3].ux,
    //              &p[4].ux, &p[5].ux, &p[6].ux, &p[7].ux,
    //              ux, uy, uz, q );

    //--------------------------------------------------------------------------
    // Update momentum.
    //--------------------------------------------------------------------------
    ux  += hax;
    uy  += hay;
    uz  += haz;

    v00  = qdt_4mc * rsqrt( one + fma( ux, ux, fma( uy, uy, uz * uz ) ) );
    v01  = fma( cbx, cbx, fma( cby, cby, cbz * cbz ) );
    v02  = ( v00 * v00 ) * v01;
    v03  = v00 * fma( v02, fma( v02, two_fifteenths, one_third ), one );
    v04  = v03 * rcp( fma( v03 * v03, v01, one ) );
    v04 += v04;

    v00  = fma( fms( uy, cbz, uz * cby ), v03, ux );
    v01  = fma( fms( uz, cbx, ux * cbz ), v03, uy );
    v02  = fma( fms( ux, cby, uy * cbx ), v03, uz );

    ux   = fma( fms( v01, cbz, v02 * cby ), v04, ux );
    uy   = fma( fms( v02, cbx, v00 * cbz ), v04, uy );
    uz   = fma( fms( v00, cby, v01 * cbx ), v04, uz );

    //--------------------------------------------------------------------------
    // Store particle momentum data.
    //--------------------------------------------------------------------------
    store_8x1( ux, &pb[0].ux[0] );
    store_8x1( uy, &pb[0].uy[0] );
    store_8x1( uz, &pb[0].uz[0] );
  }
}
#else // VPIC_USE_AOSOA_P is not defined i.e. VPIC_USE_AOS_P case.
void
center_p_pipeline_v8( center_p_pipeline_args_t * args,
                      int pipeline_rank,
                      int n_pipeline )
{
  const interpolator_t * ALIGNED(128) f0 = args->f0;

  particle_t           * ALIGNED(128) p;

  const species_t      * sp = args->sp;

  const v8float qdt_2mc(    args->qdt_2mc);
  const v8float qdt_4mc(0.5*args->qdt_2mc); // For half Boris rotate
  const v8float one(1.0);
  const v8float one_third(1.0/3.0);
  const v8float two_fifteenths(2.0/15.0);

  v8float ex, dexdy, dexdz, d2exdydz;
  v8float ey, deydz, deydx, d2eydzdx;
  v8float ez, dezdx, dezdy, d2ezdxdy;

  v8float cbx, dcbxdx;
  v8float cby, dcbydy;
  v8float cbz, dcbzdz;

  v8float dx, dy, dz;
  v8float ux, uy, uz, q;
  v8float hax, hay, haz;
  v8float cbxp, cbyp, cbzp;
  v8float v00, v01, v02, v03, v04;
  v8int   ii;

  int first_part; // Index of first particle for this thread.
  int  last_part; // Index of last  particle for this thread.
  int     n_part; // Number of particles for this thread.

  int previous_vox; // Index of previous voxel.
  int    first_vox; // Index of first voxel for this thread.
  int     last_vox; // Index of last  voxel for this thread.
  int        n_vox; // Number of voxels for this thread.
  int          vox; // Index of current voxel.

  int sum_part = 0;

  //--------------------------------------------------------------------------//
  // Compute an equal division of particles across pipeline processes.
  //--------------------------------------------------------------------------//

  DISTRIBUTE( args->np, 1, pipeline_rank, n_pipeline, first_part, n_part );

  last_part = first_part + n_part - 1;

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

  vox = VOXEL( first_ix, first_iy, first_iz,
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

      for( int i = 0; i < part_count; i+=8, p+=8 )
      {
        //--------------------------------------------------------------------------
        // Load particle position.
        //--------------------------------------------------------------------------

        load_8x4_tr( &p[0].dx, &p[1].dx, &p[2].dx, &p[3].dx,
                     &p[4].dx, &p[5].dx, &p[6].dx, &p[7].dx,
                     dx, dy, dz, ii );

        //--------------------------------------------------------------------------
        // Interpolate E.
        //--------------------------------------------------------------------------

        hax = qdt_2mc*fma( fma( dy, d2exdydz, dexdz ), dz, fma( dy, dexdy, ex ) );
        hay = qdt_2mc*fma( fma( dz, d2eydzdx, deydx ), dx, fma( dz, deydz, ey ) );
        haz = qdt_2mc*fma( fma( dx, d2ezdxdy, dezdy ), dy, fma( dx, dezdx, ez ) );

        //--------------------------------------------------------------------------
        // Interpolate B.
        //--------------------------------------------------------------------------

        cbxp = fma( dcbxdx, dx, cbx );
        cbyp = fma( dcbydy, dy, cby );
        cbzp = fma( dcbzdz, dz, cbz );

        //--------------------------------------------------------------------------
        // Load particle momentum.  Could use load_8x3_tr.
        //--------------------------------------------------------------------------

        load_8x4_tr( &p[0].ux, &p[1].ux, &p[2].ux, &p[3].ux,
                     &p[4].ux, &p[5].ux, &p[6].ux, &p[7].ux,
                     ux, uy, uz, q );

        //--------------------------------------------------------------------------
        // Half advance E.
        //--------------------------------------------------------------------------

        ux  += hax;
        uy  += hay;
        uz  += haz;

        //--------------------------------------------------------------------------
        // Boris - scalars.
        //--------------------------------------------------------------------------

        v00  = qdt_4mc * rsqrt( one + fma( ux, ux, fma( uy, uy, uz * uz ) ) );
        v01  = fma( cbxp, cbxp, fma( cbyp, cbyp, cbzp * cbzp ) );
        v02  = ( v00 * v00 ) * v01;
        v03  = v00 * fma( v02, fma( v02, two_fifteenths, one_third ), one );
        v04  = v03 * rcp( fma( v03 * v03, v01, one ) );
        v04 += v04;

        //--------------------------------------------------------------------------
        // Boris - uprime.
        //--------------------------------------------------------------------------

        v00  = fma( fms( uy, cbzp, uz * cbyp ), v03, ux );
        v01  = fma( fms( uz, cbxp, ux * cbzp ), v03, uy );
        v02  = fma( fms( ux, cbyp, uy * cbxp ), v03, uz );

        //--------------------------------------------------------------------------
        // Boris - rotation.
        //--------------------------------------------------------------------------

        ux   = fma( fms( v01, cbzp, v02 * cbyp ), v04, ux );
        uy   = fma( fms( v02, cbxp, v00 * cbzp ), v04, uy );
        uz   = fma( fms( v00, cbyp, v01 * cbxp ), v04, uz );

        //--------------------------------------------------------------------------
        // Store particle momentum.  Could use store_8x3_tr.
        //--------------------------------------------------------------------------

        store_8x4_tr( ux, uy, uz, q,
                      &p[0].ux, &p[1].ux, &p[2].ux, &p[3].ux,
                      &p[4].ux, &p[5].ux, &p[6].ux, &p[7].ux );
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
}
#endif // End of VPIC_USE_AOSOA_P vs VPIC_USE_AOS_P selection.

#else // V8_ACCELERATION is not defined.

void
center_p_pipeline_v8( center_p_pipeline_args_t * args,
                      int pipeline_rank,
                      int n_pipeline )
{
  // No v8 implementation.
  ERROR( ( "No center_p_pipeline_v8 implementation." ) );
}

#endif // End of V8_ACCELERATION selection.
