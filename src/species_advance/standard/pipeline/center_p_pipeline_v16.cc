#define IN_spa

#include "spa_private.h"

#include "species_advance_pipeline.h"

#if defined(V16_ACCELERATION)

using namespace v16;

#if defined(VPIC_USE_AOSOA_P)

void
center_p_pipeline_v16( center_p_pipeline_args_t * args,
                       int pipeline_rank,
                       int n_pipeline )
{
  const interpolator_t * ALIGNED(128) f0 = args->f0;

  particle_block_t     * ALIGNED(128) pb;

  const species_t      * sp = args->sp;

  const v16float qdt_2mc(    args->qdt_2mc);
  const v16float qdt_4mc(0.5*args->qdt_2mc); // For half Boris rotate.
  const v16float one(1.0);
  const v16float one_third(1.0/3.0);
  const v16float two_fifteenths(2.0/15.0);

  v16float ex, dexdy, dexdz, d2exdydz;
  v16float ey, deydz, deydx, d2eydzdx;
  v16float ez, dezdx, dezdy, d2ezdxdy;

  v16float cbx, dcbxdx;
  v16float cby, dcbydy;
  v16float cbz, dcbzdz;

  v16float dx, dy, dz;
  v16float ux, uy, uz;
  v16float hax, hay, haz;
  v16float cbxp, cbyp, cbzp;
  v16float v00, v01, v02, v03, v04;

  int first_part; // Index of first particle for this thread.
  int  last_part; // Index of last  particle for this thread.
  int     n_part; // Number of particles for this thread.

  int      n_vox; // Number of voxels for this thread.
  int        vox; // Index of current voxel.

  int first_ix = 0;
  int first_iy = 0;
  int first_iz = 0;

  //--------------------------------------------------------------------------//
  // Compute an equal division of particles across pipeline processes.
  //--------------------------------------------------------------------------//

  DISTRIBUTE( args->np, 1, pipeline_rank, n_pipeline, first_part, n_part );

  last_part = first_part + n_part - 1;

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
      pb = args->pb0 + part_start / PARTICLE_BLOCK_SIZE;

      int ib = 0;

      for( int i = 0; i < part_count; i += PARTICLE_BLOCK_SIZE )
      {
        ib = i / PARTICLE_BLOCK_SIZE;          // Index of particle block.

        //--------------------------------------------------------------------------
        // Load particle position and momentum.
        //--------------------------------------------------------------------------

        load_16x1( &pb[ib].dx[0], dx );
        load_16x1( &pb[ib].dy[0], dy );
        load_16x1( &pb[ib].dz[0], dz );

        load_16x1( &pb[ib].ux[0], ux );
        load_16x1( &pb[ib].uy[0], uy );
        load_16x1( &pb[ib].uz[0], uz );

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
        // Store particle momentum.
        //--------------------------------------------------------------------------

        #if 0
        stream_16x1( ux, &pb[ib].ux[0] );
        stream_16x1( uy, &pb[ib].uy[0] );
        stream_16x1( uz, &pb[ib].uz[0] );
        #endif

        store_16x1( ux, &pb[ib].ux[0] );
        store_16x1( uy, &pb[ib].uy[0] );
        store_16x1( uz, &pb[ib].uz[0] );
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

#else // VPIC_USE_AOSOA_P is not defined i.e. VPIC_USE_AOS_P case.

void
center_p_pipeline_v16( center_p_pipeline_args_t * args,
                       int pipeline_rank,
                       int n_pipeline )
{
  const interpolator_t * ALIGNED(128) f0 = args->f0;

  particle_t           * ALIGNED(128) p;

  const species_t      * sp = args->sp;

  const v16float qdt_2mc(    args->qdt_2mc);
  const v16float qdt_4mc(0.5*args->qdt_2mc); // For half Boris rotate.
  const v16float one(1.0);
  const v16float one_third(1.0/3.0);
  const v16float two_fifteenths(2.0/15.0);

  v16float ex, dexdy, dexdz, d2exdydz;
  v16float ey, deydz, deydx, d2eydzdx;
  v16float ez, dezdx, dezdy, d2ezdxdy;

  v16float cbx, dcbxdx;
  v16float cby, dcbydy;
  v16float cbz, dcbzdz;

  v16float dx, dy, dz;
  v16float ux, uy, uz, q;
  v16float hax, hay, haz;
  v16float cbxp, cbyp, cbzp;
  v16float v00, v01, v02, v03, v04;
  v16int   ii;

  int first_part; // Index of first particle for this thread.
  int  last_part; // Index of last  particle for this thread.
  int     n_part; // Number of particles for this thread.

  int      n_vox; // Number of voxels for this thread.
  int        vox; // Index of current voxel.

  int first_ix = 0;
  int first_iy = 0;
  int first_iz = 0;

  //--------------------------------------------------------------------------//
  // Compute an equal division of particles across pipeline processes.
  //--------------------------------------------------------------------------//

  DISTRIBUTE( args->np, 1, pipeline_rank, n_pipeline, first_part, n_part );

  last_part = first_part + n_part - 1;

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

      for( int i = 0; i < part_count; i += 16, p += 16 )
      {
        //--------------------------------------------------------------------------
        // Load particle position and momentum.
        //--------------------------------------------------------------------------

        load_16x8_tr_p( &p[ 0].dx, &p[ 2].dx, &p[ 4].dx, &p[ 6].dx,
                        &p[ 8].dx, &p[10].dx, &p[12].dx, &p[14].dx,
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
        // Store particle momentum.
        //--------------------------------------------------------------------------

        store_16x8_tr_p( dx, dy, dz, ii, ux, uy, uz, q,
                         &p[ 0].dx, &p[ 2].dx, &p[ 4].dx, &p[ 6].dx,
                         &p[ 8].dx, &p[10].dx, &p[12].dx, &p[14].dx );
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

#else // V16_ACCELERATION is not defined.

void
center_p_pipeline_v16( center_p_pipeline_args_t * args,
                       int pipeline_rank,
                       int n_pipeline )
{
  // No v16 implementation.
  ERROR( ( "No center_p_pipeline_v16 implementation." ) );
}

#endif // End of V16_ACCELERATION selection.
