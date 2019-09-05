#define IN_spa

#define HAS_V4_PIPELINE
#define HAS_V8_PIPELINE
#define HAS_V16_PIPELINE

#include "spa_private.h"

#include "species_advance_pipeline.h"

#include "../../../util/pipelines/pipelines_exec.h"

//----------------------------------------------------------------------------//
// Reference implementation for an uncenter_p pipeline function which does not
// make use of explicit calls to vector intrinsic functions.
//----------------------------------------------------------------------------//

#if defined(VPIC_USE_AOSOA_P)

void
uncenter_p_pipeline_scalar( center_p_pipeline_args_t * args,
                            int pipeline_rank,
                            int n_pipeline )
{
  const interpolator_t * ALIGNED(128) f0 = args->f0;

  particle_block_t     * ALIGNED(32)  pb;

  const species_t      * sp = args->sp;

  const float qdt_2mc        =     -args->qdt_2mc; // For backward half advance
  const float qdt_4mc        = -0.5*args->qdt_2mc; // For backward half rotate
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
  float v0, v1, v2, v3, v4;

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

      // Process the particles in a cell.
      #define VPIC_SIMD_LEN 16
      for( int i = 0; i < part_count; i += VPIC_SIMD_LEN )
      {
        ib = i / PARTICLE_BLOCK_SIZE;          // Index of particle block.

        #ifdef VPIC_SIMD_LEN
        #pragma omp simd simdlen(VPIC_SIMD_LEN)
        #endif
	for( int j = 0; j < VPIC_SIMD_LEN; j++ )
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

          // Boris - scalars.
          v0   = qdt_4mc / (float) sqrt( one + ( ux * ux + ( uy * uy + uz * uz ) ) );
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
          pb[ib].ux[j] = ux;
          pb[ib].uy[j] = uy;
          pb[ib].uz[j] = uz;
        }
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
uncenter_p_pipeline_scalar( center_p_pipeline_args_t * args,
                            int pipeline_rank,
                            int n_pipeline )
{
  const interpolator_t * ALIGNED(128) f0 = args->f0;

  particle_t           * ALIGNED(32)  p;

  const species_t      * sp = args->sp;

  const float qdt_2mc        =     -args->qdt_2mc; // For backward half advance
  const float qdt_4mc        = -0.5*args->qdt_2mc; // For backward half rotate
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
  float v0, v1, v2, v3, v4;

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

      // Process the particles in a cell.
      for( int i = 0; i < part_count; i++, p++ )
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

	// Boris - scalars.
        v0   = qdt_4mc / (float) sqrt( one + ( ux * ux + ( uy * uy + uz * uz ) ) );
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
        p->ux = ux;
        p->uy = uy;
        p->uz = uz;
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

//----------------------------------------------------------------------------//
// Top level function to select and call the proper uncenter_p pipeline
// function.
//----------------------------------------------------------------------------//

#if defined(VPIC_USE_AOSOA_P)

void
uncenter_p_pipeline( species_t * RESTRICT sp,
                     const interpolator_array_t * RESTRICT ia )
{
  DECLARE_ALIGNED_ARRAY( center_p_pipeline_args_t, 128, args, 1 );

  if ( ! sp ||
       ! ia ||
       sp->g != ia->g )
  {
    ERROR( ( "Bad args." ) );
  }

  // Have the pipelines do the bulk of particles in blocks and have the
  // host do the final incomplete block.

  args->pb0     = sp->pb;
  args->f0      = ia->i;
  args->qdt_2mc = ( sp->q * sp->g->dt ) / ( 2 * sp->m * sp->g->cvac );
  args->np      = sp->np;
  args->sp      = sp;

  EXEC_PIPELINES( uncenter_p, args, 0 );

  WAIT_PIPELINES();
}

#else // VPIC_USE_AOSOA_P is not defined i.e. VPIC_USE_AOS_P case.

void
uncenter_p_pipeline( species_t * RESTRICT sp,
                     const interpolator_array_t * RESTRICT ia )
{
  DECLARE_ALIGNED_ARRAY( center_p_pipeline_args_t, 128, args, 1 );

  if ( ! sp ||
       ! ia ||
       sp->g != ia->g )
  {
    ERROR( ( "Bad args." ) );
  }

  // Have the pipelines do the bulk of particles in blocks and have the
  // host do the final incomplete block.

  args->p0      = sp->p;
  args->f0      = ia->i;
  args->qdt_2mc = ( sp->q * sp->g->dt ) / ( 2 * sp->m * sp->g->cvac );
  args->np      = sp->np;
  args->sp      = sp;

  EXEC_PIPELINES( uncenter_p, args, 0 );

  WAIT_PIPELINES();
}

#endif // End of VPIC_USE_AOSOA_P vs VPIC_USE_AOS_P selection.
