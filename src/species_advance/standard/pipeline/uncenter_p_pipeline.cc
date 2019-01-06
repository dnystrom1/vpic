#define IN_spa

#define HAS_V4_PIPELINE
#define HAS_V8_PIPELINE
#define HAS_V16_PIPELINE

#include "spa_private.h"

#include "../../../util/pipelines/pipelines_exec.h"

#include <iostream>

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

  const interpolator_t * ALIGNED(16)  f;

  const float qdt_2mc        =     -args->qdt_2mc; // For backward half advance
  const float qdt_4mc        = -0.5*args->qdt_2mc; // For backward half rotate
  const float one            = 1.0;
  const float one_third      = 1.0/3.0;
  const float two_fifteenths = 2.0/15.0;

  float dx, dy, dz, ux, uy, uz;
  float hax, hay, haz, cbx, cby, cbz;
  float v0, v1, v2, v3, v4;
  int   ii;

  int first, n;

  // Determine which particles this pipeline processes.

  DISTRIBUTE( args->np, PARTICLE_BLOCK_SIZE, pipeline_rank, n_pipeline, first, n );

  // p = args->p0 + first;

  pb = args->pb0 + first / PARTICLE_BLOCK_SIZE;

  // int pb_size = PARTICLE_BLOCK_SIZE;
  // std::cout << "-----------------------------------------------------------" << std::endl
  //           << "uncenter_p_pipeline_scalar, pipeline = " << pipeline_rank    << std::endl
  //           << "-----------------------------------------------------------" << std::endl
  //           << "Number of pipelines                  = " << n_pipeline       << std::endl
  //           << "Total number of particles            = " << args->np         << std::endl
  //           << "Index of first particle              = " << first            << std::endl
  //           << "Index of first block                 = " << first / pb_size  << std::endl
  //           << "Number of particles, this pipeline   = " << n                << std::endl;

  // ERROR( ( "Need AoSoA implementation." ) );

  // Process particles for this pipeline.

  int ib = 0;
  int ip = 0;

  // for( ; n; n--, p++ )
  for( int i = 0 ; i < n; i++ )
  {
    ib   = i / PARTICLE_BLOCK_SIZE;          // Index of particle block.
    ip   = i - PARTICLE_BLOCK_SIZE * ib;     // Index of next particle in block.

    dx   = pb[ib].dx[ip];                    // Load position
    dy   = pb[ib].dy[ip];
    dz   = pb[ib].dz[ip];
    ii   = pb[ib].i [ip];

    f    = f0 + ii;                          // Interpolate E

    hax  = qdt_2mc*(    ( f->ex    + dy*f->dexdy    ) +
                     dz*( f->dexdz + dy*f->d2exdydz ) );

    hay  = qdt_2mc*(    ( f->ey    + dz*f->deydz    ) +
                     dx*( f->deydx + dz*f->d2eydzdx ) );

    haz  = qdt_2mc*(    ( f->ez    + dx*f->dezdx    ) +
                     dy*( f->dezdy + dx*f->d2ezdxdy ) );

    cbx  = f->cbx + dx*f->dcbxdx;            // Interpolate B
    cby  = f->cby + dy*f->dcbydy;
    cbz  = f->cbz + dz*f->dcbzdz;

    ux   = pb[ib].ux[ip];                    // Load momentum
    uy   = pb[ib].uy[ip];
    uz   = pb[ib].uz[ip];

    v0   = qdt_4mc/(float)sqrt(one + (ux*ux + (uy*uy + uz*uz)));
    /**/                                     // Boris - scalars
    v1   = cbx*cbx + (cby*cby + cbz*cbz);
    v2   = (v0*v0)*v1;
    v3   = v0*(one+v2*(one_third+v2*two_fifteenths));
    v4   = v3/(one+v1*(v3*v3));
    v4  += v4;

    v0   = ux + v3*( uy*cbz - uz*cby );      // Boris - uprime
    v1   = uy + v3*( uz*cbx - ux*cbz );
    v2   = uz + v3*( ux*cby - uy*cbx );

    ux  += v4*( v1*cbz - v2*cby );           // Boris - rotation
    uy  += v4*( v2*cbx - v0*cbz );
    uz  += v4*( v0*cby - v1*cbx );

    ux  += hax;                              // Half advance E
    uy  += hay;
    uz  += haz;

    pb[ib].ux[ip] = ux;                      // Store momentum
    pb[ib].uy[ip] = uy;
    pb[ib].uz[ip] = uz;
  }
}
#else
void
uncenter_p_pipeline_scalar( center_p_pipeline_args_t * args,
                            int pipeline_rank,
                            int n_pipeline )
{
  const interpolator_t * ALIGNED(128) f0 = args->f0;

  particle_t           * ALIGNED(32)  p;

  const interpolator_t * ALIGNED(16)  f;

  const float qdt_2mc        =     -args->qdt_2mc; // For backward half advance
  const float qdt_4mc        = -0.5*args->qdt_2mc; // For backward half rotate
  const float one            = 1.0;
  const float one_third      = 1.0/3.0;
  const float two_fifteenths = 2.0/15.0;

  float dx, dy, dz, ux, uy, uz;
  float hax, hay, haz, cbx, cby, cbz;
  float v0, v1, v2, v3, v4;
  int   ii;

  int first, n;

  // Determine which particles this pipeline processes.

  DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, first, n );

  p = args->p0 + first;

  // Process particles for this pipeline.

  for( ; n; n--, p++ )
  {
    dx   = p->dx;                            // Load position
    dy   = p->dy;
    dz   = p->dz;
    ii   = p->i;

    f    = f0 + ii;                          // Interpolate E

    hax  = qdt_2mc*(    ( f->ex    + dy*f->dexdy    ) +
                     dz*( f->dexdz + dy*f->d2exdydz ) );

    hay  = qdt_2mc*(    ( f->ey    + dz*f->deydz    ) +
                     dx*( f->deydx + dz*f->d2eydzdx ) );

    haz  = qdt_2mc*(    ( f->ez    + dx*f->dezdx    ) +
                     dy*( f->dezdy + dx*f->d2ezdxdy ) );

    cbx  = f->cbx + dx*f->dcbxdx;            // Interpolate B
    cby  = f->cby + dy*f->dcbydy;
    cbz  = f->cbz + dz*f->dcbzdz;

    ux   = p->ux;                            // Load momentum
    uy   = p->uy;
    uz   = p->uz;

    v0   = qdt_4mc/(float)sqrt(one + (ux*ux + (uy*uy + uz*uz)));
    /**/                                     // Boris - scalars
    v1   = cbx*cbx + (cby*cby + cbz*cbz);
    v2   = (v0*v0)*v1;
    v3   = v0*(one+v2*(one_third+v2*two_fifteenths));
    v4   = v3/(one+v1*(v3*v3));
    v4  += v4;

    v0   = ux + v3*( uy*cbz - uz*cby );      // Boris - uprime
    v1   = uy + v3*( uz*cbx - ux*cbz );
    v2   = uz + v3*( ux*cby - uy*cbx );

    ux  += v4*( v1*cbz - v2*cby );           // Boris - rotation
    uy  += v4*( v2*cbx - v0*cbz );
    uz  += v4*( v0*cby - v1*cbx );

    ux  += hax;                              // Half advance E
    uy  += hay;
    uz  += haz;

    p->ux = ux;                              // Store momentum
    p->uy = uy;
    p->uz = uz;
  }
}
#endif

//----------------------------------------------------------------------------//
// Top level function to select and call the proper uncenter_p pipeline
// function.
//----------------------------------------------------------------------------//

#if defined(VPIC_USE_AOSOA_P)
// void
// uncenter_p_pipeline( species_t * RESTRICT sp,
//                      const interpolator_array_t * RESTRICT ia )
// {
//   ERROR(("Need AoSoA implementation."));
// }

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

  EXEC_PIPELINES( uncenter_p, args, 0 );

  WAIT_PIPELINES();
}
#else
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

  EXEC_PIPELINES( uncenter_p, args, 0 );

  WAIT_PIPELINES();
}
#endif
