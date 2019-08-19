#define IN_spa

#include "spa_private.h"

#if defined(V4_ACCELERATION)

using namespace v4;

#if defined(VPIC_USE_AOSOA_P)

void
uncenter_p_pipeline_v4( center_p_pipeline_args_t * args,
                        int pipeline_rank,
                        int n_pipeline )
{
  const interpolator_t * ALIGNED(128) f0 = args->f0;

  particle_block_t     * ALIGNED(128) pb;

  const float          * ALIGNED(16)  vp00;
  const float          * ALIGNED(16)  vp01;
  const float          * ALIGNED(16)  vp02;
  const float          * ALIGNED(16)  vp03;

  const v4float qdt_2mc(    -args->qdt_2mc); // For backward half advance.
  const v4float qdt_4mc(-0.5*args->qdt_2mc); // For backward half Boris rotate.
  const v4float one(1.0);
  const v4float one_third(1.0/3.0);
  const v4float two_fifteenths(2.0/15.0);

  v4float dx, dy, dz, ux, uy, uz, q;
  v4float hax, hay, haz, cbx, cby, cbz;
  v4float v00, v01, v02, v03, v04, v05;
  v4int   ii;

  int first, nq;

  // Determine which particle blocks this pipeline processes.

  DISTRIBUTE( args->np, PARTICLE_BLOCK_SIZE, pipeline_rank, n_pipeline, first, nq );

  pb = args->pb0 + first / PARTICLE_BLOCK_SIZE;

  nq >>= 2;

  // Process the particle blocks for this pipeline.  For the initial AoSoA version,
  // assume that PARTICLE_BLOCK_SIZE is the same as the vector length.  A future
  // version might allow PARTICLE_BLOCK_SIZE to be an integral multiple of the vector
  // length.

  for( ; nq; nq--, pb++ )
  {
    //--------------------------------------------------------------------------
    // Load particle position data.
    //--------------------------------------------------------------------------
    load_4x1( &pb[0].dx[0], dx );
    load_4x1( &pb[0].dy[0], dy );
    load_4x1( &pb[0].dz[0], dz );
    load_4x1( &pb[0].i [0], ii );

    //--------------------------------------------------------------------------
    // Set field interpolation pointers.
    //--------------------------------------------------------------------------
    vp00 = ( const float * ALIGNED(16) ) ( f0 + ii(0) );
    vp01 = ( const float * ALIGNED(16) ) ( f0 + ii(1) );
    vp02 = ( const float * ALIGNED(16) ) ( f0 + ii(2) );
    vp03 = ( const float * ALIGNED(16) ) ( f0 + ii(3) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_4x4_tr( vp00, vp01, vp02, vp03,
		 hax, v00, v01, v02 );

    hax = qdt_2mc*fma( fma( dy, v02, v01 ), dz, fma( dy, v00, hax ) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_4x4_tr( vp00+4, vp01+4, vp02+4, vp03+4,
		 hay, v03, v04, v05 );

    hay = qdt_2mc*fma( fma( dz, v05, v04 ), dx, fma( dz, v03, hay ) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_4x4_tr( vp00+8, vp01+8, vp02+8, vp03+8,
		 haz, v00, v01, v02 );

    haz = qdt_2mc*fma( fma( dx, v02, v01 ), dy, fma( dx, v00, haz ) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_4x4_tr( vp00+12, vp01+12, vp02+12, vp03+12,
		 cbx, v03, cby, v04 );

    cbx = fma( v03, dx, cbx );
    cby = fma( v04, dy, cby );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles, final.
    //--------------------------------------------------------------------------
    load_4x2_tr( vp00+16, vp01+16, vp02+16, vp03+16,
		 cbz, v05 );

    cbz = fma( v05, dz, cbz );

    //--------------------------------------------------------------------------
    // Load particle momentum data.
    //--------------------------------------------------------------------------
    load_4x1( &pb[0].ux[0], ux );
    load_4x1( &pb[0].uy[0], uy );
    load_4x1( &pb[0].uz[0], uz );

    // load_4x4( &pb[0].ux[0], &pb[0].uy[0], &pb[0].uz[0], &pb[0].w[0],
    //           ux, uy, uz, q );

    //--------------------------------------------------------------------------
    // Update momentum.
    //--------------------------------------------------------------------------
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

    ux  += hax;
    uy  += hay;
    uz  += haz;

    //--------------------------------------------------------------------------
    // Store particle momentum data.
    //--------------------------------------------------------------------------
    store_4x1( ux, &pb[0].ux[0] );
    store_4x1( uy, &pb[0].uy[0] );
    store_4x1( uz, &pb[0].uz[0] );
  }
}

#else

void
uncenter_p_pipeline_v4( center_p_pipeline_args_t * args,
                        int pipeline_rank,
                        int n_pipeline )
{
  const interpolator_t * ALIGNED(128) f0 = args->f0;

  particle_t           * ALIGNED(128) p;

  const float          * ALIGNED(16)  vp00;
  const float          * ALIGNED(16)  vp01;
  const float          * ALIGNED(16)  vp02;
  const float          * ALIGNED(16)  vp03;

  const v4float qdt_2mc(    -args->qdt_2mc); // For backward half advance.
  const v4float qdt_4mc(-0.5*args->qdt_2mc); // For backward half Boris rotate.
  const v4float one(1.0);
  const v4float one_third(1.0/3.0);
  const v4float two_fifteenths(2.0/15.0);

  v4float dx, dy, dz, ux, uy, uz, q;
  v4float hax, hay, haz, cbx, cby, cbz;
  v4float v00, v01, v02, v03, v04, v05;
  v4int   ii;

  int first, nq;

  // Determine which particle quads this pipeline processes.

  DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, first, nq );

  p = args->p0 + first;

  nq >>= 2;

  // Process the particle quads for this pipeline.

  for( ; nq; nq--, p+=4 )
  {
    //--------------------------------------------------------------------------
    // Load particle position data.
    //--------------------------------------------------------------------------
    load_4x4_tr( &p[0].dx, &p[1].dx, &p[2].dx, &p[3].dx,
    		 dx, dy, dz, ii );

    //--------------------------------------------------------------------------
    // Set field interpolation pointers.
    //--------------------------------------------------------------------------
    vp00 = ( const float * ALIGNED(16) ) ( f0 + ii(0) );
    vp01 = ( const float * ALIGNED(16) ) ( f0 + ii(1) );
    vp02 = ( const float * ALIGNED(16) ) ( f0 + ii(2) );
    vp03 = ( const float * ALIGNED(16) ) ( f0 + ii(3) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_4x4_tr( vp00, vp01, vp02, vp03,
    		 hax, v00, v01, v02 );

    hax = qdt_2mc * fma( fma( dy, v02, v01 ), dz, fma( dy, v00, hax ) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_4x4_tr( vp00+4, vp01+4, vp02+4, vp03+4,
    		 hay, v03, v04, v05 );

    hay = qdt_2mc * fma( fma( dz, v05, v04 ), dx, fma( dz, v03, hay ) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_4x4_tr( vp00+8, vp01+8, vp02+8, vp03+8,
    		 haz, v00, v01, v02 );

    haz = qdt_2mc * fma( fma( dx, v02, v01 ), dy, fma( dx, v00, haz ) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_4x4_tr( vp00+12, vp01+12, vp02+12, vp03+12,
    		 cbx, v03, cby, v04 );

    cbx = fma( v03, dx, cbx );
    cby = fma( v04, dy, cby );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles, final.
    //--------------------------------------------------------------------------
    load_4x2_tr( vp00+16, vp01+16, vp02+16, vp03+16,
		 cbz, v05 );

    cbz = fma( v05, dz, cbz );

    //--------------------------------------------------------------------------
    // Load particle momentum data.  Could use load_4x3_tr.
    //--------------------------------------------------------------------------
    load_4x4_tr( &p[0].ux, &p[1].ux, &p[2].ux, &p[3].ux,
		 ux, uy, uz, q );

    //--------------------------------------------------------------------------
    // Update momentum.
    //--------------------------------------------------------------------------
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

    ux  += hax;
    uy  += hay;
    uz  += haz;

    //--------------------------------------------------------------------------
    // Store particle data.  Could use store_4x3_tr.
    //--------------------------------------------------------------------------
    store_4x4_tr( ux, uy, uz, q,
		  &p[0].ux, &p[1].ux, &p[2].ux, &p[3].ux );
  }
}

#endif

#else

void
uncenter_p_pipeline_v4( center_p_pipeline_args_t * args,
                        int pipeline_rank,
                        int n_pipeline )
{
  // No v4 implementation.
  ERROR( ( "No uncenter_p_pipeline_v4 implementation." ) );
}

#endif
