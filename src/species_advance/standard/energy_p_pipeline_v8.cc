using namespace v8;

void
energy_p_pipeline_v8( energy_p_pipeline_args_t * args,
                      int pipeline_rank,
                      int n_pipeline )
{
  const interpolator_t * RESTRICT ALIGNED(128) f = args->f;
  const particle_t     * RESTRICT ALIGNED(128) p = args->p;

  const float          * RESTRICT ALIGNED(32)  vp00;
  const float          * RESTRICT ALIGNED(32)  vp01;
  const float          * RESTRICT ALIGNED(32)  vp02;
  const float          * RESTRICT ALIGNED(32)  vp03;
  const float          * RESTRICT ALIGNED(32)  vp04;
  const float          * RESTRICT ALIGNED(32)  vp05;
  const float          * RESTRICT ALIGNED(32)  vp06;
  const float          * RESTRICT ALIGNED(32)  vp07;

  const v8float qdt_2mc(args->qdt_2mc);
  const v8float msp(args->msp);
  const v8float one(1.0);

  v8float dx, dy, dz;
  v8float ex, ey, ez;
  v8float v00, v01, v02, w;
  v8int i;

  double en00 = 0.0, en01 = 0.0, en02 = 0.0, en03 = 0.0;
  double en04 = 0.0, en05 = 0.0, en06 = 0.0, en07 = 0.0;

  int n0, nq;

  // Determine which particle blocks this pipeline processes.

  DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, n0, nq );

  p += n0;

  nq >>= 3;

  // Process the particle blocks for this pipeline.

  for( ; nq; nq--, p+=8 )
  {
    //--------------------------------------------------------------------------
    // Load particle position data.
    //--------------------------------------------------------------------------
    load_8x4_tr( &p[0].dx, &p[1].dx, &p[2].dx, &p[3].dx,
                 &p[4].dx, &p[5].dx, &p[6].dx, &p[7].dx,
                 dx, dy, dz, i );

    //--------------------------------------------------------------------------
    // Set field interpolation pointers.
    //--------------------------------------------------------------------------
    vp00 = ( float * ) ( f + i(0) );
    vp01 = ( float * ) ( f + i(1) );
    vp02 = ( float * ) ( f + i(2) );
    vp03 = ( float * ) ( f + i(3) );
    vp04 = ( float * ) ( f + i(4) );
    vp05 = ( float * ) ( f + i(5) );
    vp06 = ( float * ) ( f + i(6) );
    vp07 = ( float * ) ( f + i(7) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_8x4_tr( vp00, vp01, vp02, vp03,
		 vp04, vp05, vp06, vp07,
		 ex, v00, v01, v02 );

    ex = fma( fma( dy, v02, v01 ), dz, fma( dy, v00, ex ) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_8x4_tr( vp00+4, vp01+4, vp02+4, vp03+4,
		 vp04+4, vp05+4, vp06+4, vp07+4,
		 ey, v00, v01, v02 );

    ey = fma( fma( dz, v02, v01 ), dx, fma( dz, v00, ey ) );

    //--------------------------------------------------------------------------
    // Load interpolation data for particles.
    //--------------------------------------------------------------------------
    load_8x4_tr( vp00+8, vp01+8, vp02+8, vp03+8,
		 vp04+8, vp05+8, vp06+8, vp07+8,
		 ez, v00, v01, v02 );

    ez = fma( fma( dx, v02, v01 ), dy, fma( dx, v00, ez ) );

    //--------------------------------------------------------------------------
    // Load particle momentum data.
    //--------------------------------------------------------------------------
    load_8x4_tr( &p[0].ux, &p[1].ux, &p[2].ux, &p[3].ux,
		 &p[4].ux, &p[5].ux, &p[6].ux, &p[7].ux,
		 v00, v01, v02, w );

    //--------------------------------------------------------------------------
    // Update momentum to half step. Note that Boris rotation does not change
    // energy and thus is not necessary.
    //--------------------------------------------------------------------------
    v00 = fma( ex, qdt_2mc, v00 );
    v01 = fma( ey, qdt_2mc, v01 );
    v02 = fma( ez, qdt_2mc, v02 );

    //--------------------------------------------------------------------------
    // Calculate kinetic energy of particles.
    //--------------------------------------------------------------------------
    v00 = fma( v00, v00, fma( v01, v01, v02 * v02 ) );

    v00 = ( msp * w ) * ( v00 / ( one + sqrt( one + v00 ) ) ); 

    //--------------------------------------------------------------------------
    // Accumulate energy for each vector element.
    //--------------------------------------------------------------------------
    en00 += ( double ) v00(0);
    en01 += ( double ) v00(1);
    en02 += ( double ) v00(2);
    en03 += ( double ) v00(3);
    en04 += ( double ) v00(4);
    en05 += ( double ) v00(5);
    en06 += ( double ) v00(6);
    en07 += ( double ) v00(7);
  }

  //--------------------------------------------------------------------------
  // Accumulate energy for each rank or thread.
  //--------------------------------------------------------------------------
  args->en[pipeline_rank] = en00 + en01 + en02 + en03 +
                            en04 + en05 + en06 + en07;
}
