#include "vpic.h"
#define FAK field_array->kernel

// Use this for Intel and VTune.
#if defined(VPIC_USE_VTUNE_CENTER_P)
#include "ittnotify.h"
#endif

// Use this for LikWid.
#if defined(VPIC_USE_LIKWID_CENTER_P)
#include <likwid-marker.h>
#endif

// Use this for Armie.
#if defined(VPIC_USE_ARMIE_CENTER_P)
#define __ARMIE_START_TRACE() { asm volatile (".inst 0x2520e020"); }
#define __ARMIE_STOP_TRACE() { asm volatile (".inst 0x2520e040"); }
#endif

void
vpic_simulation::initialize( int argc,
                             char **argv ) {
  double err;
  species_t * sp;

  // Call the user initialize the simulation

  TIC user_initialization( argc, argv ); TOC( user_initialization, 1 );

  // Do some consistency checks on user initialized fields

  if( rank()==0 ) MESSAGE(( "Checking interdomain synchronization" ));
  TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
  if( rank()==0 ) MESSAGE(( "Error = %e (arb units)", err ));

  if( rank()==0 ) MESSAGE(( "Checking magnetic field divergence" ));
  TIC FAK->compute_div_b_err( field_array ); TOC( compute_div_b_err, 1 );
  TIC err = FAK->compute_rms_div_b_err( field_array ); TOC( compute_rms_div_b_err, 1 );
  if( rank()==0 ) MESSAGE(( "RMS error = %e (charge/volume)", err ));
  TIC FAK->clean_div_b( field_array ); TOC( clean_div_b, 1 );

  // Load fields not initialized by the user

  if( rank()==0 ) MESSAGE(( "Initializing radiation damping fields" ));
  TIC FAK->compute_curl_b( field_array ); TOC( compute_curl_b, 1 );

  if( rank()==0 ) MESSAGE(( "Initializing bound charge density" ));
  TIC FAK->clear_rhof( field_array ); TOC( clear_rhof, 1 );
  LIST_FOR_EACH( sp, species_list ) TIC accumulate_rho_p( field_array, sp ); TOC( accumulate_rho_p, 1 );
  TIC FAK->synchronize_rho( field_array ); TOC( synchronize_rho, 1 );
  TIC FAK->compute_rhob( field_array ); TOC( compute_rhob, 1 );

  // Internal sanity checks

  if( rank()==0 ) MESSAGE(( "Checking electric field divergence" ));

  TIC FAK->compute_div_e_err( field_array ); TOC( compute_div_e_err, 1 );
  TIC err = FAK->compute_rms_div_e_err( field_array ); TOC( compute_rms_div_e_err, 1 );
  if( rank()==0 ) MESSAGE(( "RMS error = %e (charge/volume)", err ));
  TIC FAK->clean_div_e( field_array ); TOC( clean_div_e, 1 );

  if( rank()==0 ) MESSAGE(( "Rechecking interdomain synchronization" ));
  TIC err = FAK->synchronize_tang_e_norm_b( field_array ); TOC( synchronize_tang_e_norm_b, 1 );
  if( rank()==0 ) MESSAGE(( "Error = %e (arb units)", err ));

  if( species_list ) {
    if( rank()==0 ) MESSAGE(( "Uncentering particles" ));
    TIC load_interpolator_array( interpolator_array, field_array ); TOC( load_interpolator, 1 );
  }
  //------------------------------------------------------------------------------------------
  // Conditionally resume profiling with Intel VTune.
  #if defined(VPIC_USE_VTUNE_CENTER_P)
  __itt_resume();
  #endif

  // Conditionally profile marked sections with LikWid..
  #if defined(VPIC_USE_LIKWID_CENTER_P)
  LIKWID_MARKER_START("center_p");
  barrier();
  #endif

  // Conditionally initialize profiling with Armie.
  #if defined(VPIC_USE_ARMIE_CENTER_P)
  __ARMIE_START_TRACE();
  #endif

  for( int iwdn = 0; iwdn < 100; iwdn++ )
  {
    LIST_FOR_EACH( sp, species_list ) TIC uncenter_p( sp, interpolator_array ); TOC( uncenter_p, 1 );
    LIST_FOR_EACH( sp, species_list ) TIC   center_p( sp, interpolator_array ); TOC(   center_p, 1 );
  }

  // Conditionally terminate profiling with Armie.
  #if defined(VPIC_USE_ARMIE_CENTER_P)
  __ARMIE_STOP_TRACE();
  #endif

  // Conditionally profile marked sections with LikWid..
  #if defined(VPIC_USE_LIKWID_CENTER_P)
  barrier();
  LIKWID_MARKER_STOP("center_p");
  #endif

  // Conditionally pause profiling with Intel VTune.
  #if defined(VPIC_USE_VTUNE_CENTER_P)
  __itt_pause();
  #endif
  //------------------------------------------------------------------------------------------
  LIST_FOR_EACH( sp, species_list ) TIC uncenter_p( sp, interpolator_array ); TOC( uncenter_p, 1 );

  if( rank()==0 ) MESSAGE(( "Performing initial diagnostics" ));

  // Let the user to perform diagnostics on the initial condition
  // field(i,j,k).jfx, jfy, jfz will not be valid at this point.
  TIC user_diagnostics(); TOC( user_diagnostics, 1 );

  if( rank()==0 ) MESSAGE(( "Initialization complete" ));
  update_profile( rank()==0 ); // Let the user know how initialization went
}

void
vpic_simulation::finalize( void ) {
  barrier();
  update_profile( rank()==0 );
}
