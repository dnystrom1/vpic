#define IN_sfa

#include "sfa_private.h"

//----------------------------------------------------------------------------//
// Include various programming model implementation files. For now, this is
// just the pipeline model. When there are more models, probably want these
// to be conditionally included.
//----------------------------------------------------------------------------//

#include "compute_curl_b_pipeline.cc"

//----------------------------------------------------------------------------//
// Top level function to select and call the proper compute_curl_b function.
//----------------------------------------------------------------------------//

void
compute_curl_b( field_array_t * RESTRICT fa )
{
  if ( !fa )
  {
    ERROR( ( "Bad args" ) );
  }

  // Conditionally execute this when more abstractions are available.
  compute_curl_b_pipeline( fa );
}
