#define IN_spa

#include "../species_advance.h"

//----------------------------------------------------------------------------//
// Top level function to select and call particle advance function using the
// desired particle advance abstraction.  Currently, the only abstraction
// available is the pipeline abstraction.
//----------------------------------------------------------------------------//

void
uncenter_p_plus( species_t * RESTRICT sp,
                 accumulator_array_t * RESTRICT aa,
                 const interpolator_array_t * RESTRICT ia )
{
  // Once more options are available, this should be conditionally executed
  // based on user choice.
  uncenter_p_plus_pipeline( sp, aa, ia );
}
