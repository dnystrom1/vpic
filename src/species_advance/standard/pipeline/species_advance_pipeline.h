// Written by:
//   Dave Nystrom
//   Applications Readiness Team
//   Group HPC-ENV
//   HPC Division
//   Los Alamos National Laboratory

#ifndef _species_advance_pipeline_h_
#define _species_advance_pipeline_h_

#include "../../species_advance.h"

//----------------------------------------------------------------------------//
// Declare methods.
//----------------------------------------------------------------------------//

void
distribute_voxels( int& first_ix, int& first_iy, int& first_iz, int& n_vox,
		   const species_t *sp, int pipeline_rank, int n_pipeline,
		   int n_part, int first_part, int last_part );

void
get_constants( float& wdn_zero, float& wdn_one, int& nx );

#endif // _species_advance_pipeline_h_
