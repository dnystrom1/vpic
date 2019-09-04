// Written by:
//   Dave Nystrom
//   Applications Readiness Team
//   Group HPC-ENV
//   HPC Division
//   Los Alamos National Laboratory

#include "species_advance_pipeline.h"

//----------------------------------------------------------------------------//
// Determine the first and last voxel for each pipeline and the number of
// voxels for each pipeline.
//----------------------------------------------------------------------------//

void
distribute_voxels( int& first_ix, int& first_iy, int& first_iz, int& n_vox,
		   const species_t *sp, int pipeline_rank, int n_pipeline,
		   int n_part, int first_part, int last_part )
{
  int previous_vox; // Index of previous voxel.
  int    first_vox; // Index of first voxel for this thread.
  int     last_vox; // Index of last  voxel for this thread.
  int          vox; // Index of current voxel.

  int n_voxel  = 0; // Number of voxels in this MPI domain.
  int sum_part = 0;
  int ix       = 0;
  int iy       = 0;
  int iz       = 0;

  first_ix = 0;
  first_iy = 0;
  first_iz = 0;

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
}

//----------------------------------------------------------------------------//
// Function to use when trying to confuse compiler so it will not optimize
// away operations which do not change results.
//----------------------------------------------------------------------------//

void
get_constants( float &wdn_zero, float &wdn_one, int &nx )
{
  if ( nx < 1024 )
  {
    wdn_zero = 0.0;
    wdn_one  = 1.0;
  }

  else
  {
    wdn_zero = 100.0;
    wdn_one  = 1000.0;
  }
}
