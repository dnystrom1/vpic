/* 
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Original version
 *
 */

#include "species_advance.h"

/* Private interface *********************************************************/

#if defined(VPIC_USE_AOSOA_P)
void
checkpt_species( const species_t * sp )
{
  ERROR( ( "Need AoSoA implementation." ) );

  #if 0
  // Need to figure out how to do this.  Issues include:
  // 1. Figure out what all this does.
  // 2. Can or must we support reading in a restart file with a different
  //    value of PARTICLE_BLOCK_SIZE?
  // 3. Should we write out and read in particle_t data i.e. construct a
  //    buffer of particle_t data to write out. Then read it in as buffers
  //    of particle_t data and load it into the particle_block_t array.
  CHECKPT( sp, 1 );
  CHECKPT_STR( sp->name );
  checkpt_data( sp->p,
                sp->np    *sizeof(particle_block_t),
                sp->max_np*sizeof(particle_block_t), 1, 1, 128 );
  checkpt_data( sp->pm,
                sp->nm    *sizeof(particle_mover_t),
                sp->max_nm*sizeof(particle_mover_t), 1, 1, 128 );
  CHECKPT_ALIGNED( sp->partition, sp->g->nv+1, 128 );
  CHECKPT_PTR( sp->g );
  CHECKPT_PTR( sp->next );
  #endif
}
#else
void
checkpt_species( const species_t * sp )
{
  CHECKPT( sp, 1 );
  CHECKPT_STR( sp->name );
  checkpt_data( sp->p,
                sp->np    *sizeof(particle_t),
                sp->max_np*sizeof(particle_t), 1, 1, 128 );
  checkpt_data( sp->pm,
                sp->nm    *sizeof(particle_mover_t),
                sp->max_nm*sizeof(particle_mover_t), 1, 1, 128 );
  CHECKPT_ALIGNED( sp->partition, sp->g->nv+1, 128 );
  CHECKPT_PTR( sp->g );
  CHECKPT_PTR( sp->next );
}
#endif

#if defined(VPIC_USE_AOSOA_P)
species_t *
restore_species( void )
{
  species_t * sp;
  ERROR( ( "Need AoSoA implementation." ) );
  // Need to figure out how to do this.  See above.
  return sp;
}
#else
species_t *
restore_species( void )
{
  species_t * sp;
  RESTORE( sp );
  RESTORE_STR( sp->name );
  sp->p  = (particle_t *)      restore_data();
  sp->pm = (particle_mover_t *)restore_data();
  RESTORE_ALIGNED( sp->partition );
  RESTORE_PTR( sp->g );
  RESTORE_PTR( sp->next );
  return sp;
}
#endif

#if defined(VPIC_USE_AOSOA_P)
void
delete_species( species_t * sp )
{
  UNREGISTER_OBJECT( sp );
  FREE_ALIGNED( sp->partition );
  FREE_ALIGNED( sp->pm );
  FREE_ALIGNED( sp->pb );
  FREE( sp->name );
  FREE( sp );
}
#else
void
delete_species( species_t * sp )
{
  UNREGISTER_OBJECT( sp );
  FREE_ALIGNED( sp->partition );
  FREE_ALIGNED( sp->pm );
  FREE_ALIGNED( sp->p );
  FREE( sp->name );
  FREE( sp );
}
#endif

/* Public interface **********************************************************/

int
num_species( const species_t * sp_list )
{
  return sp_list ? sp_list->id+1 : 0;
}

void
delete_species_list( species_t * sp_list )
{
  species_t * sp;
  while( sp_list ) {
    sp = sp_list;
    sp_list = sp_list->next;
    delete_species( sp );
  }
}

species_t *
find_species_id( species_id id,
                 species_t * sp_list )
{
  species_t * sp;
  LIST_FIND_FIRST( sp, sp_list, sp->id==id );
  return sp;
}

species_t *
find_species_name( const char * name,
                   species_t * sp_list )
{
  species_t * sp;
  if( !name ) return NULL;
  LIST_FIND_FIRST( sp, sp_list, strcmp( sp->name, name )==0 );
  return sp;
}

species_t *
append_species( species_t * sp,
                species_t ** sp_list )
{
  if( !sp || !sp_list ) ERROR(( "Bad args" ));
  if( sp->next ) ERROR(( "Species \"%s\" already in a list", sp->name ));
  if( find_species_name( sp->name, *sp_list ) )
    ERROR(( "There is already a species in the list named \"%s\"", sp->name ));
  if( (*sp_list) && sp->g!=(*sp_list)->g )
    ERROR(( "Species \"%s\" uses a different grid from this list", sp->name ));
  sp->id   = num_species( *sp_list );
  sp->next = *sp_list;
  *sp_list = sp;
  return sp;
}

#if defined(VPIC_USE_AOSOA_P)
species_t *
species( const char * name,
         float q,
         float m,
         size_t max_local_np,
         size_t max_local_nm,
         int sort_interval,
         int sort_out_of_place,
         grid_t * g )
{
  species_t * sp;

  int len = name ? strlen(name) : 0;

  if ( ! len )
  {
    ERROR( ( "Cannot create a nameless species." ) );
  }
  if ( ! g )
  {
    ERROR( ( "NULL grid." ) );
  }
  if ( g->nv == 0 )
  {
    ERROR( ( "Allocate grid before defining species." ) );
  }
  if ( max_local_np < 1 )
  {
    max_local_np = 1;
  }
  if ( max_local_nm < 1 )
  {
    max_local_nm = 1;
  }

  MALLOC( sp, 1 );
  CLEAR( sp, 1 );

  MALLOC( sp->name, len + 1 );
  strcpy( sp->name, name );

  sp->q = q;
  sp->m = m;

  // Temporarily assume max_local_np divides evenly by PARTICLE_BLOCK_SIZE for
  // debug and verification purposes.
  // int max_local_nblocks = max_local_np / PARTICLE_BLOCK_SIZE + 1;
  int max_local_nblocks = max_local_np / PARTICLE_BLOCK_SIZE;
  MALLOC_ALIGNED( sp->pb, max_local_nblocks, 128 );
  sp->max_np = max_local_nblocks * PARTICLE_BLOCK_SIZE;
  // sp->max_np = max_local_np;

  MALLOC_ALIGNED( sp->pm, max_local_nm, 128 );
  sp->max_nm = max_local_nm;

  sp->last_sorted       = INT64_MIN;
  sp->sort_interval     = sort_interval;
  sp->sort_out_of_place = sort_out_of_place;

  MALLOC_ALIGNED( sp->partition, g->nv+1, 128 );
  MALLOC_ALIGNED( sp->counts,    g->nv+1, 128 );
  MALLOC_ALIGNED( sp->maxes,     g->nv+1, 128 );
  MALLOC_ALIGNED( sp->copies,    g->nv+1, 128 );

  sp->g = g;   

  // Configure cell statistics arrays.
  // Declarations.
  int ix, iy, iz, n_voxel;

  int part_sum = 0;

  // Initialize arrays to zero.
  for( int i = 0; i < sp->g->nv + 1; i++ )
  {
    sp->partition[i] = 0;
    sp->counts[i]    = 0;
    sp->maxes[i]     = 0;
    sp->copies[i]    = 0;
  }

  // May want to move this up so we can compute a value for max_local_blocks that
  // is integrably divisible by n_voxel.
  DISTRIBUTE_VOXELS( 1, sp->g->nx,
		     1, sp->g->ny,
		     1, sp->g->nz,
		     1,
                     0,
		     1,
                     ix, iy, iz, n_voxel );

  int vox = VOXEL( ix, iy, iz, sp->g->nx, sp->g->ny, sp->g->nz );

  int vox_block_start  = 0;
  int vox_block_number = 0;
  for( int i = 0; i < n_voxel; i++ )
  {
    // DISTRIBUTE( sp->max_np, 1, i, n_voxel, sp->partition[vox], sp->maxes[vox] );
    DISTRIBUTE( max_local_nblocks, 1, i, n_voxel, vox_block_start, vox_block_number );

    sp->partition[vox] = PARTICLE_BLOCK_SIZE * vox_block_start;
    sp->maxes    [vox] = PARTICLE_BLOCK_SIZE * vox_block_number;

    part_sum += sp->maxes[vox];

    NEXT_VOXEL( vox, ix, iy, iz,
                1, sp->g->nx,
                1, sp->g->ny,
                1, sp->g->nz,
                sp->g->nx,
                sp->g->ny,
                sp->g->nz );
  }

  if ( part_sum != sp->max_np )
  {
    WARNING( ( "Number of particles per voxel does not sum to max particles. (%i, %i)",
	       part_sum,
	       sp->max_np ) );
  }

  /* id, next are set by append species */

  REGISTER_OBJECT( sp, checkpt_species, restore_species, NULL );

  return sp;
}
#else
species_t *
species( const char * name,
         float q,
         float m,
         size_t max_local_np,
         size_t max_local_nm,
         int sort_interval,
         int sort_out_of_place,
         grid_t * g )
{
  species_t * sp;
  int len = name ? strlen(name) : 0;

  if( !len ) ERROR(( "Cannot create a nameless species" ));
  if( !g ) ERROR(( "NULL grid" ));
  if( g->nv == 0) ERROR(( "Allocate grid before defining species." ));
  if( max_local_np<1 ) max_local_np = 1;
  if( max_local_nm<1 ) max_local_nm = 1;

  MALLOC( sp, 1 );
  CLEAR( sp, 1 );

  MALLOC( sp->name, len+1 );
  strcpy( sp->name, name );

  sp->q = q;
  sp->m = m;

  MALLOC_ALIGNED( sp->p, max_local_np, 128 );
  sp->max_np = max_local_np;

  MALLOC_ALIGNED( sp->pm, max_local_nm, 128 );
  sp->max_nm = max_local_nm;

  sp->last_sorted       = INT64_MIN;
  sp->sort_interval     = sort_interval;
  sp->sort_out_of_place = sort_out_of_place;

  MALLOC_ALIGNED( sp->partition, g->nv+1, 128 );
  MALLOC_ALIGNED( sp->counts,    g->nv+1, 128 );
  MALLOC_ALIGNED( sp->maxes,     g->nv+1, 128 );
  MALLOC_ALIGNED( sp->copies,    g->nv+1, 128 );

  sp->g = g;   

  // Configure cell statistics arrays.
  // Declarations.
  int ix, iy, iz, n_voxel;

  int part_sum = 0;

  // Initialize arrays to zero.
  for( int i = 0; i < sp->g->nv + 1; i++ )
  {
    sp->partition[i] = 0;
    sp->counts[i]    = 0;
    sp->maxes[i]     = 0;
    sp->copies[i]    = 0;
  }

  DISTRIBUTE_VOXELS( 1, sp->g->nx,
		     1, sp->g->ny,
		     1, sp->g->nz,
		     1,
                     0,
		     1,
                     ix, iy, iz, n_voxel );

  int vox = VOXEL( ix, iy, iz, sp->g->nx, sp->g->ny, sp->g->nz );

  for( int i = 0; i < n_voxel; i++ )
  {
    DISTRIBUTE( sp->max_np, 1, i, n_voxel, sp->partition[vox], sp->maxes[vox] );

    part_sum += sp->maxes[vox];

    NEXT_VOXEL( vox, ix, iy, iz,
                1, sp->g->nx,
                1, sp->g->ny,
                1, sp->g->nz,
                sp->g->nx,
                sp->g->ny,
                sp->g->nz );
  }

  if ( part_sum != sp->max_np )
  {
    WARNING( ( "Number of particles per voxel does not sum to max particles. (%i, %i)",
	       part_sum,
	       sp->max_np ) );
  }

  /* id, next are set by append species */

  REGISTER_OBJECT( sp, checkpt_species, restore_species, NULL );
  return sp;
}
#endif

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
