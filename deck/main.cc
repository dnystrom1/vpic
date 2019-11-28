/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Revised from earlier V4PIC versions
 *
 */

#include "vpic/vpic.h"

// Use this for Intel and VTune.
#if defined(VPIC_USE_VTUNE_ADVANCE)
#include "ittnotify.h"
#endif

// Use this for LikWid.
#if defined(VPIC_USE_LIKWID_ADVANCE) || defined(VPIC_USE_LIKWID_CENTER_P)
#include <likwid-marker.h>
#endif

// Use this for Armie.
#if defined(VPIC_USE_ARMIE) || defined(VPIC_USE_ARMIE_ADVANCE)
#define __ARMIE_START_TRACE() { asm volatile (".inst 0x2520e020"); }
#define __ARMIE_STOP_TRACE() { asm volatile (".inst 0x2520e040"); }
#endif

// The simulation variable is set up this way so both the checkpt
// service and main can see it.  This allows main to find where
// the restored objects are after a restore.
vpic_simulation * simulation = NULL;


/**
 * @brief Function to checkout main simulation object for restarting
 *
 * @param _simulation Simulation object to checkpoint
 */
void checkpt_main(vpic_simulation** _simulation)
{
    CHECKPT_PTR( simulation );
}

/**
 * @brief Function to handle the recovery of the main simulation object at
 * restart
 *
 * @return Returns a double pointer (**) to the simulation object
 */
vpic_simulation** restore_main(void)
{
    RESTORE_PTR( simulation );
    return &simulation;
}

/**
 * @brief Main checkpoint function to trigger a full checkpointing
 *
 * @param fbase File name base for dumping
 * @param tag File tag to label what this checkpoint is (often used: time step)
 */
void checkpt(const char* fbase, int tag)
{
    char fname[256];
    if( !fbase ) ERROR(( "NULL filename base" ));
    sprintf( fname, "%s.%i.%i", fbase, tag, world_rank );
    if( world_rank==0 ) log_printf( "*** Checkpointing to \"%s\"\n", fbase );
    checkpt_objects( fname );
}

/**
 * @brief Program main which triggers a vpic run
 *
 * @param argc Standard arguments
 * @param argv Standard arguments
 *
 * @return Application error code
 */
int main(int argc, char** argv)
{

    // Initialize underlying threads and services
    boot_services( &argc, &argv );

    // TODO: this would be better if it was bool-like in nature
    const char * fbase = strip_cmdline_string(&argc, &argv, "--restore", NULL);

    // Conditionally initialize profiling with Armie.
    #if defined(VPIC_USE_ARMIE)
    __ARMIE_START_TRACE();
    #endif

    // Conditionally initialize profiling with LikWid.
    #if defined(VPIC_USE_LIKWID_ADVANCE) || defined(VPIC_USE_LIKWID_CENTER_P)
    LIKWID_MARKER_INIT;
    #endif

    // Detect if we should perform a restore as per the user request
    if( fbase )
    {

        // We are restoring from a checkpoint.  Determine checkpt file
        // for this process, restore all the objects in that file,
        // wait for all other processes to finishing restoring (such
        // that communication within reanimate functions is safe),
        // reanimate all the objects and issue a final barrier to
        // so that all processes come of a restore together.
        if( world_rank==0 ) log_printf( "*** Restoring from \"%s\"\n", fbase );
        char fname[256];
        sprintf( fname, "%s.%i", fbase, world_rank );
        restore_objects( fname );
        mp_barrier();
        reanimate_objects();
        mp_barrier();

    }
    else // We are initializing from scratch.
    {
        // Perform basic initialization
        if( world_rank==0 )
        {
            log_printf( "*** Initializing\n" );
        }
        simulation = new vpic_simulation();
        simulation->initialize( argc, argv );
        REGISTER_OBJECT( &simulation, checkpt_main, restore_main, NULL );
    }

    // Do any post init/restore simulation modifications

    // Detect if the "modify" option is passed, which allows users to change
    // options (such as quota, num_step, etc) when restoring
    fbase = strip_cmdline_string( &argc, &argv, "--modify", NULL );
    if( fbase )
    {
        if( world_rank==0 ) log_printf( "*** Modifying from \"%s\"\n", fbase );
        simulation->modify( fbase );
    }

    #define VPIC_NORMAL_RUN
    #ifdef VPIC_NORMAL_RUN
    // Conditionally resume profiling with Intel VTune.
    #if defined(VPIC_USE_VTUNE_ADVANCE)
    __itt_resume();
    #endif

    // Conditionally profile marked sections with LikWid..
    #if defined(VPIC_USE_LIKWID_ADVANCE)
    LIKWID_MARKER_START("advance");
    mp_barrier();
    #endif

    // Conditionally initialize profiling with Armie.
    #if defined(VPIC_USE_ARMIE_ADVANCE)
    __ARMIE_START_TRACE();
    #endif

    // Perform the main simulation
    if( world_rank==0 ) log_printf( "*** Advancing\n" );
    double elapsed = wallclock();

    // Call the actual advance until it's done
    // TODO: Can we make this into a bounded loop
    while( simulation->advance() );

    elapsed = wallclock() - elapsed;

    // Conditionally terminate profiling with Armie.
    #if defined(VPIC_USE_ARMIE_ADVANCE)
    __ARMIE_STOP_TRACE();
    #endif

    // Conditionally profile marked sections with LikWid..
    #if defined(VPIC_USE_LIKWID_ADVANCE)
    mp_barrier();
    LIKWID_MARKER_STOP("advance");
    #endif

    // Conditionally pause profiling with Intel VTune.
    #if defined(VPIC_USE_VTUNE_ADVANCE)
    __itt_pause();
    #endif

    // Report run time information on rank 0
    if( world_rank==0 )
    {
        // Calculate time info
        int  s = (int)elapsed, m  = s/60, h  = m/60, d  = h/24, w = d/ 7;
        s -= m*60;
        m -= h*60;
        h -= d*24;
        d -= w*7;

        log_printf( "*** Done (%gs / %iw:%id:%ih:%im:%is elapsed)\n",
                elapsed, w, d, h, m, s );
    }
    #endif

    if( world_rank==0 ) log_printf( "*** Cleaning up\n" );

    // Conditionally terminate profiling with LikWid..
    #if defined(VPIC_USE_LIKWID_ADVANCE) || defined(VPIC_USE_LIKWID_CENTER_P)
    LIKWID_MARKER_CLOSE;
    #endif

    // Conditionally terminate profiling with Armie.
    #if defined(VPIC_USE_ARMIE)
    __ARMIE_STOP_TRACE();
    #endif

    // Perform Clean up, including de-registering objects
    UNREGISTER_OBJECT( &simulation );
    simulation->finalize();
    delete simulation;

    // Check everything went well
    if( world_rank==0 ) log_printf( "normal exit\n" );

    halt_services();
    return 0;
}
