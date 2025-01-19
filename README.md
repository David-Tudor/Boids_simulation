The commands shown were used to run the programs on an MacBook Air M1. This repository is a duplicate of a private one containing extra files.

First, change directory to this folder.

## Run OpenMP program
Compile using the following command.
> CC=gcc-14 python setup_boids_omp.py build_ext -fi

The command to run the program is shown below, as well as example arguments.
> python run_boids_omp.py NUM_BOIDS NUM_HAWKS NUM_THREADS

> python run_boids_omp.py 10000 15 4


## Run MPI program 
Compile using the following command.
> CC=gcc-14 python setup_boids_mpi.py build_ext -O3 -fi

The command to run the program is shown below, as well as example arguments.
> mpirun -np NUM_THREADS python3 -m run_boids_mpi NUM_BOIDS NUM_HAWKS

> mpirun -np 4 python3 -m run_boids_mpi 10000 15


## Changing the grid size

Grid size is an optional extra argument which can be added to the end of the OpenMP and MPI run commands and defaults to 8 which is the minimum value as that is the vision radius of the boids and hawks.
This changes the size of the grid cells the bounding box is split into. However, larger values than 8 slow the calculation down.

The bounding box will resize so that its dimensions are multiples of the specified grid size and so that there are at least 3 grid cells in each direction.
Recommended values that do not resize the box are 8, 9, 12, 18, 24, 36, 72.

Examples using the grid size argument:
> python run_boids_omp.py 10000 15 4 24

> mpirun -np 4 python3 -m run_boids_mpi 10000 15 24


## Other included files

*David_Tudor_Boids_report* is the report.

*plot_animation_func.py* contains the function to plot an animation from the recorded data.

*example_30k_boids_15_hawks.mp4* shows an example animation for 30,000 boids and 15 hawks.

*graphs_from_log.py* can be run to produce the graphs used in the report using the data file *log_for_graph*. It is run with the following command.

> python graphs_from_log.py
