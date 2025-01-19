import sys
import numpy as np
from mpi4py import MPI
from boids_mpi import main
from plot_animation_func import plot_animation

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
MANAGER = 0

if int(len(sys.argv)) == 3 or int(len(sys.argv)) == 4:

    if int(len(sys.argv)) == 3:
        grid_size = 8
        return_code = main(int(sys.argv[1]), int(sys.argv[2]), grid_size)
    else:
        return_code = main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

    if rank == MANAGER:
        if return_code == 1:
            print("At least one boid must be simulated.")
        elif return_code == 2:
            print("The number of hawks cannot be negative.")
        elif return_code == 3:
            print("At least one thread must be used.")
        else:

            # Load output.
            output = np.load("output_mpi.npz")
            (run_parameters, t_record, data_record, hawk_record, cumulative_num_eaten) = (output['arr_0'], output['arr_1'], output['arr_2'], output['arr_3'], output['arr_4'])
            (computation_time, num_boids, num_hawks, threads, grid_size, max_x, max_y) = run_parameters

            print(f"Computation time was: {round(computation_time, 3)}s")
            sys.stdout.flush()
            plot_animation(t_record, data_record, hawk_record, cumulative_num_eaten, max_x, max_y, computation_time)

else:
    if rank == MANAGER:
        print(f"Usage: {sys.argv[0]} <NUM_BOIDS> <NUM_HAWKS> <GRID_SIZE>\nNote that <GRID_SIZE> is an optional argument.")

if rank == MANAGER:
    print("Program end.")
