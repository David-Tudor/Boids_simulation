import sys
import numpy as np
from boids_omp import main
from plot_animation_func import plot_animation

if int(len(sys.argv)) == 4 or int(len(sys.argv)) == 5:

    if int(len(sys.argv)) == 4:
        grid_size = 8
        return_code = main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), grid_size)
    else:
        return_code = main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

    if return_code == 1:
        print("At least one boid must be simulated.")
    elif return_code == 2:
        print("The number of hawks cannot be negative.")
    elif return_code == 3:
        print("At least one thread must be used.")
    else:

        # Load output.
        output = np.load("output_omp.npz")
        (run_parameters, t_record, data_record, hawk_record, cumulative_num_eaten) = (output['arr_0'], output['arr_1'], output['arr_2'], output['arr_3'], output['arr_4'])
        (computation_time, num_boids, num_hawks, threads, grid_size, max_x, max_y) = run_parameters

        print(f"Computation time was: {round(computation_time, 3)}s")
        plot_animation(t_record, data_record, hawk_record, cumulative_num_eaten, max_x, max_y)

else:
    print(f"Usage: {sys.argv[0]} <NUM_BOIDS> <NUM_HAWKS> <NUM_THREADS> <GRID_SIZE>\nNote that <GRID_SIZE> is an optional argument.")

print("Program end.")