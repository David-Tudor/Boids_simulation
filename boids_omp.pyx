from libc.math cimport sqrt, cos, sin
from cython.parallel cimport prange
cimport openmp

import os
os.environ ["MKL_NUM_THREADS"] = "1"
os.environ ["NUMEXPR_NUM_THREADS"] = "1"
os.environ ["OMP_NUM_THREADS"] = "1"
import numpy as np
import cython


def init_data(memory_view, num, speed, max_x, max_y):
    """
    Initalise memory_view with num boids at random x,y values in a rectangle between (0,0) and (max_x, max_y).
    Each boid moves at the same speed (=speed) in a random direction.
    Returns the filled memory_view: [[float]]
    memory_view: [[float]]
    num: int
    speed: float
    max_x: int
    max_y: int
    """
    memory_view = np.random.rand(num, 4) * [max_x, max_y, 0, 2*np.pi]
    for i in range(num):
        memory_view[i,2] = speed * cos(memory_view[i,3])
        memory_view[i,3] = speed * sin(memory_view[i,3]) 
    return memory_view


cpdef double coord_diff(double q1, double q2, int max_q) noexcept nogil:
    """
    Returns the difference (double) between coordinates q1 and q2, accounting for periodic boundaries.
    q1: double
    q2: double
    max_q: int
    """
    if abs(q2 - q1) <= abs(q2 - max_q - q1):
        return q2 - q1
    else:
        return q2 - max_q - q1


cpdef bint in_vision(double dist_squared, int nearby_threshold_squared, double x_diff, double y_diff, double vx, double vy, double inverse_sqrt_speed, double cos_vision_ang) noexcept nogil:
    """
    Test if in vision with a distance and angle (dot product) check.
    Return True if in vision, else False.
    dist_squared: double
    nearby_threshold_squared: int
    x_diff: double
    y_diff: double
    vx: double
    vy: double
    inverse_sqrt_speed: double
    cos_vision_ang: double
    """
    if dist_squared > nearby_threshold_squared or dist_squared == 0:
        return False

    if (x_diff*vx+y_diff*vy)/sqrt(dist_squared)*inverse_sqrt_speed < cos_vision_ang:
        return False
    
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] apply_periodic_boundaries(double[:] entity, int max_x, int max_y) noexcept nogil:
    """
    Moves an entity to the other side of the bounding box if it has moved outside of the box. Returns an array of 4 doubles.
    entity: [double]
    max_x: int
    max_y: int
    """
    entity[0] = entity[0] % max_x
    entity[1] = entity[1] % max_y
    return entity


cpdef int key_value(int x, int y, int len_grid_x, int len_grid_y) noexcept nogil:
    """
    For a given grid cell (x,y), returns an int unique to the grid cell.
    x: int
    y: int
    len_grid_x: int
    len_grid_y: int
    """
    return y%len_grid_y * len_grid_x + x%len_grid_x



@cython.boundscheck(False)
def main( int num_boids, int num_hawks, int threads, int grid_size ):
    """
    Main function of program which simulates the boids and hawks. Returns 0 if successful.
    
    Output:
    run parameters (computation_time, num_boids, num_hawks, threads, grid_size, max_x, max_y) are recorded in a new line of log_file.
    They are also written to output_omp.npz along with t_record, data_record, hawk_record, cumulative_num_eaten.
    t_record is an array of times: [double]
    data_record contains an array for each time with an array of all the boid positions and velocities (x,y,vx,vy): [[[double]]]
    hawk_record contains an array for each time with an array of all the hawk positions and velocities (x,y,vx,vy): [[[double]]]
    cumulative_num_eaten records the cumulative number of boids eaten for each time: [int]
    """
    
    start_time = openmp.omp_get_wtime()
    file = "output_omp"
    log_file = "log_file"
    
    # Return if input arguments are invalid.
    if num_boids < 1:
        return 1
    elif num_hawks < 0:
        return 2
    elif threads < 1:
        return 3

    cdef:
        # Setup variables.
        double t = 0
        double dt = 0.1
        double t_max = 40
        double initial_spd = 1.8
        double hawk_speed = 2.3
        int nearby_threshold = 8
        double vision_ang = 2.4 # 0 to pi
        int close_threshold = 3
        int max_x = 360
        int max_y = 216

        # Weight of each rule.
        double alignment_factor = 0.8
        double cohesion_factor = 0.3
        double separation_factor = 100
        double flee_factor = 0.5
        double hawk_factor = 5.0

        # Reduction variables.
        int num_nearby_boids = 0
        int num_nearby_hawks = 0
        double sum_nearby_x_diffs = 0
        double sum_nearby_y_diffs = 0
        double sum_nearby_vxs = 0
        double sum_nearby_vys = 0
        double sum_close_x_diffs = 0
        double sum_close_y_diffs = 0
        double sum_hawk_x_diffs = 0
        double sum_hawk_y_diffs = 0

        # Vision calculation variables.
        double x1 = 0.0
        double y1 = 0.0
        double x2 = 0.0
        double y2 = 0.0
        double vx = 0.0
        double vy = 0.0
        double x_diff = 0.0
        double y_diff = 0.0
        double dist_squared = 0.0
        int nearby_threshold_squared = nearby_threshold**2
        int close_threshold_squared = close_threshold**2
        double inverse_sqrt_speed
        double cos_vision_ang = cos(vision_ang)
        double speed

        # Indices.
        int i
        int j
        int boid_i
        int boid_j
        int hawk_i
        int cumulative_lens_i
        int resultant_i
        int cell_key
        int grid_i
        int centre_cell_i

        # Other variables.
        int thread_num
        int num_grid_boids
        int num_grid_hawks
        int cell_len
        int centre_cell_len = 0
        bint are_others_nearby
        bint is_fleeing
        double min_dist_squared = nearby_threshold_squared
        double best_x_diff = 0
        double best_y_diff = 0
        double hawk_eat_threshold_squared = 0.5**2
        int num_eaten = 0

        # Arrays.
        double[:,::1] data = np.zeros((num_boids, 4), dtype=np.double)
        double[:,::1] hawk_data = np.zeros((num_hawks, 4), dtype=np.double)
        double [:,::1] new_data = data
        double [:,::1] new_hawk_data = hawk_data
        double[::1] boid = data[0] 
        double[::1] hawk = hawk_data[0] 
        int[:,::1] nearby_boid_is = np.zeros((num_boids, threads), dtype=np.int32)
        int[:,::1] nearby_hawk_is = np.zeros((num_hawks, threads), dtype=np.int32)
        int[::1] is_boid_j_eaten = np.zeros(num_boids, dtype=np.int32)


    if grid_size < nearby_threshold:
        grid_size = nearby_threshold
        print(f"grid_size must not be less than nearby_threshold. Setting grid_size to {nearby_threshold}.")

    # Make box a multiple of grid_size.
    if max_x % grid_size != 0:
        old_max = max_x
        max_x += grid_size - max_x % grid_size
        print(f"Need to split the box into an integer number of grid cells: changed max_x from {old_max} to {max_x}.")
    if max_y % grid_size != 0:
        old_max = max_y
        max_y += grid_size - max_y % grid_size
        print(f"Need to split the box into an integer number of grid cells: changed max_y from {old_max} to {max_y}.")

    # The grid must be at least 3x3.
    if max_x / grid_size < 3:
        old_max = max_x
        max_x = 3 * grid_size
        print(f"Need at least 3 grid cells in the x direction: changed max_x from {old_max} to {max_x}.")
    if max_y / grid_size < 3:
        old_max = max_y
        max_y = 3 * grid_size
        print(f"Need at least 3 grid cells in the y direction: changed max_y from {old_max} to {max_y}.")

    print(f"Calculating. Number of boids = {num_boids}, number of hawks = {num_hawks}, number of threads = {threads}, grid size = {grid_size}.")


    # Grid variables defined after box is resized.
    cdef:
        int grid_x
        int grid_y
        int search_key
        int len_grid_x = max_x//grid_size
        int len_grid_y = max_y//grid_size
        int num_grid_cells = len_grid_x * len_grid_y
        int[::1] indices_list = np.zeros(num_boids, dtype=np.int32)
        int[::1] lens_list = np.zeros(num_grid_cells, dtype=np.int32)
        int[::1] cumulative_lens = np.zeros(num_grid_cells, dtype=np.int32)
        int[::1] num_boids_per_cell = np.zeros(num_grid_cells, dtype=np.int32)
        int[::1] hawk_indices_list = np.zeros(num_boids, dtype=np.int32)
        int[::1] hawk_lens_list = np.zeros(num_grid_cells, dtype=np.int32)
        int[::1] hawk_cumulative_lens = np.zeros(num_grid_cells, dtype=np.int32)
        int[::1] num_hawks_per_cell = np.zeros(num_grid_cells, dtype=np.int32)

    # Initialise positions and velocities of boids and hawks.
    data = init_data(data, num_boids, initial_spd, max_x, max_y)
    hawk_data = init_data(hawk_data, num_hawks, hawk_speed, max_x, max_y)

    # Initialise records.
    data_record = [np.array(data)[:,:2]]
    hawk_record = [np.array(hawk_data)[:,:2]]
    eaten_i_record = []
    cumulative_num_eaten = [0]    



    # Start simulation.
    while t < t_max:
        t += dt
        new_data = data
        new_hawk_data = hawk_data
        num_eaten = 0

        # Divide the bounding box into square sections of length "grid_size" to make a grid of cells, then flatten and give each cell a unique key_value.
        # Make a list of which cell each boid is in and reorder so the cell keys are ascending.
        # From this list, record the index each cell starts at and its length.
        flat_grid =[]
        for j in range(num_boids):
            boid = data[j]
            flat_grid.append( (int(key_value(int(boid[0]//grid_size), int(boid[1]//grid_size), len_grid_x, len_grid_y)), j) )

        flat_grid = np.array(flat_grid)

        # The tuples are sorted by ascending grid values.
        sort_grid = flat_grid[flat_grid[:, 0].argsort()]

        # Pull out the ordered boid indices.
        indices_list = np.int32(sort_grid[:,1])

        # Count the frequency of each grid value showing how many boids are in each grid cell.
        lens_list = np.int32(np.bincount(sort_grid[:,0], minlength=num_grid_cells))

        # Take the cumulative sum of the lengths to give the starting index of each grid cell.
        cumulative_lens = np.int32(np.append(0, np.cumsum(lens_list)[:-1]))


        # Create a similar grid for hawks.
        if num_hawks != 0:
            hawk_grid =[]
            for j in range(num_hawks):
                hawk = hawk_data[j]
                hawk_grid.append( (int(key_value(int(hawk[0]//grid_size), int(hawk[1]//grid_size), len_grid_x, len_grid_y)), j) )

            hawk_grid = np.array(hawk_grid)
            sort_hawk_grid = hawk_grid[hawk_grid[:, 0].argsort()]
            hawk_indices_list = np.int32(sort_hawk_grid[:,1])
            hawk_lens_list = np.int32(np.bincount(sort_hawk_grid[:,0], minlength=num_grid_cells))
            hawk_cumulative_lens = np.int32(np.append(0, np.cumsum(hawk_lens_list)[:-1]))


        # Loop over every cell in the grid
        for cell_key in prange(num_grid_cells, nogil=True, num_threads=threads, schedule="dynamic"):

            thread_num = openmp.omp_get_thread_num()
            grid_x = cell_key%len_grid_x
            grid_y = cell_key//len_grid_x
            num_boids_per_cell[cell_key] = 0
            num_hawks_per_cell[cell_key] = 0

            # Look up the nearby boids & hawks in the 3x3 of cells around the selected cell and store them in nearby_boid_is and nearby_hawk_is.
            for i in range(-1,2):
                for j in range(-1,2):
                    
                    # Get the starting index & length of the nearby cell in the boid grid.
                    search_key = key_value(grid_x+i, grid_y+j, len_grid_x, len_grid_y)
                    cumulative_lens_i = cumulative_lens[search_key]
                    cell_len = lens_list[search_key]

                    # Record where the boids in the central cell of the 3x3 are stored in nearby_boid_is.
                    if i == 0 and j == 0:
                        centre_cell_i = num_boids_per_cell[cell_key]
                        centre_cell_len = cell_len

                    # Fill nearby_boid_is with the boids in the nearby cell.
                    for grid_i in range(cell_len):
                        resultant_i = cumulative_lens_i + grid_i
                        nearby_boid_is[num_boids_per_cell[cell_key]][thread_num] = indices_list[resultant_i]
                        num_boids_per_cell[cell_key] += 1

                    # Do the same for hawks in the hawk grid.
                    if num_hawks != 0:
                        # Search hawk grid with the same search_key.
                        cumulative_lens_i = hawk_cumulative_lens[search_key]
                        cell_len = hawk_lens_list[search_key]

                        # Get all the hawks in the selected cell.
                        for grid_i in range(cell_len):
                            resultant_i = cumulative_lens_i + grid_i
                            nearby_hawk_is[num_hawks_per_cell[cell_key]][thread_num] = hawk_indices_list[resultant_i]
                            num_hawks_per_cell[cell_key] += 1

            num_grid_boids = num_boids_per_cell[cell_key]
            num_grid_hawks = num_hawks_per_cell[cell_key]


            # Loop over all the boids in the central cell.
            for i in range(centre_cell_len):

                # Get boid in central cell.
                resultant_i = centre_cell_i + i
                boid_i = nearby_boid_is[resultant_i][thread_num]

                # Skip if eaten.
                if is_boid_j_eaten[boid_i] == 1:
                        continue

                # Reset variables
                num_nearby_boids = 0
                num_nearby_hawks = 0
                sum_nearby_x_diffs = 0
                sum_nearby_y_diffs = 0
                sum_nearby_vxs = 0
                sum_nearby_vys = 0
                sum_close_x_diffs = 0
                sum_close_y_diffs = 0
                sum_hawk_x_diffs = 0
                sum_hawk_y_diffs = 0
                are_others_nearby = False
                is_fleeing = False
                
                x1 = data[boid_i, 0]
                y1 = data[boid_i, 1]
                vx = data[boid_i, 2]
                vy = data[boid_i, 3]
                
                if vx != 0.0 and vy != 0.0:
                    inverse_sqrt_speed = 1/sqrt(vx*vx+vy*vy)
                else:
                    inverse_sqrt_speed = 1/initial_spd


                if num_hawks != 0:
                    for j in range(num_grid_hawks):
                        hawk_i = nearby_hawk_is[j][thread_num]
                        x2 = hawk_data[hawk_i, 0]
                        y2 = hawk_data[hawk_i, 1]
                        x_diff = coord_diff(x1, x2, max_x)
                        y_diff = coord_diff(y1, y2, max_y)
                        dist_squared = x_diff*x_diff + y_diff*y_diff

                        # Vision check (distance and angle).
                        if not in_vision(dist_squared, nearby_threshold_squared, x_diff, y_diff, vx, vy, inverse_sqrt_speed, cos_vision_ang):
                            continue

                        is_fleeing = True
                        num_nearby_hawks += 1
                        sum_hawk_x_diffs += x_diff
                        sum_hawk_y_diffs += y_diff

                # Flee from hawk if in vision, else flock.
                if is_fleeing:
                    # Change velocity to move away from average position of hawks.
                    new_data[boid_i,2] += -flee_factor * (sum_hawk_x_diffs / num_nearby_hawks) * dt
                    new_data[boid_i,3] += -flee_factor * (sum_hawk_y_diffs / num_nearby_hawks) * dt

                else:
                    # Compare boid_i to every other boid in the nearby 3x3 of cells.
                    for j in range(num_grid_boids):
                        boid_j = nearby_boid_is[j][thread_num]

                        # Skip itself and eaten boids.
                        if boid_i == boid_j or is_boid_j_eaten[boid_j] == 1:
                            continue
                        
                        x2 = data[boid_j, 0]
                        y2 = data[boid_j, 1]
                        x_diff = coord_diff(x1, x2, max_x)
                        y_diff = coord_diff(y1, y2, max_y)
                        dist_squared = x_diff*x_diff + y_diff*y_diff

                        # Vision check (distance and angle).
                        if not in_vision(dist_squared, nearby_threshold_squared, x_diff, y_diff, vx, vy, inverse_sqrt_speed, cos_vision_ang):
                            continue

                        if not are_others_nearby:
                            are_others_nearby = True

                        # Reductions.
                        num_nearby_boids += 1
                        sum_nearby_x_diffs += x_diff
                        sum_nearby_y_diffs += y_diff
                        sum_nearby_vxs += data[boid_j, 2]
                        sum_nearby_vys += data[boid_j, 3]

                        if dist_squared < close_threshold_squared:
                            # Divide by dist_squared as diffs are proportional to distance.
                            sum_close_x_diffs += x_diff/dist_squared
                            sum_close_y_diffs += y_diff/dist_squared

                        
                    if are_others_nearby:
                        # Alignment: change velocity to be closer to the average velocity of nearby boids.
                        # Cohesion: change velocity to move towards the average position of nearby boids.
                        # Separation: change velocity to move away from the average position of close boids. 
                        new_data[boid_i,2] += (alignment_factor * (sum_nearby_vxs / num_nearby_boids - data[boid_i,2]) + cohesion_factor * (sum_nearby_x_diffs / num_nearby_boids) + -separation_factor * (sum_close_x_diffs / num_nearby_boids)) * dt
                        new_data[boid_i,3] += (alignment_factor * (sum_nearby_vys / num_nearby_boids - data[boid_i,3]) + cohesion_factor * (sum_nearby_y_diffs / num_nearby_boids) + -separation_factor * (sum_close_y_diffs / num_nearby_boids)) * dt



        # Hawk loop.
        for hawk_i in range(num_hawks):
            
            hawk = hawk_data[hawk_i]
            x1 = hawk[0]
            y1 = hawk[1]
            vx = hawk[2]
            vy = hawk[3]
            
            if vx != 0.0 and vy != 0.0:
                inverse_sqrt_speed = 1/sqrt(vx*vx+vy*vy)
            else:
                inverse_sqrt_speed = 1/initial_spd

            grid_x = int(hawk[0]//grid_size)
            grid_y = int(hawk[1]//grid_size)

            min_dist_squared = nearby_threshold_squared
            best_x_diff = 0
            best_y_diff = 0

            # Search the 3x3 grid of nearby cells.
            for i in range(-1,2):
                for j in range(-1,2):
                    search_key = key_value(grid_x+i, grid_y+j, len_grid_x, len_grid_y)
                    cumulative_lens_i = cumulative_lens[search_key]
                    cell_len = lens_list[search_key]

                    # Get all the boids in the selected cell.
                    for grid_i in range(cell_len):
                        resultant_i = cumulative_lens_i + grid_i
                        boid_j = indices_list[resultant_i]

                        # Skip eaten boids.
                        if is_boid_j_eaten[boid_j] == 1:
                            continue

                        x2 = data[boid_j][0]
                        y2 = data[boid_j][1]
                        x_diff = coord_diff(x1, x2, max_x)
                        y_diff = coord_diff(y1, y2, max_y)
                        dist_squared = x_diff*x_diff + y_diff*y_diff

                        # Vision check (distance and angle).
                        if not in_vision(dist_squared, nearby_threshold_squared, x_diff, y_diff, vx, vy, inverse_sqrt_speed, cos_vision_ang):
                            continue

                        # As in vision, check which is closest.
                        if dist_squared < min_dist_squared:
                            min_dist_squared = dist_squared
                            best_x_diff = x_diff
                            best_y_diff = y_diff

                        # Eat very close boids.
                        if dist_squared < hawk_eat_threshold_squared:
                            is_boid_j_eaten[boid_j] = 1
                            eaten_i_record.append(boid_j)
                            num_eaten += 1


            # Move hawk towards closest boid.
            if min_dist_squared != nearby_threshold_squared:
                new_hawk_data[hawk_i,2] +=  hawk_factor * best_x_diff * dt
                new_hawk_data[hawk_i,3] +=  hawk_factor * best_y_diff * dt
 

        # Update position with velocity.
        for i in prange(num_boids, nogil=True, num_threads=threads):
            if is_boid_j_eaten[i] == 0:
                # Make speeds the same.
                speed = sqrt(new_data[i,2]*new_data[i,2] + new_data[i,3]*new_data[i,3])
                if speed == 0:
                    speed = initial_spd
                data[i,2], data[i,3] = new_data[i,2]*initial_spd/speed, new_data[i,3]*initial_spd/speed

                data[i,0] = data[i,0] + data[i,2] * dt
                data[i,1] = data[i,1] + data[i,3] * dt

                # Periodic boundaries.
                data[i] = apply_periodic_boundaries(data[i], max_x, max_y)
        
        for i in prange(num_hawks, nogil=True, num_threads=threads):
            # Make speeds the same.
            speed = sqrt(new_hawk_data[i,2]*new_hawk_data[i,2] + new_hawk_data[i,3]*new_hawk_data[i,3])
            hawk_data[i,2], hawk_data[i,3] = new_hawk_data[i,2]*hawk_speed/speed, new_hawk_data[i,3]*hawk_speed/speed

            hawk_data[i,0] = hawk_data[i,0] + new_hawk_data[i,2] * dt
            hawk_data[i,1] = hawk_data[i,1] + new_hawk_data[i,3] * dt

            # Periodic boundaries.
            hawk_data[i] = apply_periodic_boundaries(hawk_data[i], max_x, max_y)


        # Record data.
        data_record.append(np.array(data)[:,:2])
        hawk_record.append(np.array(hawk_data)[:,:2])
        cumulative_num_eaten.append(num_eaten + cumulative_num_eaten[len(cumulative_num_eaten)-1])

    # End of time simulation.
    computation_time = openmp.omp_get_wtime() - start_time


    # Finalise records.
    t_record = np.arange(0,t_max,dt)
    data_record = np.array(data_record)
    hawk_record = np.array(hawk_record)

    # For all times, replace ith entry with reordered_indices[i]th entry. 
    # Then boids at the start of the list are eaten first so array can be sliced to plot different colours.
    reordered_indices = np.arange(0,num_boids,1)
    reordered_indices[:len(eaten_i_record)] = eaten_i_record
    data_record = data_record[:,reordered_indices]

    # Save records.
    run_parameters = np.array([computation_time, num_boids, num_hawks, threads, grid_size, max_x, max_y])
    np.savez(file, run_parameters, t_record, data_record, hawk_record, cumulative_num_eaten)

    with open(log_file, "a") as f:
        log_str = "omp"
        for parameter in run_parameters:
            log_str = log_str + " " + str(parameter)

        f.write(log_str + "\n")
    
    return 0
