import numpy as np
from matplotlib import pyplot as plt


def get_name(program_type):
    """
    Return the string name corresponding to the program_type.
    program_type: string
    """
    match program_type:
        case "omp":
            return "MacBook Air M1 OpenMP"
        case "mpi":
            return "MacBook Air M1 MPI"
        case "ompBC4":
            return "OpenMP"
        case "mpiBC4":
            return "MPI"
        case "DompBC4":
            return "Dynamic scheduling"
        case "SompBC4":
            return "Static scheduling"
        case "RompBC4":
            return "Runtime scheduling"
        case "NmpiBC4":
            return "Method 1"
        case "SmpiBC4":
            return "Method 2"
        case "MmpiBC4":
            return "Method 3"
        case _:
            return "unknown"
        

def colour_gradient(i, max):
    """
    Return a colour tuple that interpolates from blue to orange as i is changed from 0 to max-1.
    i: int
    max: int
    """
    if max == 1:
        return (0,0,1)
    else:
        return (i/(max-1), 0.5 * i/(max-1), 1-i/(max-1))


def get_data(log_file):
    """
    Format the data in log_file into a numpy array.
    Each entry contains: program_type, time, num_boids, num_hawks, threads, grid_size, max_x, may_x.
    log_file: string
    """
    data_log = []
    with open(log_file, "r") as f:
        for line in f:
            
            line_arr = line.split(" ")
            line_arr[-1] = line_arr[-1][:-1] # Remove the "\n" from the last element.

            # Skip lines not in the correct format.
            if not line_arr[0] in ["omp", "mpi", "ompBC4", "mpiBC4", "DompBC4", "NmpiBC4", "SompBC4", "MmpiBC4", "RompBC4", "SmpiBC4"]:
                continue
            
            # Cast contents to the correct types.
            line_arr[1], line_arr[2], line_arr[3], line_arr[4], line_arr[5], line_arr[6], line_arr[7] = float(line_arr[1]), int(float(line_arr[2])), int(float(line_arr[3])), int(float(line_arr[4])), int(float(line_arr[5])), int(float(line_arr[6])), int(float(line_arr[7]))
            data_log.append(line_arr)
    
    return np.array(data_log, dtype=object)


def contains_arr(arr, record):
    """
    Search for arr in record. If found return the index it is at, else return -1.
    Returns an int.
    arr is an array of specific parameters (time, program_type, num_boids, num_hawks, threads, grid_size, max_x, may_x).
    record is an array of arrays of parameters.
    """
    for i in range(len(record)):
        ans = True
        record_arr = record[i]

        for j in range(len(arr)):
            if arr[j] != record_arr[j]:
                ans = False
                break
        
        if ans == True:
            return i
                  
    return -1


def search_data(data, program_type, boids_arr, hawks_arr, threads_arr, grid_size_arr=[8]):
    """
    Searches data for the times and errors of all combinations of parameters specified by program_type, boids_arr, hawks_arr, threads_arr, grid_size_arr.
    Returns two flat arrays of doubles containing the times and errors.
    As the returned arrays are flat, only one of the array arguments (boids_arr, hawks_arr, threads_arr, grid_size_arr) should have a length not equal to 1.
    data is an array of [time, error, program_type, num_boids, num_hawks, threads, grid_size, max_x, may_x]
    program_type: string
    boids_arr: [int]
    hawks_arr: [int]
    threads_arr: [int]
    grid_size_arr: [int], default [8]
    """
    time_result = []
    err_result = []

    for boids in boids_arr:
        for hawks in hawks_arr:
            for threads in threads_arr:
                for grid_size in grid_size_arr:
                    
                    # Find the entry in data and get its time and error.
                    is_found = False
                    for entry in data:
                        if entry[2] == program_type and entry[3] == boids and entry[4] == hawks and entry[5] == threads and entry[6] == grid_size:
                            time_result.append(entry[0])
                            err_result.append(entry[1])
                            is_found = True
                            break
                    
                    if not is_found:
                        print(f"Entry for program_type {program_type}, boids {boids}, hawks {hawks}, threads {threads}, grid_size {grid_size} not found")

    return (time_result, err_result)



# Program start.
log_file = "log_for_graph.txt"
data_log = get_data(log_file)

# Reorder data to: time, program_type, num_boids, num_hawks, threads, grid_size, max_x, may_x.
data_log[:,[0,1]] = data_log[:,[1,0]]

# For each set of unique parameters, average the first three recorded computation times and take their standard deviation.
unique_params = [data_log[0,1:]]
param_times = [[data_log[0,0]]]

for entry in data_log[1:]:

    i = contains_arr(entry[1:], unique_params)

    # If parameters already in unique_params, add the time to the corresponding entry in param_times to average later.
    if i >= 0:
        param_times[i].append(entry[0])
    
    # Else start a new entry.
    else:
        unique_params.append(entry[1:])
        param_times.append([entry[0]])
        
# Now do averages and standared deviation.
param_times = np.array(param_times, dtype=object)
averaged_data = []
for i in range(len(unique_params)):
    if len(param_times[i]) >= 3:
        param_times3 = [param_times[i][0], param_times[i][1], param_times[i][2]]
        avg = np.average(param_times3)
        err = np.std(param_times3, ddof=1)
        averaged_data.append( np.append([avg, err], unique_params[i]) )
    else:
        print(f"Not 3 pieces of data for {unique_params[i]}")

# averaged_data contains entries of [time, error, program_type, num_boids, num_hawks, threads, grid_size, max_x, may_x]
averaged_data = np.array(averaged_data)


### Now plot all the graphs in the report ###

## Scaling with system size.
low_boids_arr = np.arange(1000, 11000, 1000)
boids_arr1 = np.append(low_boids_arr, np.arange(15000, 45000, 5000))
boids_arr4 = np.append(np.append(low_boids_arr, np.arange(15000, 45000, 5000)), np.arange(45000, 70000, 20000))
boids_arr7 = np.append(low_boids_arr, np.arange(25000, 110000, 20000))
boids_arr14 = np.append(low_boids_arr, np.arange(25000, 150000, 20000))
boids_arr28 = np.append(low_boids_arr, np.arange(25000, 250000, 20000))
print("")

for program_type in ["ompBC4", "mpiBC4"]:
    for axis_type in range(2):
        fig, ax = plt.subplots()
        if axis_type == 0:
            ax.set(xlabel="Number of boids", ylabel="Computation time /s", xlim=[500, 260000])
        else:
            ax.set(xlabel="Number of boids", ylabel="Computation time /s", xlim=[500, 260000], xscale="log", yscale="log")

        # 1 thread
        (time_result1, err_result) = search_data(averaged_data, program_type, boids_arr1, [0], [1], [8])
        label = f"{get_name(program_type)} with 1 thread"
        plt.errorbar(boids_arr1, time_result1, yerr = err_result, label=label, capsize=2, color=colour_gradient(0,5))

        # 4 threads
        (time_result, err_result) = search_data(averaged_data, program_type, boids_arr4, [0], [4], [8])
        label = f"{get_name(program_type)} with 4 threads"
        plt.errorbar(boids_arr4, time_result, yerr = err_result, label=label, capsize=2, color=colour_gradient(1,5))

        # 7 threads
        (time_result, err_result) = search_data(averaged_data, program_type, boids_arr7, [0], [7], [8])
        label = f"{get_name(program_type)} with 7 threads"
        plt.errorbar(boids_arr7, time_result, yerr = err_result, label=label, capsize=2, color=colour_gradient(2,5))

        # 14 threads
        (time_result, err_result) = search_data(averaged_data, program_type, boids_arr14, [0], [14], [8])
        label = f"{get_name(program_type)} with 14 threads"
        plt.errorbar(boids_arr14, time_result, yerr = err_result, label=label, capsize=2, color=colour_gradient(3,5))

        # 28 threads
        (time_result, err_result) = search_data(averaged_data, program_type, boids_arr28, [0], [28], [8])
        label = f"{get_name(program_type)} with 28 threads"
        plt.errorbar(boids_arr28, time_result, yerr = err_result, label=label, capsize=2, color=colour_gradient(4,5))

        ax.legend()

    # Find gradient of log lines.
    len1 = len(boids_arr1)
    [gradient,_] = np.polyfit(np.log10(boids_arr1[len1-6:]), np.log10(np.array(time_result1)[len1-6:]), 1)
    print(f"{get_name(program_type)}: Gradient of log graph for 1 thread plotting time against system size is {round(gradient,2)}")

    len28 = len(boids_arr28)
    [gradient,_] = np.polyfit(np.log10(boids_arr28[len28-10:]), np.log10(np.array(time_result)[len28-10:]), 1)
    print(f"{get_name(program_type)}: Gradient of log graph for 28 threads plotting time against system size is {round(gradient,2)}")

print("")

## Scaling with threads.

fig, ax = plt.subplots()
ax.set(xlabel="Number of threads", ylabel="Computation time /s")
threads_arr = np.arange(1, 29, 1)
threads_arr250k = np.array([1,6,12,18,24,28])
threads_arr_mac = np.arange(1, 9, 1)
linestyle = ""
for program_type in ["ompBC4", "mpiBC4"]:
    linestyle += "-"
    (time_result, err_result) = search_data(averaged_data, program_type, [30000], [0], threads_arr)
    label = f"{get_name(program_type)} with 30,000 boids"
    plt.plot(threads_arr, time_result, label=label, linestyle=linestyle, color=colour_gradient(0, 4))

linestyle = ""
for program_type in ["omp", "mpi"]:
    linestyle += "-"
    (time_result, err_result) = search_data(averaged_data, program_type, [30000], [0], threads_arr_mac)
    label = f"{get_name(program_type)} with 30,000 boids"
    plt.plot(threads_arr_mac, time_result, label=label,  linestyle=linestyle, color=colour_gradient(2, 4))    

ax.legend()


fig, ax = plt.subplots()
ax.set(xlabel="Number of threads", ylabel="Computation time /s")
linestyle = ""
for program_type in ["ompBC4", "mpiBC4"]:
    linestyle += "-"

    (time_result, err_result) = search_data(averaged_data, program_type, [100000], [0], threads_arr)
    label = f"{get_name(program_type)} with 100,000 boids"
    plt.plot(threads_arr, time_result, label=label, linestyle=linestyle, color=colour_gradient(1, 4))

    (time_result, err_result) = search_data(averaged_data, program_type, [250000], [0], threads_arr250k)
    label = f"{get_name(program_type)} with 250,000 boids"
    plt.plot(threads_arr250k, time_result, label=label, linestyle=linestyle, color=colour_gradient(3, 4))

ax.legend()


## Speedup

fig, ax = plt.subplots()
ax.set(xlabel="Number of threads", ylabel="Speed-up")
linestyle = ""
for program_type in ["ompBC4", "mpiBC4"]:
    linestyle += "-"

    (time_result, err_result) = search_data(averaged_data, program_type, [30000], [0], threads_arr)
    label = f"{get_name(program_type)} with 30,000 boids"
    plt.plot(threads_arr, time_result[0]/np.array(time_result), label=label, linestyle=linestyle, color=colour_gradient(0, 4))
    print(f"{label}: Max speed-up with {get_name(program_type)} was {round(np.max(time_result[0]/np.array(time_result)), 2)}.")

    (time_result, err_result) = search_data(averaged_data, program_type, [100000], [0], threads_arr)
    label = f"{get_name(program_type)} with 100,000 boids"
    plt.plot(threads_arr, time_result[0]/np.array(time_result), label=label, linestyle=linestyle, color=colour_gradient(1, 4))
    print(f"{label}: Max speed-up with {get_name(program_type)} was {round(np.max(time_result[0]/np.array(time_result)), 2)}.")

    (time_result, err_result) = search_data(averaged_data, program_type, [250000], [0], threads_arr250k)
    label = f"{get_name(program_type)} with 250,000 boids"
    plt.plot(threads_arr250k, time_result[0]/np.array(time_result), label=label, linestyle=linestyle, color=colour_gradient(3, 4))
    print(f"{label}: Max speed-up with {get_name(program_type)} was {round(np.max(time_result[0]/np.array(time_result)), 2)}.")
    
print("")
linestyle = ""
for program_type in ["omp", "mpi"]:
    linestyle += "-"

    (time_result, err_result) = search_data(averaged_data, program_type, [30000], [0], threads_arr_mac)
    label = f"{get_name(program_type)} with 30,000 boids"
    plt.plot(threads_arr_mac, time_result[0]/np.array(time_result), label=label, linestyle=linestyle, color=colour_gradient(2, 4))
    print(f"{label}: Max speed-up with {get_name(program_type)} was {round(np.max(time_result[0]/np.array(time_result)), 2)}.")
 
ax.legend()
print("")


## Efficiency

fig, ax = plt.subplots()
ax.set(xlabel="Number of threads", ylabel="Efficiency")
linestyle = ""
for program_type in ["ompBC4", "mpiBC4"]:
    linestyle += "-"

    (time_result, err_result) = search_data(averaged_data, program_type, [30000], [0], threads_arr)
    label = f"{get_name(program_type)} with 30,000 boids"
    plt.plot(threads_arr, time_result[0]/np.array(time_result)/threads_arr, label=label, linestyle=linestyle, color=colour_gradient(0, 4))

    (time_result, err_result) = search_data(averaged_data, program_type, [100000], [0], threads_arr)
    label = f"{get_name(program_type)} with 100,000 boids"
    plt.plot(threads_arr, time_result[0]/np.array(time_result)/threads_arr, label=label, linestyle=linestyle, color=colour_gradient(1, 4))

    (time_result, err_result) = search_data(averaged_data, program_type, [250000], [0], threads_arr250k)
    label = f"{get_name(program_type)} with 250,000 boids"
    plt.plot(threads_arr250k, time_result[0]/np.array(time_result)/threads_arr250k, label=label, linestyle=linestyle, color=colour_gradient(3, 4))

linestyle = ""
for program_type in ["omp", "mpi"]:
    linestyle += "-"
    (time_result, err_result) = search_data(averaged_data, program_type, [30000], [0], threads_arr_mac)
    label = f"{get_name(program_type)} with 30,000 boids"
    plt.plot(threads_arr_mac, time_result[0]/np.array(time_result)/threads_arr_mac, label=label, linestyle=linestyle, color=colour_gradient(2, 4))

ax.legend()


## Change grid size.

fig, ax = plt.subplots()
ax.set(xlabel="Grid size", ylabel="Computation time /s", xlim=[5,75])
grid_arr = [8, 9, 12, 18, 24, 36, 72]
col_i = 0
for program_type in ["ompBC4", "mpiBC4"]:
    (time_result, err_result) = search_data(averaged_data, program_type, [30000], [0], [4], grid_arr)
    label = f"{get_name(program_type)} with 30,000 boids"
    plt.errorbar(grid_arr, time_result, yerr = err_result, label=label, capsize=2, color=colour_gradient(col_i, 2))
    col_i += 1

ax.legend()


## Change hawks

fig, ax = plt.subplots()
ax.set(xlabel="Number of hawks", ylabel="Computation time /s")
hawk_arr = np.arange(0, 440, 40)
col_i = 0
for program_type in ["ompBC4", "mpiBC4"]:
    (time_result, err_result) = search_data(averaged_data, program_type, [30000], hawk_arr, [4], [8])
    label = f"{get_name(program_type)} with 30,000 boids"
    plt.errorbar(hawk_arr, time_result, yerr = err_result, label=label, capsize=2, color=colour_gradient(col_i, 4))

    col_i += 1
    (time_result, err_result) = search_data(averaged_data, program_type, [100000], hawk_arr, [4], [8])
    label = f"{get_name(program_type)} with 100,000 boids"
    plt.errorbar(hawk_arr, time_result, yerr = err_result, label=label, capsize=2, color=colour_gradient(col_i, 4))
    col_i += 1

ax.legend()

## OpenMP scheduling

fig, ax = plt.subplots()
ax.set(xlabel="Number of boids", ylabel="Computation time /s", xlim=[0,100000])
boids_arr8 = np.arange(5000, 100000, 10000)
boids_arr2 = np.arange(5000, 80000, 10000)

col_i = 0
for program_type in ["SompBC4", "DompBC4", "RompBC4"]:
    (time_result, err_result) = search_data(averaged_data, program_type, boids_arr2, [0], [2])
    label = f"{get_name(program_type)} with 2 threads"
    plt.errorbar(boids_arr2, time_result, yerr = err_result, label=label, capsize=2, linestyle="--", color=colour_gradient(col_i, 3))
    col_i += 1

col_i = 0
for program_type in ["SompBC4", "DompBC4", "RompBC4"]:
    (time_result, err_result) = search_data(averaged_data, program_type, boids_arr8, [0], [8])
    label = f"{get_name(program_type)} with 8 threads"
    plt.errorbar(boids_arr8, time_result, yerr = err_result, label=label, capsize=2, color=colour_gradient(col_i, 3))
    col_i += 1

ax.legend()

## MPI allocation methods

fig, ax = plt.subplots()
ax.set(xlabel="Number of boids", ylabel="Computation time /s", xlim=[0,100000])

col_i = 0
for program_type in ["NmpiBC4", "SmpiBC4", "MmpiBC4"]:
    (time_result, err_result) = search_data(averaged_data, program_type, boids_arr2, [0], [2])
    label = f"{get_name(program_type)} with 2 threads"
    plt.errorbar(boids_arr2, time_result, yerr = err_result, label=label, capsize=2, linestyle="--", color=colour_gradient(col_i, 3))
    col_i += 1

col_i = 0
for program_type in ["NmpiBC4", "SmpiBC4", "MmpiBC4"]:
    (time_result, err_result) = search_data(averaged_data, program_type, boids_arr8, [0], [8])
    label = f"{get_name(program_type)} with 8 threads"
    plt.errorbar(boids_arr8, time_result, yerr = err_result, label=label, capsize=2, color=colour_gradient(col_i, 3))
    col_i += 1

ax.legend()

plt.show()
