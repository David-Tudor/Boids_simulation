from matplotlib import pyplot as plt
import matplotlib.animation
import numpy as np

def plot_animation(t_record, data_record, hawk_record, cumulative_num_eaten, max_x, max_y):
    """
    Plot an animation using the recorded data.
    t_record is an array of times: [double]
    data_record contains an array for each time with an array of all the boid positions and velocities (x,y,vx,vy): [[[double]]]
    hawk_record contains an array for each time with an array of all the hawk positions and velocities (x,y,vx,vy): [[[double]]]
    cumulative_num_eaten records the cumulative number of boids eaten for each time: [int]
    max_x: int
    max_y: int
    """
    # Plot colours.
    boid_col_tup = (0,0,1)
    eaten_col_tup = (0.8,0.8,0.8)
    hawk_col_tup = (1,0,0)

    def update_plot(ti):
        current_data = data_record[ti]
        current_hawk_data = hawk_record[ti]
        current_num_eaten = cumulative_num_eaten[ti]

        # The array slicing with "current_num_eaten" works because when the records were made, the boids were put in the order that they're eaten in.
        boid_plot.set_offsets(np.transpose([current_data[current_num_eaten:,0],current_data[current_num_eaten:,1]]))
        eaten_plot.set_offsets(np.transpose([current_data[:current_num_eaten,0],current_data[:current_num_eaten,1]]))
        hawk_plot.set_offsets(np.transpose([current_hawk_data[:,0],current_hawk_data[:,1]]))

        ax.set_title("Time={:.3f} s".format(t_record[ti]))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set(xlabel="x", ylabel="y", xlim=(0,max_x), ylim=(0,max_y), title=f"Time=0.000 s")

    boid_plot = ax.scatter(data_record[0,:,0], data_record[0,:,1], s=0.1, color=boid_col_tup)
    eaten_plot = ax.scatter([], [], s=0.1, color=eaten_col_tup)
    hawk_plot = ax.scatter(hawk_record[0,:,0], hawk_record[0,:,1], s=0.1, color=hawk_col_tup)

    anim = matplotlib.animation.FuncAnimation(fig, update_plot, len(t_record), interval=60, blit=False)
    anim.save("boids_animation.mp4", fps=20)

    plt.show()