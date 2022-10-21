import scipy.integrate
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import math

debug_interp = False

def lorenz():
    def f(t, x):    # returns the first derivative of the Lorenz system at time t, coordinates x, with the parameters defined in the function
        r = 126.52
        s = 10.0
        b = 8.0/3.0
        return np.array([s*(x[1] - x[0]), x[0]*(r - x[2]) - x[1], x[0]*x[1] - b*x[2]])

    x0 = np.array([-7.69, -15.61, 102.0])   # initial condition
    t0 = 0.0                                # initial time
    dur = 4.5                               # duration

    # integrate the equations over the time period
    sol = scipy.integrate.solve_ivp(f, [0.0, dur], x0, method='RK45', dense_output=True)

    N = 10000                                           # number of temporal steps to subdivide interval
    dt = dur/N                                          # time step
    t_init = 42*dt
    t_term = dur
    t = np.linspace(t_init, t_term, N, endpoint=True)   # break the [0,dur] interval into a N steps
    x = sol.sol(t).T                                    # calculate the coordinates at each time step
    ch = np.transpose(x)                                # transpose, putting the three coordinates into their own channels ch[0..2]

    # form interpolated section
    N_overlap = 3
    delta_x_int = x[0] - x[-1]                    # vector from last data point to first data point
    max_abs_delta_x_int = max(abs(delta_x_int[0]), max(abs(delta_x_int[1]), abs(delta_x_int[2])))
    delta_x_ext = x[N_overlap] - x[-1-N_overlap]   # vector extending over the whole interpolation region
    max_abs_delta_x_ext = max(abs(delta_x_ext[0]), max(abs(delta_x_ext[1]), abs(delta_x_ext[2])))
    if max_abs_delta_x_int == 0 or max_abs_delta_x_ext <= max_abs_delta_x_int:
        print("Poor configuration for interpolation.  Exiting.")
        return
    N_int = int(math.ceil(2*float(N_overlap)*max_abs_delta_x_int/(max_abs_delta_x_ext-max_abs_delta_x_int)))
    N_ext = 2*N_overlap + N_int
    t_ext = np.linspace(0.0, dt*N_ext, N_ext, endpoint=True)
    t_int = t_ext[N_overlap:-N_overlap]
    t_overlap = np.append(t_ext[:N_overlap+1], t_ext[-1-N_overlap:])
    ch_overlap0 = np.append(ch[0][-1-N_overlap:], ch[0][:N_overlap+1])
    ch_overlap2 = np.append(ch[2][-1-N_overlap:], ch[2][:N_overlap+1])
    f0_ext = interp1d(t_overlap, ch_overlap0, kind='cubic')
    f2_ext = interp1d(t_overlap, ch_overlap2, kind='cubic')

    # append interpolated data in to data channels
    ch0 = np.append(ch[0], f0_ext(t_int[1:]))
    ch2 = np.append(ch[2], f2_ext(t_int[1:]))

    plt.box(False)
    if not debug_interp:
        plt.plot(ch0, ch2, linewidth=4.0, color='black')
    else:
        plt.plot(ch[0], ch[2], linewidth=4.0, color='black')
        plt.plot(f0_ext(t_int), f2_ext(t_int), linewidth=4.0, color='red')
    plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.savefig("lorenz.svg")
    plt.show()

if __name__ == "__main__":
    lorenz()
