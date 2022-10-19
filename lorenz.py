import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

def lorenz():
    def f(t, x):
        r = 126.52
        s = 10.0
        b = 8.0/3.0
        return np.array([s*(x[1] - x[0]), x[0]*(r - x[2]) - x[1], x[0]*x[1] - b*x[2]])

    x0 = np.array([-7.69, -15.61, 102.0])
    t0 = 0.0
    dur = 8.505

    sol = scipy.integrate.solve_ivp(f, [0.0, dur], x0, method='RK45', dense_output=True)

    t = np.linspace(0.0, dur, 10000)
    x = sol.sol(t)

    ch = np.transpose(x.T[307:])
    plt.box(False)
    plt.plot(ch[0], ch[2], linewidth=3.0, color='black')
    plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.savefig("lorenz.svg")
    plt.show()

if __name__ == "__main__":
    lorenz()
