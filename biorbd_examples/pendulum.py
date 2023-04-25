"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
This example is a simple pendulum and compares the integrations obtained by 3 different integrators.
"""
import biorbd_casadi

from varint.minimal_variational_integrator import VariationalIntegrator

from biorbd_examples.utils import *


def pendulum(time: float = 600, time_step: float = 0.01, unit_test: bool = False):
    biorbd_casadi_model = biorbd_casadi.Model(Models.PENDULUM.value)
    biorbd_model = biorbd.Model(Models.PENDULUM.value)

    import time as t

    multistep_integrator = "DOP853"

    tic0 = t.time()
    # dop853 integrator
    from scipy.integrate import solve_ivp

    q0 = 1.54
    qdot0 = 0.0

    x0 = np.hstack((q0, qdot0))
    fd = lambda t, x: forward_dynamics(biorbd_model, np.array([x[0]]), np.array([x[1]]), np.array([0]))
    q_rk45 = solve_ivp(fd, [0, time], x0, method=multistep_integrator, t_eval=np.arange(0, time, time_step)).y
    from ode_solvers import RK4

    q_rk4 = RK4(np.arange(0, time, time_step), fd, x0)

    tic1 = t.time()

    print(tic1 - tic0)

    # variational integrator
    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        q_init=np.array([[q0]]),
        q_dot_init=np.array([[0.0]]),
    )
    # vi.set_initial_values(q_prev=q_rk45[0, 0], q_cur=q_rk45[0, 1])
    q_vi, _, q_vi_dot = vi.integrate()

    tic2 = t.time()
    print(tic2 - tic1)

    if unit_test:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(q_vi[0, 1:], label="Variational Integrator", color="red", linestyle="-", marker="", markersize=2)
        plt.plot(q_rk45[0, 0:-1], label=multistep_integrator, color="blue", linestyle="-", marker="", markersize=2)
        plt.plot(q_rk4[0, 0:-1], label="RK4", color="green", linestyle="-", marker="", markersize=2)
        plt.title(f"Generalized coordinates comparison between RK45, {multistep_integrator} and variational integrator")
        plt.legend()

        # plot total energy for both methods
        plt.figure()
        plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), label="Variational Integrator", color="red")
        plt.plot(total_energy(biorbd_model, q_rk45[0, :], q_rk45[1, :]), label=multistep_integrator, color="blue")
        plt.plot(total_energy(biorbd_model, q_rk4[0, :], q_rk4[1, :]), label="RK4", color="green")
        plt.title(f"Total energy comparison between RK45, {multistep_integrator} and variational integrator")
        plt.legend()

        plt.show()

        np.set_printoptions(formatter={"float": lambda x: "{0:0.15f}".format(x)})
        print(q_vi[:, -1], q_vi_dot)

    return q_vi, q_vi_dot


if __name__ == "__main__":
    pendulum(unit_test=True)
