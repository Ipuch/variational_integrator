"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
This example is a double pendulum and compares the integrations obtained by 3 different integrators.
"""
import biorbd_casadi

from varint.minimal_variational_integrator import VariationalIntegrator, QuadratureRule

from utils import *


def double_pendulum(time: float = 60, time_step: float = 0.05, unit_test: bool = False):
    biorbd_casadi_model = biorbd_casadi.Model(Models.DOUBLE_PENDULUM.value)
    biorbd_model = biorbd.Model(Models.DOUBLE_PENDULUM.value)

    import time as t

    # multistep_integrator = "RK45"  # DOP853
    multistep_integrator = "DOP853"  # DOP853

    tic0 = t.time()

    from scipy.integrate import solve_ivp

    q0 = np.array([[1.54, 1.545], [1.54, 1.545]])
    qdot0 = (q0[:, 0] - q0[:, 1]) / time_step
    x0 = np.hstack((q0[:, 0], qdot0))
    fd = lambda t, x: forward_dynamics(biorbd_model, x[0:2], x[2:4], np.array([0]))
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
        q_init=np.concatenate((q_rk45[:2, 0:1], q_rk45[:2, 1:2]), axis=1),
        discrete_approximation=QuadratureRule.MIDPOINT,
    )
    # vi.set_initial_values(q_prev=1.54, q_cur=1.545)
    q_vi, _ = vi.integrate()

    tic2 = t.time()
    print(tic2 - tic1)

    if unit_test:
        import bioviz

        b = bioviz.Viz(Models.DOUBLE_PENDULUM.value)
        b.load_movement(q_vi)
        b.exec()

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(
            np.arange(0, time, time_step),
            q_vi[0, :],
            label="Variational Integrator",
            color="red",
            linestyle="-",
            marker="",
            markersize=2,
        )
        axs[0].plot(
            np.arange(0, time, time_step),
            q_rk45[0, :],
            label=multistep_integrator,
            color="blue",
            linestyle="-",
            marker="",
            markersize=2,
        )
        axs[0].plot(
            np.arange(0, time, time_step), q_rk4[0, :], label="RK4", color="green", linestyle="-", marker="", markersize=2
        )
        axs[0].set_title("q0")
        axs[0].legend()
        axs[1].plot(
            np.arange(0, time, time_step),
            q_vi[1, :],
            label="Variational Integrator",
            color="red",
            linestyle="-",
            marker="",
            markersize=2,
        )
        axs[1].plot(
            np.arange(0, time, time_step),
            q_rk45[1, :],
            label=multistep_integrator,
            color="blue",
            linestyle="-",
            marker="",
            markersize=2,
        )
        axs[1].plot(
            np.arange(0, time, time_step), q_rk4[1, :], label="RK4", color="green", linestyle="-", marker="", markersize=2
        )
        axs[1].set_title("q1")
        axs[1].legend()

        # plot total energy for both methods
        plt.figure()
        plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), label="Variational Integrator", color="red")
        plt.plot(total_energy(biorbd_model, q_rk45[0, :], q_rk45[1, :]), label=multistep_integrator, color="blue")
        plt.plot(total_energy(biorbd_model, q_rk4[0, :], q_rk4[1, :]), label="RK4", color="green")
        plt.title(f"Total energy comparison between RK45, {multistep_integrator} and variational integrator")
        plt.legend()

        plt.show()

    return q_vi


if __name__ == "__main__":
    double_pendulum(unit_test=True)
