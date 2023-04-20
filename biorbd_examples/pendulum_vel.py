"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
This example is a simple pendulum and compares the integrations obtained by 3 different integrators.
In this example an initial state and initial velocity are given to the variational integrator.
The solution is compared with the integration with two initial states.
"""
import biorbd_casadi

from varint.minimal_variational_integrator import VariationalIntegrator

from biorbd_examples.utils import *


def pendulum(time: float = 10, time_step: float = 0.01, unit_test: bool = False):
    biorbd_casadi_model = biorbd_casadi.Model(Models.PENDULUM.value)
    biorbd_model = biorbd.Model(Models.PENDULUM.value)

    import time as t

    multistep_integrator = "DOP853"  # DOP853

    tic0 = t.time()
    # dop853 integrator
    from scipy.integrate import solve_ivp

    q0 = np.array([1.54])
    qdot0 = 0.0
    x0 = np.hstack((q0[0], qdot0))
    fd = lambda t, x: forward_dynamics(biorbd_model, np.array([x[0]]), np.array([x[1]]), np.array([0]))
    q_rk45 = solve_ivp(fd, [0, time], x0, method=multistep_integrator, t_eval=np.arange(0, time, time_step)).y
    from ode_solvers import RK4

    q_rk4 = RK4(np.arange(0, time, time_step), fd, x0)

    # variational integrator
    vi_vel = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        q_init=np.array([[1.54]]),
        q_dot_init=np.array([[0.0]]),
    )
    # vi.set_initial_values(q_prev=q_rk45[0, 0], q_cur=q_rk45[0, 1])
    q_vi_vel, *_ = vi_vel.integrate()

    # variational integrator
    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        q_init=np.array([[1.54, 1.54]]),
    )
    # vi.set_initial_values(q_prev=q_rk45[0, 0], q_cur=q_rk45[0, 1])
    q_vi, _, qdot_final = vi.integrate()

    print(f"Final velocity with variational integrator (initial velocity): {qdot_final}, "
          f"with variational integrator (two initial states): {vi.compute_final_velocity(q_vi[:, -2], q_vi[:, -1])}, "
          f"with {multistep_integrator}: {q_rk45[1, -1]}, "
          f"with RK4: {q_rk4[1, -1]}")

    if unit_test:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(q_vi_vel[0, 1:], label="Variational Integrator with initial velocity")
        plt.plot(q_vi[0, 1:], label="Variational Integrator with two initial states")
        plt.plot(q_rk45[0, 0:-1], label=multistep_integrator)
        plt.plot(q_rk4[0, 0:-1], label="RK4")
        plt.title(f"Generalized coordinates comparison between RK45, {multistep_integrator} and variational integrator")
        plt.legend()

        # plot total energy for both methods
        plt.figure()
        plt.plot(discrete_total_energy(biorbd_model, q_vi_vel, time_step), label="Variational Integrator with initial velocity")
        plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), label="Variational Integrator with two initial states")
        plt.plot(total_energy(biorbd_model, q_rk45[0, :], q_rk45[1, :]), label=multistep_integrator)
        plt.plot(total_energy(biorbd_model, q_rk4[0, :], q_rk4[1, :]), label="RK4")
        plt.title(f"Total energy comparison between RK45, {multistep_integrator} and variational integrator")
        plt.legend()

        plt.show()

    return q_vi_vel


if __name__ == "__main__":
    pendulum(unit_test=True)
