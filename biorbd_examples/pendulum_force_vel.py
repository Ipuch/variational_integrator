"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
This example is a simple pendulum controlled by a constant torque.
In this example an initial state and initial velocity are given to the variational integrator.
The solution is compared with the integration with two initial states.
"""

import biorbd_casadi

from varint.minimal_variational_integrator import VariationalIntegrator

from biorbd_examples.utils import *


def one_pendulum_force(time: float = 10, time_step: float = 0.05, unit_test: bool = False):
    biorbd_casadi_model = biorbd_casadi.Model(Models.PENDULUM.value)
    biorbd_model = biorbd.Model(Models.PENDULUM.value)

    nb_frames = int(time / time_step)

    q_t0 = np.array([1.54])
    q_t1 = np.array([1.545])

    # controls
    torque = 20
    tau = np.zeros((biorbd_model.nbGeneralizedTorque(), nb_frames))
    tau[-1, :] = torque

    # variational integrator
    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        controls=tau,
        q_init=np.concatenate((q_t0[:, np.newaxis], q_t1[:, np.newaxis]), axis=1),
    )
    q_vi, lambdas_vi = vi.integrate()

    vi_vel = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        controls=tau,
        q_init=q_t0[:, np.newaxis],
        q_dot_init=np.array([[0.0]]),
    )
    q_vi_vel, lambdas_vi_vel, _ = vi_vel.integrate()

    if unit_test:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(q_vi_vel[0, 1:], label="Variational Integrator with initial velocity")
        plt.plot(q_vi[0, 1:], label="Variational Integrator with two initial states")
        plt.legend()
        plt.title("Comparison of the two methods")

        # Plot total energy for both methods
        plt.figure()
        plt.plot(discrete_total_energy(biorbd_model, q_vi_vel, time_step).reshape(discrete_total_energy(biorbd_model, q_vi_vel, time_step).shape[0]) - work(tau, q_vi_vel)[:-1], label="Total energy with initial velocity")
        plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step).reshape(discrete_total_energy(biorbd_model, q_vi, time_step).shape[0]) - work(tau, q_vi)[:-1], label="Total energy with two initial states")
        plt.legend()
        plt.title("Total energy for both methods")

        plt.show()

    return q_vi


if __name__ == "__main__":
    one_pendulum_force(unit_test=True)
