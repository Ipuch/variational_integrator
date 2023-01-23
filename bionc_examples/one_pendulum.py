"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
"""
import numpy as np

import bionc
from bionc.bionc_numpy import SegmentNaturalCoordinates, NaturalCoordinates

from varint.natural_variational_integrator import VariationalIntegrator, QuadratureRule
from models.enums import Models


# def forward_dynamics(biorbd_model: biorbd.Model, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray) -> np.ndarray:
#     """
#     Forward dynamics of a biorbd model
#
#     Parameters
#     ----------
#     biorbd_model: biorbd.Model
#         The biorbd model
#     q: np.ndarray
#         The generalized coordinates
#     qdot: np.ndarray
#         The generalized velocities
#     tau: np.ndarray
#         The generalized torques
#
#     Returns
#     -------
#     The generalized accelerations
#     """
#
#     return np.concatenate((qdot, biorbd_model.ForwardDynamics(q, qdot, tau).to_array()))
#
#
#
#
# def total_energy(biorbd_model: biorbd.Model, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
#     """
#     Compute the total energy of a biorbd model
#
#     Parameters
#     ----------
#     biorbd_model: biorbd.Model
#         The biorbd model
#     q: np.ndarray
#         The generalized coordinates
#     qdot: np.ndarray
#         The generalized velocities
#
#     Returns
#     -------
#     The total energy
#     """
#     H = np.zeros((q.shape[0]))
#     for i in range(q.shape[0]):
#         H[i] = total_energy_i(biorbd_model, q[i : i + 1], qdot[i : i + 1])
#
#     return H
#
#
# def discrete_total_energy_i(biorbd_model: biorbd.Model, q1: np.ndarray, q2: np.ndarray, time_step) -> np.ndarray:
#     """
#     Compute the discrete total energy of a biorbd model
#
#     Parameters
#     ----------
#     biorbd_model: biorbd.Model
#         The biorbd model
#     q1: np.ndarray
#         The generalized coordinates at the first time step
#     q2: np.ndarray
#         The generalized coordinates at the second time step
#     time_step: float
#         The time step
#
#     Returns
#     -------
#     The discrete total energy
#     """
#     q_middle = (q1 + q2) / 2
#     qdot_middle = (q2 - q1) / time_step
#     return total_energy_i(biorbd_model, np.array(q_middle), np.array(qdot_middle))
#
#
def discrete_total_energy(biomodel: bionc.BiomechanicalModel, q: np.ndarray, time_step) -> np.ndarray:
    """
    Compute the discrete total energy of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    time_step: float
        The time step

    Returns
    -------
    The discrete total energy
    """
    n_frames = q.shape[1]
    discrete_total_energy = np.zeros((n_frames - 1, 1))
    for i in range(n_frames - 1):
        discrete_total_energy[i] = discrete_total_energy_i(biorbd_model, q[:, i], q[:, i + 1], time_step)
    return discrete_total_energy


def one_pendulum():
    biomodel = bionc.BiomechanicalModel.load(Models.ONE_PENDULUM.value)
    casadi_biomodel = biomodel.to_mx()

    import time as t

    time = 10
    time_step = 0.05

    tic0 = t.time()

    Qi = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])
    Q = NaturalCoordinates(Qi)
    all_q_t0 = Q

    Qi = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -0.99, 0.14106735979665883], w=[0, 0, 1])
    Q = NaturalCoordinates(Qi)
    all_q_t1 = Q

    # variational integrator
    vi = VariationalIntegrator(
        biomodel=casadi_biomodel,
        time_step=time_step,
        time=time,
        discrete_lagrangian_approximation=QuadratureRule.TRAPEZOIDAL,
        q_init=np.concatenate((all_q_t0[:, np.newaxis], all_q_t1[:, np.newaxis]), axis=1),
    )
    # vi.set_initial_values(q_prev=1.54, q_cur=1.545)
    q_vi, lambdas_vi = vi.integrate()

    tic2 = t.time()
    print(tic2 - tic0)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 2)
    axs[0, 0].plot(
        np.arange(0, time, time_step), q_vi[0, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[0, 0].plot(
        np.arange(0, time, time_step), q_vi[1, :], label="Variational Integrator", color="green", linestyle="-"
    )
    axs[0, 0].plot(
        np.arange(0, time, time_step), q_vi[2, :], label="Variational Integrator", color="blue", linestyle="-"
    )
    axs[1, 0].plot(
        np.arange(0, time, time_step), q_vi[3, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[1, 0].plot(
        np.arange(0, time, time_step), q_vi[4, :], label="Variational Integrator", color="green", linestyle="-"
    )
    axs[1, 0].plot(
        np.arange(0, time, time_step), q_vi[5, :], label="Variational Integrator", color="blue", linestyle="-"
    )
    axs[2, 0].plot(
        np.arange(0, time, time_step), q_vi[6, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[2, 0].plot(
        np.arange(0, time, time_step), q_vi[7, :], label="Variational Integrator", color="green", linestyle="-"
    )
    axs[2, 0].plot(
        np.arange(0, time, time_step), q_vi[8, :], label="Variational Integrator", color="blue", linestyle="-"
    )
    axs[3, 0].plot(
        np.arange(0, time, time_step), q_vi[9, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[3, 0].plot(
        np.arange(0, time, time_step), q_vi[10, :], label="Variational Integrator", color="green", linestyle="-"
    )
    axs[3, 0].plot(
        np.arange(0, time, time_step), q_vi[11, :], label="Variational Integrator", color="blue", linestyle="-"
    )

    for i in range(biomodel.nb_rigid_body_constraints):
        axs[0, 1].plot(
            np.arange(0, time, time_step), lambdas_vi[i, :], label="Variational Integrator", color="red", linestyle="-"
        )
    for i in range(biomodel.nb_rigid_body_constraints, biomodel.nb_rigid_body_constraints + biomodel.nb_joint_constraints):
        axs[1, 1].plot(
            np.arange(0, time, time_step), lambdas_vi[i, :], label="Variational Integrator", color="green", linestyle="-"
        )

    axs[0, 0].set_title("u")
    axs[1, 0].set_title("rp")
    axs[2, 0].set_title("rd")
    axs[3, 0].set_title("w")
    axs[0, 1].set_title("rigid body constraints")
    axs[1, 1].set_title("joint constraints")
    axs[0, 0].legend()

    plt.show()

    # plot total energy for both methods
    plt.figure()
    plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), label="Variational Integrator", color="red")
    plt.title("Total energy comparison between RK45 and variational integrator")
    plt.legend()
    #
    # # verify the constraint respect
    # plt.figure()
    # plt.plot(fcn_constraint(q_vi).toarray()[0, :], label="Variational Integrator", color="red")
    # plt.plot(fcn_constraint(q_vi).toarray()[1, :], label="Variational Integrator", color="red")
    # plt.title("Constraint respect")

    # plt.show()

    # import bioviz
    #
    # b = bioviz.Viz(Models.ONE_PENDULUM.value)
    # b.load_movement(q_vi)
    # b.exec()
    # return print("Hello World")


if __name__ == "__main__":
    one_pendulum()
