"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
"""
import numpy as np
import bionc
import matplotlib.pyplot as plt
import pandas as pd
import time as t

# from ..models.enums import Models
from twenty_pendulum.sim import StandardSim, VariationalSim


# def forward_dynamics(biorbd_numpy_model: biorbd.Model, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray) -> np.ndarray:
#     """
#     Forward dynamics of a biorbd model
#
#     Parameters
#     ----------
#     biorbd_numpy_model: biorbd.Model
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
#     return np.concatenate((qdot, biorbd_numpy_model.ForwardDynamics(q, qdot, tau).to_array()))
#
#
#
#
# def total_energy(biorbd_numpy_model: biorbd.Model, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
#     """
#     Compute the total energy of a biorbd model
#
#     Parameters
#     ----------
#     biorbd_numpy_model: biorbd.Model
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
#         H[i] = total_energy_i(biorbd_numpy_model, q[i : i + 1], qdot[i : i + 1])
#
#     return H
#
#
# def discrete_total_energy_i(biorbd_numpy_model: biorbd.Model, q1: np.ndarray, q2: np.ndarray, time_step) -> np.ndarray:
#     """
#     Compute the discrete total energy of a biorbd model
#
#     Parameters
#     ----------
#     biorbd_numpy_model: biorbd.Model
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
#     return total_energy_i(biorbd_numpy_model, np.array(q_middle), np.array(qdot_middle))
#
#
def discrete_total_energy(biomodel: bionc.BiomechanicalModel, q: np.ndarray, time_step) -> np.ndarray:
    """
    Compute the discrete total energy of a biorbd model

    Parameters
    ----------
    biorbd_numpy_model: biorbd.Model
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


def twenty_pendulum():
    biomodel = bionc.BiomechanicalModel.load("models/20_link_pendulum.nMod")
    casadi_biomodel = biomodel.to_mx()

    nb_segments = biomodel.nb_segments
    print(f"Number of segments: {nb_segments}")

    time = 15
    dt = 0.04

    results = pd.DataFrame(
        columns=[
            "time",
            "time_steps",
            "states",
            "q",
            "qdot",
            "lagrange_multipliers",
            "Etot",
            "Ekin",
            "Epot",
            "Phi_r",
            "Phi_j",
            "Phi_rdot",
            "Phi_jdot",
            "Phi_rddot",
            "Phi_jddot",
        ]
    )

    # sim_rk4 = StandardSim(biomodel, final_time=time, dt=dt, RK="RK4")
    # sim_rk45 = StandardSim(biomodel, final_time=1.2, dt=dt, RK="RK45")

    # sim_rk4.plot_Q()
    # sim_rk45.plot_Q()
    # plt.show()

    # sim_rk4.plot_energy()
    # sim_rk45.plot_energy()
    # plt.show()
    #
    # print(f"Energy at t=0: {sim_rk4.results['Etot'][0]}")
    # print(f"Energy at t=1: {sim_rk4.results['Etot'][-1]}")
    #
    # # print(f"Rigidbody Constraint at t=0: {sim_rk4.results['Phi_r'][:, 0]}")
    # print(f"Rigidbody Constraint at t=end, min: {sim_rk4.results['Phi_r'][:, -1].min()}")
    # print(f"Rigidbody Constraint at t=end, max: {sim_rk4.results['Phi_r'][:, -1].max()}")
    # # print(f"Rigidbody Constraint at t=end, median: {sim_rk4.results['Phi_r'][:, -1].median()}")
    #
    # # print(f"Joint Constraint at t=0: {sim_rk4.results['Phi_j'][:, 0]}")
    # print(f"Joint Constraint at t=end, min: {sim_rk4.results['Phi_j'][:, -1].min()}")
    # print(f"Joint Constraint at t=end, max: {sim_rk4.results['Phi_j'][:, -1].max()}")
    # # print(f"Joint Constraint at t=end, median: {sim_rk4.results['Phi_j'][:, -1].median()}")
    #
    # all_q_t0 = sim_rk45.results["q"][: biomodel.nb_Q, 0:1]
    # # get the q at the second frame for the discrete euler lagrange equation
    # all_q_t1 = sim_rk45.results["q"][: biomodel.nb_Q, 1:2]

    # vi_sim = VariationalSim(casadi_biomodel, final_time=1.2, dt=dt, all_q_t0=all_q_t0, all_q_t1=all_q_t1)
    vi_sim = VariationalSim(casadi_biomodel, final_time=15, dt=dt)
    viz = bionc.vizualization.animations.Viz(biomodel, background_color=(1, 1, 1))
    q = vi_sim.results["q"][: biomodel.nb_Q, :]
    viz.animate(bionc.NaturalCoordinates(q), None, frame_rate=1 / dt)
    vi_sim.plot_energy()
    print(f"Energy at t=0: {vi_sim.results['Etot'][0]}")
    print(f"Energy at t=end: {vi_sim.results['Etot'][-1]}")

    # print(f"Rigidbody Constraint at t=0: {vi_sim.results['Phi_r'][:, 0]}")
    print(f"Rigidbody Constraint at t=end, min: {vi_sim.results['Phi_r'][:, -1].min()}")
    print(f"Rigidbody Constraint at t=end, max: {vi_sim.results['Phi_r'][:, -1].max()}")
    # print(f"Rigidbody Constraint at t=end, median: {vi_sim.results['Phi_r'][:, -1].median()}")

    # print(f"Joint Constraint at t=0: {vi_sim.results['Phi_j'][:, 0]}")
    print(f"Joint Constraint at t=end, min: {vi_sim.results['Phi_j'][:, -1].min()}")
    print(f"Joint Constraint at t=end, max: {vi_sim.results['Phi_j'][:, -1].max()}")
    # print(f"Joint Constraint at t=end, median: {vi_sim.results['Phi_j'][:, -1].median()}")

    vi_sim.plot_Q()


if __name__ == "__main__":
    twenty_pendulum()
