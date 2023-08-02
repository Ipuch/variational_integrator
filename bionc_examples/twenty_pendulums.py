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

    time = 0.4
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

    sim_rk4 = StandardSim(biomodel, final_time=time, dt=dt, RK="RK4")
    sim_rk45 = StandardSim(biomodel, final_time=time, dt=dt, RK="RK45")

    # viz = bionc.vizualization.animations.Viz(
    #     biomodel,
    #     background_color=(1, 1, 1),
    #     show_natural_mesh=True,
    #     window_size=(1800, 900),
    #     camera_position=(41.22295145870256, 0.07359041918618726, -9.019234652667775),
    #     camera_focus_point=(
    #         0.6036061985552914, -0.227817999898422, -9.198309631280022),
    #     camera_zoom=0.09498020579891492,
    #     camera_roll=-90.0,
    # )
    # q = sim_rk45.results["q"][: biomodel.nb_Q, :]
    # viz.animate(bionc.NaturalCoordinates(q),
    #             None,
    #             frame_rate=1 / dt,
    #             )

    # sim_rk4.plot_Q()
    # sim_rk45.plot_Q()
    # plt.show()

    # sim_rk4.plot_energy()
    # sim_rk45.plot_energy()
    # plt.show()
    # #
    # print(f"Energy at t=0: {sim_rk4.results['Etot'][0]}")
    # print(f"Energy at t=1: {sim_rk4.results['Etot'][-1]}")
    #
    print(f"Rigidbody Constraint at t=0: {sim_rk4.results['Phi_r'][:, 0]}")
    print(f"Rigidbody Constraint at t=end, min: {sim_rk4.results['Phi_r'][:, -1].min()}")
    print(f"Rigidbody Constraint at t=end, max: {sim_rk4.results['Phi_r'][:, -1].max()}")
    print(f"Rigidbody Constraint at t=end, median: {np.median(sim_rk4.results['Phi_r'][:, -1])}")

    print(f"Rigidbody Constraint at t=0: {sim_rk4.results['Phi_r'][:, 0]}")
    print(f"Absolute Rigidbody Constraint at t=end, min: {np.abs(sim_rk4.results['Phi_r'][:, -1]).min()}")
    print(f"Absolute Rigidbody Constraint at t=end, max: {np.abs(sim_rk4.results['Phi_r'][:, -1]).max()}")
    print(f"Absolute Rigidbody Constraint at t=end, median: {np.median(np.abs(sim_rk4.results['Phi_r'][:, -1]))}")

    print(f"Joint Constraint at t=0: {sim_rk4.results['Phi_j'][:, 0]}")
    print(f"Joint Constraint at t=end, min: {sim_rk4.results['Phi_j'][:, -1].min()}")
    print(f"Joint Constraint at t=end, max: {sim_rk4.results['Phi_j'][:, -1].max()}")
    print(f"Joint Constraint at t=end, median: {np.median(sim_rk4.results['Phi_j'][:, -1])}")

    # RK45
    print(f"Absolute Joint Constraint at t=0: {sim_rk45.results['Phi_j'][:, 0]}")
    print(f"Absolute Joint Constraint at t=end, min: {np.abs(sim_rk45.results['Phi_j'][:, -1]).min()}")
    print(f"Absolute Joint Constraint at t=end, max: {np.abs(sim_rk45.results['Phi_j'][:, -1]).max()}")
    print(f"Absolute Joint Constraint at t=end, median: {np.median(np.abs(sim_rk45.results['Phi_j'][:, -1]))}")

    print(f"Rigidbody Constraint at t=0: {sim_rk45.results['Phi_r'][:, 0]}")
    print(f"Rigidbody Constraint at t=end, min: {sim_rk45.results['Phi_r'][:, -1].min()}")
    print(f"Rigidbody Constraint at t=end, max: {sim_rk45.results['Phi_r'][:, -1].max()}")
    print(f"Rigidbody Constraint at t=end, median: {np.median(sim_rk45.results['Phi_r'][:, -1])}")

    print(f"Rigidbody Constraint at t=0: {sim_rk45.results['Phi_r'][:, 0]}")
    print(f"Absolute Rigidbody Constraint at t=end, min: {np.abs(sim_rk45.results['Phi_r'][:, -1]).min()}")
    print(f"Absolute Rigidbody Constraint at t=end, max: {np.abs(sim_rk45.results['Phi_r'][:, -1]).max()}")
    print(f"Absolute Rigidbody Constraint at t=end, median: {np.abs(np.median(sim_rk45.results['Phi_r'][:, -1]))}")

    print(f"Joint Constraint at t=0: {sim_rk45.results['Phi_j'][:, 0]}")
    print(f"Joint Constraint at t=end, min: {sim_rk45.results['Phi_j'][:, -1].min()}")
    print(f"Joint Constraint at t=end, max: {sim_rk45.results['Phi_j'][:, -1].max()}")
    print(f"Joint Constraint at t=end, median: {np.median(sim_rk45.results['Phi_j'][:, -1])}")

    print(f"Absolute Joint Constraint at t=0: {sim_rk45.results['Phi_j'][:, 0]}")
    print(f"Absolute Joint Constraint at t=end, min: {np.abs(sim_rk45.results['Phi_j'][:, -1]).min()}")
    print(f"Absolute Joint Constraint at t=end, max: {np.abs(sim_rk45.results['Phi_j'][:, -1]).max()}")
    print(f"Absolute Joint Constraint at t=end, median: {np.abs(np.median(sim_rk45.results['Phi_j'][:, -1]))}")

    # all_q_t0 = sim_rk45.results["q"][: biomodel.nb_Q, 0:1]
    # # get the q at the second frame for the discrete euler lagrange equation
    # all_q_t1 = sim_rk45.results["q"][: biomodel.nb_Q, 1:2]

    # vi_sim = VariationalSim(casadi_biomodel, final_time=1.2, dt=dt, all_q_t0=all_q_t0, all_q_t1=all_q_t1)
    tic = t.time()
    vi_sim = VariationalSim(casadi_biomodel, final_time=time, dt=dt)
    toc = t.time() - tic
    print(f"VI: {toc}")

    # viz = bionc.vizualization.animations.Viz(
    #     biomodel,
    #     background_color=(1, 1, 1),
    #     show_natural_mesh=True,
    #     window_size=(1800, 900),
    #     camera_position=(41.22295145870256, 0.07359041918618726, -9.019234652667775),
    #     camera_focus_point=(
    #         0.6036061985552914, -0.227817999898422, -9.198309631280022),
    #     camera_zoom=0.09498020579891492,
    #     camera_roll=-90.0,
    # )
    # q = vi_sim.results["q"][: biomodel.nb_Q, :]
    # viz.animate(bionc.NaturalCoordinates(q),
    #             None,
    #             frame_rate=1 / dt,
    #             )

    # vi_sim.plot_energy()
    # print(f"Energy at t=0: {vi_sim.results['Etot'][0]}")
    # print(f"Energy at t=end: {vi_sim.results['Etot'][-1]}")
    #
    # # print(f"Rigidbody Constraint at t=0: {vi_sim.results['Phi_r'][:, 0]}")
    # print(f"Rigidbody Constraint at t=end, min: {vi_sim.results['Phi_r'][:, -1].min()}")
    # print(f"Rigidbody Constraint at t=end, max: {vi_sim.results['Phi_r'][:, -1].max()}")
    # # print(f"Rigidbody Constraint at t=end, median: {vi_sim.results['Phi_r'][:, -1].median()}")
    #
    # # print(f"Joint Constraint at t=0: {vi_sim.results['Phi_j'][:, 0]}")
    # print(f"Joint Constraint at t=end, min: {vi_sim.results['Phi_j'][:, -1].min()}")
    # print(f"Joint Constraint at t=end, max: {vi_sim.results['Phi_j'][:, -1].max()}")
    # # print(f"Joint Constraint at t=end, median: {vi_sim.results['Phi_j'][:, -1].median()}")

    # display all the constraints in one graph
    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(np.max(np.abs(sim_rk4.results["Phi_j"][:, :]), axis=0), color="blue")
    ax2.plot(np.max(np.abs(sim_rk4.results["Phi_r"][:, :]), axis=0), color="blue", label="RK4")

    ax1.plot(np.max(np.abs(sim_rk45.results["Phi_j"][:, :]), axis=0), color="red")
    ax2.plot(np.max(np.abs(sim_rk45.results["Phi_r"][:, :]), axis=0), color="red", label="RK45")

    ax1.plot(np.max(np.abs(vi_sim.results["Phi_j"][:, :]), axis=0), color="green")
    ax2.plot(np.max(np.abs(vi_sim.results["Phi_r"][:, :]), axis=0), color="green", label="Variational Integrator")

    ax1.set_yscale("log")
    ax2.set_yscale("log")

    ax1.set_title("Joint constraints")
    ax2.set_title("Rigid-body constraints")

    ax1.set_ylim([1e-22, 1])
    ax2.set_ylim([1e-27, 1])

    plt.legend()
    plt.show()

    fig, ax1 = plt.subplots(1)

    ax1.plot(sim_rk4.results["E_tot"][:, :], color="blue", label="RK4")
    ax1.plot(sim_rk45.results["E_tot"][:, :], color="red", label="RK45")
    ax1.plot(vi_sim.results["E_tot"][:, :], color="green", label="Variational Integrator")

    ax1.set_yscale("log")

    ax1.set_ylim([1e-22, 1e3])

    plt.legend()
    plt.show()


if __name__ == "__main__":
    twenty_pendulum()
