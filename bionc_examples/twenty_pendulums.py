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
from twenty_pendulum.sim import StandardSim
from variational_integrator import VariationalIntegrator, QuadratureRule


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


def twenty_pendulum():
    biomodel = bionc.BiomechanicalModel.load("models/20_link_pendulum.nMod")
    casadi_biomodel = biomodel.to_mx()

    nb_segments = biomodel.nb_segments
    print(f"Number of segments: {nb_segments}")

    time = 1
    dt = 0.05

    results = pd.DataFrame(
        columns=
        ['time', 'time_steps',
         'states','q', 'qdot', "lagrange_multipliers",
         'Etot', 'Ekin', 'Epot',
         'Phi_r', 'Phi_j',
         'Phi_rdot', 'Phi_jdot',
         'Phi_rddot', 'Phi_jddot',])

    sim_rk4 = StandardSim(biomodel, final_time=time, dt=dt, RK="RK4")
    sim_rk45 = StandardSim(biomodel, final_time=time, dt=dt, RK="RK45")

    # sim_rk4.plot_Q()
    # sim_rk45.plot_Q()
    # plt.show()

    sim_rk4.plot_energy()
    sim_rk45.plot_energy()

    all_q_t0 = sim_rk4.results["q"][:biomodel.nb_Q, 0:1]
    # get the q at the second frame for the discrete euler lagrange equation
    all_q_t1 = sim_rk4.results["q"][:biomodel.nb_Q, 1:2]

    results_VI = dict()
    # variational integrator
    vi = VariationalIntegrator(
        biomodel=casadi_biomodel,
        time_step=dt,
        time=time,
        discrete_lagrangian_approximation=QuadratureRule.TRAPEZOIDAL,
        q_init=np.concatenate((all_q_t0[:, np.newaxis], all_q_t1[:, np.newaxis]), axis=1),
    )
    tic0 = t.time()
    q_vi, lambdas_vi = vi.integrate()
    tic_end = t.time()
    results_VI["time"] = tic_end - tic0
    print(f"VI time: {results_VI['time']}")

    results_VI["q"] = q_vi
    results_VI["lagrange_multipliers"] = lambdas_vi


if __name__ == "__main__":
    twenty_pendulum()
