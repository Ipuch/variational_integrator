"""
This script compares the absolute energy error between the different ode_solvers at different jump heights.
The comparison is done between the pickle files in the same directory. The "output" of this script is two plots ()
"""
import pickle
import numpy as np

import biorbd_casadi

from varint.minimal_variational_integrator import VariationalIntegrator

from biorbd_examples.utils import *


def get_created_data_from_pickle(file: str):
    """
    Creates data from pickle

    Parameters
    ----------
    file: str
        File where data has been saved.

    Returns
    -------
    Data
    """
    with open(file, "rb") as f:
        data_tmp = pickle.load(f)

        print(f"{file}: status: {data_tmp['status']}, cost:{data_tmp['cost']}, time to optimize: "
              f"{data_tmp['real_time_to_optimize']}, "
              f"nb_it: {data_tmp['iterations']}")
        print(
            f"1ère phase : {data_tmp['time'][0][-1] - data_tmp['time'][0][0]}, "
            f"{data_tmp['states_no_intermediate'][0]['q'].shape[1]} nodes")
        print(
            f"2ère phase : {data_tmp['time'][1][-1] - data_tmp['time'][1][0]}, "
            f"{data_tmp['states_no_intermediate'][1]['q'].shape[1]} nodes")

        shape_0_1 = data_tmp["states_no_intermediate"][0]["q"].shape[1] - 1

        datas_shape = (
            data_tmp["states_no_intermediate"][0]["q"].shape[0],
            shape_0_1 + data_tmp["states_no_intermediate"][1]["q"].shape[1]
        )

        # q
        datas_q = np.zeros(datas_shape)
        datas_q[:, :shape_0_1] = data_tmp["states_no_intermediate"][0]["q"][:, :-1]
        datas_q[:, shape_0_1:] = data_tmp["states_no_intermediate"][1]["q"]

        # qdot
        datas_qdot = np.zeros(datas_shape)
        datas_qdot[:, :shape_0_1] = data_tmp["states_no_intermediate"][0]["qdot"][:, :-1]
        datas_qdot[:, shape_0_1:] = data_tmp["states_no_intermediate"][1]["qdot"]

        # Time
        datas_time = np.zeros(datas_shape[1])
        datas_time[:shape_0_1] = data_tmp["time"][0][:-1]
        datas_time[shape_0_1:] = data_tmp["time"][1][:]

        tau_shape = (
            data_tmp["controls"][0]["tau"].shape[0],
            data_tmp["controls"][0]["tau"].shape[1] - 1 + data_tmp["controls"][1]["tau"].shape[1])
        datas_tau = np.zeros((tau_shape[0], tau_shape[1]))
        datas_tau[:, :data_tmp["controls"][0]["tau"].shape[1] - 1] = data_tmp["controls"][0]["tau"][:, :-1]
        datas_tau[:, data_tmp["controls"][0]["tau"].shape[1] - 1:] = data_tmp["controls"][1]["tau"]

        return np.asarray(datas_q), np.asarray(datas_qdot), np.asarray(datas_time), np.asarray(datas_tau)


def work_f_dx(
        tau1: np.ndarray,
        q1: np.ndarray,
):
    """
    Calculates the work produced by the athlete during the jump.

    Parameters
    ----------
    tau1: np.ndarray
        The generalized controls.
    q1: np.ndarray
        The generalized coordinates.

    Returns
    -------
    work: np.ndarray
        The work produced since the timestep 0.
    """
    dq = np.zeros(q1[6:, :].shape)
    dq[:, 1:] = q1[6:, 1:] - q1[6:, :-1]
    dw = np.zeros(dq.shape)
    dw[:, 1:] = tau1[:, :-1] * dq[:, 1:]
    W = np.zeros(dw.shape)
    for i in range(dw.shape[1] - 1):
        W[:, i+1] = W[:, i] + dw[:, i+1]

    return W.sum(axis=0)


def discrete_mechanical_energy(
        biorbd_model: biorbd.Model,
        q1: np.ndarray,
        qdot1: np.ndarray,
) -> np.ndarray:
    """
    Computes the discrete mechanical energy (kinetic energy + potential gravity energy) of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q1: np.ndarray
        The generalized coordinates
    qdot1: np.ndarray
        The generalized velocities

    Returns
    -------
    The discrete total energy
    """
    n_frames = q1.shape[1]
    d_total_energy = np.zeros(n_frames)
    for i in range(n_frames):
        d_total_energy[i] = biorbd_model.KineticEnergy(q1[:, i], qdot1[:, i]) + biorbd_model.PotentialEnergy(q1[:, i])
    return d_total_energy


if __name__ == "__main__":
    biorbd_casadi_model = biorbd_casadi.Model(Models.ACROBAT.value)
    q, qdot, time, tau = get_created_data_from_pickle(
        "/home/mickaelbegon/Documents/Stage_Amandine/energetic_somersault/examples/20m_RK4_0317"
    )

    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time[1] - time[0],
        time=time[130],
        q_init=q[:, 0][:, np.newaxis],
        q_dot_init=qdot[:, 0][:, np.newaxis],
        controls=np.concatenate((np.zeros((6, 130)), tau[:, :130]), axis=0),
    )

    q_vi, *_ = vi.integrate()

    import bioviz

    b = bioviz.Viz(Models.ACROBAT.value)
    b.load_movement(q_vi)
    b.exec()
