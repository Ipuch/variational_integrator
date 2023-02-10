import numpy as np
import biorbd

from enum import Enum


class Models(Enum):
    """
    The different models
    """

    PENDULUM = "models/pendulum.bioMod"
    ONE_PENDULUM = "models/one_pendulum.bioMod"
    DOUBLE_PENDULUM = "models/double_pendulum.bioMod"
    TWO_PENDULUMS = "models/two_pendulums.bioMod"
    TRIPLE_PENDULUM = "models/triple_pendulum.bioMod"
    THREE_PENDULUMS = "models/three_pendulums.bioMod"


def forward_dynamics(biorbd_model: biorbd.Model, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    Forward dynamics of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    qdot: np.ndarray
        The generalized velocities
    tau: np.ndarray
        The generalized torques

    Returns
    -------
    The generalized accelerations
    """

    return np.concatenate((qdot, biorbd_model.ForwardDynamics(q, qdot, tau).to_array()))


def total_energy_i(biorbd_model: biorbd.Model, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    """
    Compute the total energy of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    qdot: np.ndarray
        The generalized velocities

    Returns
    -------
    The total energy
    """

    return biorbd_model.KineticEnergy(q, qdot) + biorbd_model.PotentialEnergy(q)


def total_energy(biorbd_model: biorbd.Model, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
    """
    Compute the total energy of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    qdot: np.ndarray
        The generalized velocities

    Returns
    -------
    The total energy
    """
    H = np.zeros((q.shape[0]))
    for i in range(q.shape[0]):
        H[i] = total_energy_i(biorbd_model, q[i : i + 1], qdot[i : i + 1])

    return H


def discrete_total_energy_i(biorbd_model: biorbd.Model, q1: np.ndarray, q2: np.ndarray, time_step) -> np.ndarray:
    """
    Compute the discrete total energy of a biorbd model

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q1: np.ndarray
        The generalized coordinates at the first time step
    q2: np.ndarray
        The generalized coordinates at the second time step
    time_step: float
        The time step

    Returns
    -------
    The discrete total energy
    """
    q_middle = (q1 + q2) / 2
    qdot_middle = (q2 - q1) / time_step
    return total_energy_i(biorbd_model, np.array(q_middle), np.array(qdot_middle))


def discrete_total_energy(biorbd_model: biorbd.Model, q: np.ndarray, time_step) -> np.ndarray:
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


def energy_calculation(biorbd_model, q, n, time_step):
    """
    This function can only be used for the examples two_pendulums and three_pendulums

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The generalized coordinates
    n: number of pendulums
    time_step: float
        The time step
    """
    g = 9.81

    q_coord_rel = [q[0, :]]

    for i in range(1, n):
        q_coord_rel.append(q[3 * i, :] - q[3 * (i - 1), :])

    q_coord_rel = np.asarray(q_coord_rel)

    Ec = []
    Ep = []

    for Seg in range(n):
        CoM_marker = Seg * 3 + 2
        # Rotational kinetic energy
        I_G = biorbd_model.segments()[0].characteristics().inertia().to_array()
        q_coord_rel_dot = (q_coord_rel[Seg, 1:] - q_coord_rel[Seg, :-1]) / time_step
        Ec_rot = 1 / 2 * I_G[0, 0] * q_coord_rel_dot**2
        # Translational kinetic energy
        y_com = np.asarray(
            [biorbd_model.markers(q_coord_rel[:, i])[CoM_marker].to_array()[1] for i in range(len(q_coord_rel[0, :]))]
        )
        z_com = np.asarray(
            [biorbd_model.markers(q_coord_rel[:, i])[CoM_marker].to_array()[2] for i in range(len(q_coord_rel[0, :]))]
        )
        vy_com = (y_com[1:] - y_com[:-1]) / time_step
        vz_com = (z_com[1:] - z_com[:-1]) / time_step
        V_com_sq = vy_com**2 + vz_com**2
        Ec_trs = 1 / 2 * biorbd_model.segments()[0].characteristics().mass() * V_com_sq

        Ec.append(Ec_trs + Ec_rot)

        # Potential energy
        Ep.append(biorbd_model.segments()[Seg].characteristics().mass() * g * z_com)

    return np.sum(np.asarray(Ep)[:, :-1], axis=0) + np.sum(Ec, axis=0)
