import numpy as np
from pathlib import Path
import copy
import biorbd
from varint import QuadratureRule

from enum import Enum

UNIT_DATA = Path(__file__).parent


class Models(Enum):
    """
    The different models
    """

    PENDULUM = str(UNIT_DATA / "models/pendulum.bioMod")
    ONE_PENDULUM = str(UNIT_DATA / "models/one_pendulum.bioMod")
    DOUBLE_PENDULUM = str(UNIT_DATA / "models/double_pendulum.bioMod")
    TWO_PENDULUMS = str(UNIT_DATA / "models/two_pendulums.bioMod")
    TRIPLE_PENDULUM = str(UNIT_DATA / "models/triple_pendulum.bioMod")
    THREE_PENDULUMS = str(UNIT_DATA / "models/three_pendulums.bioMod")


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
        H[i] = total_energy_i(biorbd_model, q[i: i + 1], qdot[i: i + 1])

    return H


def discrete_total_energy_i(biorbd_model: biorbd.Model, q1: np.ndarray, q2: np.ndarray, time_step, discrete_approximation) -> np.ndarray:
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
    discrete_approximation: QuadratureRule
        The chosen discrete approximation for the energy computing, must be chosen equal to the approximation chosen
        for the integration.

    Returns
    -------
    The discrete total energy
    """
    if discrete_approximation == QuadratureRule.MIDPOINT:
        q = (q1 + q2) / 2
    elif discrete_approximation == QuadratureRule.LEFT_APPROXIMATION:
        q = q1
    elif discrete_approximation == QuadratureRule.RIGHT_APPROXIMATION:
        q = q2
    elif discrete_approximation == QuadratureRule.TRAPEZOIDAL:
        q = (q1 + q2) / 2
    else:
        raise NotImplementedError(
            f"Discrete energy computation {discrete_approximation} is not implemented"
        )
    q = (q1 + q2) / 2
    qdot = (q2 - q1) / time_step
    return total_energy_i(biorbd_model, np.array(q), np.array(qdot))


def discrete_total_energy(
    biorbd_model: biorbd.Model,
    q: np.ndarray,
    time_step: float,
    discrete_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
) -> np.ndarray:
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
    discrete_approximation: QuadratureRule
        The chosen discrete approximation for the energy computing, must be chosen equal to the approximation chosen
        for the integration, trapezoidal by default.

    Returns
    -------
    The discrete total energy
    """
    n_frames = q.shape[1]
    discrete_total_energy = np.zeros((n_frames - 1, 1))
    for i in range(n_frames - 1):
        discrete_total_energy[i] = discrete_total_energy_i(biorbd_model, q[:, i], q[:, i + 1], time_step, discrete_approximation)
    return discrete_total_energy


def energy_calculation(biorbd_model, q, n, time_step):
    """
    This function can only be used for the examples two_pendulums and three_pendulums, it was used to understand how
    RBDL calculates. However, it is not really useful now.

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
        I_G = biorbd_model.segments()[Seg].characteristics().inertia().to_array()
        q_coord_rel_dot = (q_coord_rel[Seg, 1:] - q_coord_rel[Seg, :-1]) / time_step
        Ec_rot = 1 / 2 * I_G[0, 0] * q_coord_rel_dot ** 2
        # Translational kinetic energy
        y_com = np.asarray(
            [biorbd_model.markers(q[:, i])[CoM_marker].to_array()[1] for i in range(len(q[0, :]))]
        )
        z_com = np.asarray(
            [biorbd_model.markers(q[:, i])[CoM_marker].to_array()[2] for i in range(len(q[0, :]))]
        )
        vy_com = (y_com[1:] - y_com[:-1]) / time_step
        vz_com = (z_com[1:] - z_com[:-1]) / time_step
        V_com_sq = vy_com ** 2 + vz_com ** 2
        Ec_trs = 1 / 2 * biorbd_model.segments()[Seg].characteristics().mass() * V_com_sq

        Ec.append(Ec_trs + Ec_rot)

        # Potential energy
        Ep.append(biorbd_model.segments()[Seg].characteristics().mass() * g * z_com)

    return np.sum(np.asarray(Ep)[:, :-1], axis=0) + np.sum(Ec, axis=0)


def work(controls, q):
    """
    Compute the discrete total energy of a biorbd model
    /!\ Only works with constant work (test_one_pendulum_force.py)

    Parameters
    ----------
    controls: np.ndarray
        The controls
    q: np.ndarray
        The generalized coordinates

    Returns
    -------
    The total work produced by all the controls
    """
    delta = copy.deepcopy(q)
    for i in range(delta.shape[0]):
        delta[i, :] -= q[i, 0]
    return np.sum(controls * delta, axis=0)
