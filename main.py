"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
"""

import typing
import numpy as np
from casadi import MX, SX, jacobian, Function, rootfinder
import biorbd_casadi
import biorbd


class VariationalIntegrator:
    """
    This class to build a variational integrator based on the discrete Lagrangian and a first order quadrature method.

    Attributes
    ----------
    biorbd_model: biorbd_casadi.Model
        The biorbd model
    time_step: float
        The time step of the integration
    time: float
        The time of the integration
    """
    def __init__(self,
                 biorbd_model: biorbd_casadi.Model,
                 time_step: float,
                 time: float,
                 ):

        self.biorbd_model = biorbd_model
        self.time_step = time_step
        self.time = time

        self._declare_mx()
        self._declare_discrete_euler_lagrange_equations()

    def _declare_mx(self):
        """
        Declare the MX variables
        """
        self.q1 = MX.sym("q1", self.biorbd_model.nbQ(), 1)
        self.q2 = MX.sym("q2", self.biorbd_model.nbQ(), 1)
        self.q3 = MX.sym("q3", self.biorbd_model.nbQ(), 1)

    def _declare_discrete_euler_lagrange_equations(self):
        """
        Declare the discrete Euler-Lagrange equations
        """
        self.dela = Function(
            "DEL",
            [self.q1, self.q2, self.q3],
            [self.discrete_euler_lagrange_equations(self.q1, self.q2, self.q3)],
        ).expand()

    def _declare_residuals(self, q1_num, q2_num):

        self.residuals = Function(
            "Residuals",
            [self.q3],
            [self.dela(q1_num, q2_num, self.q3)],
        ).expand()

        # Create a implicit function instance to solve the system of equations
        opts = {}
        opts["abstol"] = 1e-14
        ifcn = rootfinder("ifcn", "newton", self.residuals, opts)

        return ifcn

    def lagrangian(self, q: MX | SX, qdot: MX | SX) -> MX | SX:
        """
        Compute the Lagrangian of a biorbd model

        Parameters
        ----------
        q: MX | SX
            The generalized coordinates
        qdot: MX | SX
            The generalized velocities

        Returns
        -------
        The Lagrangian
        """

        return self.biorbd_model.CalcKineticEnergy(q, qdot).to_mx() - self.biorbd_model.CalcPotentialEnergy(q).to_mx()

    def discrete_lagrangian(self, q1: MX | SX, q2: MX | SX) -> MX | SX:
        """
        Compute the discrete Lagrangian of a biorbd model

        Parameters
        ----------
        q1: MX | SX
            The generalized coordinates at the first time step
        q2: MX | SX
            The generalized coordinates at the second time step

        Returns
        -------
        The discrete Lagrangian
        """
        q_middle = (q1 + q2) / 2
        qdot_middle = (q2 - q1) / self.time_step
        return self.time_step * self.lagrangian(q_middle, qdot_middle)

    def discrete_euler_lagrange_equations(self, q1: MX | SX, q2: MX | SX, q3: MX | SX) -> MX | SX:
        """
        Compute the discrete Euler-Lagrange equations of a biorbd model

        Parameters
        ----------
        q1: MX | SX
            The generalized coordinates at the first time step
        q2: MX | SX
            The generalized coordinates at the second time step
        q3: MX | SX
            The generalized coordinates at the third time step

        Returns
        -------
        The discrete Euler-Lagrange equations
        """
        D2_Ld_q1_q2 = jacobian(self.discrete_lagrangian(q1, q2), q2)
        D1_Ld_q2_q3 = jacobian(self.discrete_lagrangian(q2, q3), q2)
        return D2_Ld_q1_q2 + D1_Ld_q2_q3

    def set_initial_values(self, q1_num, q2_num):
        """
        Set the initial values of the variational integrator

        Parameters
        ----------
        q1_num: np.array
            The generalized coordinates at the first time step
        q2_num: np.array
            The generalized coordinates at the second time step
        """
        self.q1_num = q1_num
        self.q2_num = q2_num

    def integrate(self):
        """
        Integrate the discrete euler lagrange over time
        """
        q1_num = self.q1_num
        q2_num = self.q2_num
        n_frames = int(self.time / self.time_step)

        q_all = np.zeros((self.biorbd_model.nbQ(), n_frames))
        q_all[:, 0] = q1_num
        q_all[:, 1] = q2_num

        for i in range(2, int(self.time / self.time_step)):

            # f(q1, q2, q3) = 0, only q3 is unknown
            ifcn = self._declare_residuals(q1_num, q2_num)

            # q2 as an initial guess
            q3_num = ifcn(q2_num)

            q1_num = q2_num
            q2_num = q3_num

            q_all[:, i] = q3_num

        return q_all


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

    return biorbd_model.CalcKineticEnergy(q, qdot) + biorbd_model.CalcPotentialEnergy(q)


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
        H[i] = total_energy_i(biorbd_model, q[i:i+1], qdot[i:i+1])

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


def main():
    biorbd_casadi_model = biorbd_casadi.Model("pendulum.bioMod")
    biorbd_model = biorbd.Model("pendulum.bioMod")

    time = 200
    time_step = 0.01

    # dop853 integrator
    from scipy.integrate import solve_ivp
    q0 = np.array([1.54, 1.545])
    qdot0 = (q0[1] - q0[0]) / time_step
    x0 = np.hstack((q0[0], qdot0))
    fd = lambda t, x: forward_dynamics(biorbd_model, np.array([x[0]]), np.array([x[1]]), np.array([0]))
    q_rk45 = solve_ivp(fd, [0, time], x0, method='RK45', t_eval=np.arange(0, time, time_step)).y

    # variational integrator
    vi = VariationalIntegrator(biorbd_model=biorbd_casadi_model, time_step=time_step, time=time)
    # vi.set_initial_values(q1_num=1.54, q2_num=1.545)
    vi.set_initial_values(q1_num=q_rk45[0, 0], q2_num=q_rk45[0, 1])
    q_vi = vi.integrate()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(q_vi[0, :], label="Variational Integrator", color="red", linestyle="", marker="o", markersize=2)
    plt.plot(q_rk45[0, :], label="RK45", color="blue", linestyle="", marker="o", markersize=2)
    plt.title("Generalized coordinates comparison between RK45 and variational integrator")
    plt.legend()

    # plot total energy for both methods
    plt.figure()
    plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), label="Variational Integrator", color="red")
    plt.plot(total_energy(biorbd_model, q_rk45[0, :], q_rk45[1, :]), label="RK45", color="blue")
    plt.title("Total energy comparison between RK45 and variational integrator")
    plt.legend()

    plt.show()

    return print("Hello World")


if __name__ == "__main__":
    main()
