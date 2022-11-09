"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
"""
from enum import Enum
import numpy as np
from casadi import MX, SX, jacobian, Function, rootfinder, transpose, vertcat
import biorbd_casadi
import biorbd


class Models(Enum):
    """
    The different models
    """

    PENDULUM = "pendulum.bioMod"
    DOUBLE_PENDULUM = "double_pendulum.bioMod"
    TWO_PENDULUMS = "two_pendulums.bioMod"
    ONE_PENDULUM = "one_pendulum.bioMod"


class DiscreteLagrangian(Enum):
    """
    The different discrete methods
    """

    MIDPOINT = "midpoint"
    LEFT_APPROXIMATION = "left_approximation"
    RIGHT_APPROXIMATION = "right_approximation"
    TRAPEZOIDAL = "trapezoidal"


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
    constraints: Callable
        The constraints of the system only one callable for now
    jac: Callable
        The jacobian of the constraints of the system only one callable for now
    """

    def __init__(
        self,
        biorbd_model: biorbd_casadi.Model,
        time_step: float,
        time: float,
        constraints: Function = None,
        jac: Function = None,
        discrete_lagrangian_approximation: DiscreteLagrangian = DiscreteLagrangian.MIDPOINT,
    ):

        self.biorbd_model = biorbd_model
        self.time_step = time_step
        self.time = time
        self.constraints = constraints
        self.jac = jac
        self.discrete_lagrangian_approximation = discrete_lagrangian_approximation

        self._declare_mx()
        self._declare_discrete_euler_lagrange_equations()

    def _declare_mx(self):
        """
        Declare the MX variables
        """
        # declare q for each time step of the integration
        self.q1 = MX.sym("q1", self.biorbd_model.nbQ(), 1)  # ti-1
        self.q2 = MX.sym("q2", self.biorbd_model.nbQ(), 1)  # ti
        self.q3 = MX.sym("q3", self.biorbd_model.nbQ(), 1)  # ti+1
        # declare lambda for each constraint
        if self.constraints is not None:
            self.lambdas = MX.sym("lambda", self.constraints.nnz_out(), 1)

    def _declare_discrete_euler_lagrange_equations(self):
        """
        Declare the discrete Euler-Lagrange equations
        """
        self.dela = Function(
            "DEL",
            [self.q1, self.q2, self.q3, self.lambdas] if self.constraints is not None else [self.q1, self.q2, self.q3],
            [self.discrete_euler_lagrange_equations(self.q1, self.q2, self.q3, self.lambdas)]
            if self.constraints is not None
            else [self.discrete_euler_lagrange_equations(self.q1, self.q2, self.q3)],
        ).expand()

    def _declare_residuals(self, q1_num, q2_num):

        if self.constraints is None:
            mx_residuals = self.dela(q1_num, q2_num, self.q3)

            self.residuals = Function(
                "Residuals",
                [vertcat(self.q3)],
                [mx_residuals],
            ).expand()

        else:
            mx_residuals = vertcat(
                self.dela(q1_num, q2_num, self.q3, self.lambdas),
                self.constraints(self.q3),
            )

            self.residuals = Function(
                "Residuals",
                [vertcat(self.q3, self.lambdas)],
                [mx_residuals],
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
        if self.discrete_lagrangian_approximation == DiscreteLagrangian.MIDPOINT:
            q_discrete = (q1 + q2) / 2
            qdot_discrete = (q2 - q1) / self.time_step
            return MX(self.time_step) * self.lagrangian(q_discrete, qdot_discrete)
        elif self.discrete_lagrangian_approximation == DiscreteLagrangian.LEFT_APPROXIMATION:
            q_discrete = q1
            qdot_discrete = (q2 - q1) / self.time_step
            return MX(self.time_step) * self.lagrangian(q_discrete, qdot_discrete)
        elif self.discrete_lagrangian_approximation == DiscreteLagrangian.RIGHT_APPROXIMATION:
            q_discrete = q2
            qdot_discrete = (q2 - q1) / self.time_step
            return MX(self.time_step) * self.lagrangian(q_discrete, qdot_discrete)
        elif self.discrete_lagrangian_approximation == DiscreteLagrangian.TRAPEZOIDAL:
            # from : M. West, “Variational integrators,” Ph.D. dissertation, California Inst.
            # Technol., Pasadena, CA, 2004. p 13
            qdot_discrete = (q2 - q1) / self.time_step
            return MX(self.time_step) / 2 * (self.lagrangian(q1, qdot_discrete) + self.lagrangian(q2, qdot_discrete))
        else:
            raise NotImplementedError(
                f"Discrete Lagrangian {self.discrete_lagrangian_approximation} is not implemented"
            )

    def discrete_euler_lagrange_equations(
        self, q1: MX | SX, q2: MX | SX, q3: MX | SX, lambdas: MX | SX = None
    ) -> MX | SX:
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
        lambdas: MX | SX
            The Lagrange multipliers of second time step

        Returns
        -------
        The discrete Euler-Lagrange equations
        """
        D2_Ld_q1_q2 = transpose(jacobian(self.discrete_lagrangian(q1, q2), q2))
        D1_Ld_q2_q3 = transpose(jacobian(self.discrete_lagrangian(q2, q3), q2))
        if self.constraints is None:
            return D2_Ld_q1_q2 + D1_Ld_q2_q3
        else:
            return D2_Ld_q1_q2 + D1_Ld_q2_q3 - transpose(self.jac(q2)) @ self.lambdas

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

        if self.constraints is not None:
            lambdas_all = np.zeros((self.constraints.nnz_out(), n_frames))
            lambdas_num = lambdas_all[:, 0]
        else:
            lambdas_all = None

        for i in range(2, int(self.time / self.time_step)):

            # f(q1, q2, q3) = 0, only q3 is unknown
            ifcn = self._declare_residuals(q1_num, q2_num)

            # q2 as an initial guess
            if self.constraints is not None:
                v_init = np.concatenate((q2_num, lambdas_num), axis=0)
                v_opt = ifcn(v_init)
                q3_num = v_opt[: self.biorbd_model.nbQ()]
                lambdas_num = v_opt[self.biorbd_model.nbQ() :]
            else:
                q3_num = ifcn(q2_num)

            q1_num = q2_num
            q2_num = q3_num

            if self.constraints is not None:
                lambdas_all[:, i] = lambdas_num.toarray().squeeze()

            q_all[:, i] = q3_num.toarray().squeeze()

        return q_all, lambdas_all


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


def pendulum():
    biorbd_casadi_model = biorbd_casadi.Model(Models.PENDULUM.value)
    biorbd_model = biorbd.Model(Models.PENDULUM.value)

    import time as t

    time = 600
    time_step = 0.01

    multistep_integrator = "DOP853"  # DOP853

    tic0 = t.time()
    # dop853 integrator
    from scipy.integrate import solve_ivp

    q0 = np.array([1.54, 1.54])
    qdot0 = (q0[1] - q0[0]) / time_step
    x0 = np.hstack((q0[0], qdot0))
    fd = lambda t, x: forward_dynamics(biorbd_model, np.array([x[0]]), np.array([x[1]]), np.array([0]))
    q_rk45 = solve_ivp(fd, [0, time], x0, method=multistep_integrator, t_eval=np.arange(0, time, time_step)).y
    from ode_solvers import RK4

    q_rk4 = RK4(np.arange(0, time, time_step), fd, x0)

    tic1 = t.time()

    print(tic1 - tic0)

    # variational integrator
    vi = VariationalIntegrator(biorbd_model=biorbd_casadi_model, time_step=time_step, time=time)
    vi.set_initial_values(q1_num=1.54, q2_num=1.54)
    # vi.set_initial_values(q1_num=q_rk45[0, 0], q2_num=q_rk45[0, 1])
    q_vi, _ = vi.integrate()

    tic2 = t.time()
    print(tic2 - tic1)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(q_vi[0, 1:], label="Variational Integrator", color="red", linestyle="-", marker="", markersize=2)
    plt.plot(q_rk45[0, 0:-1], label=multistep_integrator, color="blue", linestyle="-", marker="", markersize=2)
    plt.plot(q_rk4[0, 0:-1], label="RK4", color="green", linestyle="-", marker="", markersize=2)
    plt.title("Generalized coordinates comparison between RK45 and variational integrator")
    plt.legend()

    # plot total energy for both methods
    plt.figure()
    plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), label="Variational Integrator", color="red")
    plt.plot(total_energy(biorbd_model, q_rk45[0, :], q_rk45[1, :]), label=multistep_integrator, color="blue")
    plt.plot(total_energy(biorbd_model, q_rk4[0, :], q_rk4[1, :]), label="RK4", color="green")
    plt.title("Total energy comparison between RK45 and variational integrator")
    plt.legend()

    plt.show()

    return print("Hello World")


def double_pendulum():
    biorbd_casadi_model = biorbd_casadi.Model(Models.DOUBLE_PENDULUM.value)
    biorbd_model = biorbd.Model(Models.DOUBLE_PENDULUM.value)

    import time as t

    time = 60
    time_step = 0.05
    # multistep_integrator = "RK45"  # DOP853
    multistep_integrator = "DOP853"  # DOP853

    tic0 = t.time()

    from scipy.integrate import solve_ivp

    q0 = np.array([[1.54, 1.545], [1.54, 1.545]])
    qdot0 = (q0[:, 0] - q0[:, 1]) / time_step
    x0 = np.hstack((q0[:, 0], qdot0))
    fd = lambda t, x: forward_dynamics(biorbd_model, x[0:2], x[2:4], np.array([0]))
    q_rk45 = solve_ivp(fd, [0, time], x0, method=multistep_integrator, t_eval=np.arange(0, time, time_step)).y
    from ode_solvers import RK4

    q_rk4 = RK4(np.arange(0, time, time_step), fd, x0)

    tic1 = t.time()

    print(tic1 - tic0)

    # variational integrator
    vi = VariationalIntegrator(biorbd_model=biorbd_casadi_model, time_step=time_step, time=time)
    # vi.set_initial_values(q1_num=1.54, q2_num=1.545)
    vi.set_initial_values(q1_num=q_rk45[:2, 0], q2_num=q_rk45[:2, 1])
    q_vi, _ = vi.integrate()

    tic2 = t.time()
    print(tic2 - tic1)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(
        np.arange(0, time, time_step),
        q_vi[0, :],
        label="Variational Integrator",
        color="red",
        linestyle="-",
        marker="",
        markersize=2,
    )
    axs[0].plot(
        np.arange(0, time, time_step),
        q_rk45[0, :],
        label=multistep_integrator,
        color="blue",
        linestyle="-",
        marker="",
        markersize=2,
    )
    axs[0].plot(
        np.arange(0, time, time_step), q_rk4[0, :], label="RK4", color="green", linestyle="-", marker="", markersize=2
    )
    axs[0].set_title("q1")
    axs[0].legend()
    axs[1].plot(
        np.arange(0, time, time_step),
        q_vi[1, :],
        label="Variational Integrator",
        color="red",
        linestyle="-",
        marker="",
        markersize=2,
    )
    axs[1].plot(
        np.arange(0, time, time_step),
        q_rk45[1, :],
        label=multistep_integrator,
        color="blue",
        linestyle="-",
        marker="",
        markersize=2,
    )
    axs[1].plot(
        np.arange(0, time, time_step), q_rk4[1, :], label="RK4", color="green", linestyle="-", marker="", markersize=2
    )
    axs[1].set_title("q2")
    axs[1].legend()

    # plot total energy for both methods
    plt.figure()
    plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), label="Variational Integrator", color="red")
    plt.plot(total_energy(biorbd_model, q_rk45[0, :], q_rk45[1, :]), label=multistep_integrator, color="blue")
    plt.plot(total_energy(biorbd_model, q_rk4[0, :], q_rk4[1, :]), label="RK4", color="green")
    plt.title("Total energy comparison between RK45 and variational integrator")
    plt.legend()

    plt.show()

    import bioviz

    b = bioviz.Viz(Models.DOUBLE_PENDULUM.value)
    b.load_movement(q_vi)
    b.exec()
    return print("Hello World")


def two_pendulum():
    biorbd_casadi_model = biorbd_casadi.Model(Models.TWO_PENDULUMS.value)
    biorbd_model = biorbd.Model(Models.DOUBLE_PENDULUM.value)

    import time as t

    time = 10
    time_step = 0.05

    tic0 = t.time()

    q_t0 = np.array([1.54, 1.54])
    t_t0 = biorbd_model.globalJCS(q_t0, 1).to_array()[1:3, 3]
    q_t1 = np.array([1.545, 1.545])
    t_t1 = biorbd_model.globalJCS(q_t1, 1).to_array()[1:3, 3]

    all_q_t0 = np.array([q_t0[0], t_t0[0], t_t0[1], q_t0[1]])
    all_q_t1 = np.array([q_t1[0], t_t1[0], t_t1[1], q_t1[1]])

    # build  constraint
    # the origin of the second pendulum is constrained to the tip of the first pendulum
    q_sym = MX.sym("q", (biorbd_casadi_model.nbQ(), 1))
    constraint = (
        biorbd_casadi_model.markers(q_sym)[1].to_mx()[1:] - biorbd_casadi_model.globalJCS(q_sym, 1).to_mx()[1:3, 3]
    )
    fcn_constraint = Function("constraint", [q_sym], [constraint], ["q"], ["constraint"]).expand()
    fcn_jacobian = Function("jacobian", [q_sym], [jacobian(constraint, q_sym)], ["q"], ["jacobian"]).expand()

    # test the constraint
    print(fcn_constraint(all_q_t0))

    # variational integrator
    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        constraints=fcn_constraint,
        jac=fcn_jacobian,
    )
    # vi.set_initial_values(q1_num=1.54, q2_num=1.545)
    vi.set_initial_values(q1_num=all_q_t0, q2_num=all_q_t1)
    q_vi, lambdas_vi = vi.integrate()

    tic2 = t.time()
    print(tic2 - tic0)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(
        np.arange(0, time, time_step), q_vi[0, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[1, 0].plot(
        np.arange(0, time, time_step), q_vi[1, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[2, 0].plot(
        np.arange(0, time, time_step), q_vi[2, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[0, 1].plot(
        np.arange(0, time, time_step), q_vi[3, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[1, 1].plot(
        np.arange(0, time, time_step), lambdas_vi[0, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[2, 1].plot(
        np.arange(0, time, time_step), lambdas_vi[1, :], label="Variational Integrator", color="red", linestyle="-"
    )

    axs[0, 0].set_title("q1")
    axs[1, 0].set_title("q2")
    axs[2, 0].set_title("q3")
    axs[0, 1].set_title("q4")
    axs[1, 1].set_title("lambda1")
    axs[2, 1].set_title("lambda2")
    axs[0, 0].legend()

    # plot total energy for both methods
    plt.figure()
    plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), label="Variational Integrator", color="red")
    plt.title("Total energy comparison between RK45 and variational integrator")
    plt.legend()

    # verify the constraint respect
    plt.figure()
    plt.plot(fcn_constraint(q_vi).toarray()[0, :], label="Variational Integrator", color="red")
    plt.plot(fcn_constraint(q_vi).toarray()[1, :], label="Variational Integrator", color="red")
    plt.title("Constraint respect")

    plt.show()

    import bioviz

    b = bioviz.Viz(Models.TWO_PENDULUMS.value)
    b.load_movement(q_vi)
    b.exec()
    return print("Hello World")


def one_pendulum():
    biorbd_casadi_model = biorbd_casadi.Model(Models.ONE_PENDULUM.value)
    biorbd_model = biorbd.Model(Models.ONE_PENDULUM.value)

    import time as t

    time = 10
    time_step = 0.05

    tic0 = t.time()

    q_t0 = np.array([1.54])
    q_t1 = np.array([1.545])

    all_q_t0 = np.array([0, 0, q_t0[0]])
    all_q_t1 = np.array([0, 0, q_t1[0]])

    # build  constraint
    # the origin of the second pendulum is constrained to the tip of the first pendulum
    q_sym = MX.sym("q", (biorbd_casadi_model.nbQ(), 1))
    constraint = biorbd_casadi_model.markers(q_sym)[0].to_mx()[1:] - MX.zeros((2, 1))
    fcn_constraint = Function("constraint", [q_sym], [constraint], ["q"], ["constraint"]).expand()
    fcn_jacobian = Function("jacobian", [q_sym], [jacobian(constraint, q_sym)], ["q"], ["jacobian"]).expand()

    # test the constraint
    print(fcn_constraint(all_q_t0))

    # variational integrator
    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        constraints=fcn_constraint,
        jac=fcn_jacobian,
        discrete_lagrangian_approximation=DiscreteLagrangian.TRAPEZOIDAL,
    )
    # vi.set_initial_values(q1_num=1.54, q2_num=1.545)
    vi.set_initial_values(q1_num=all_q_t0, q2_num=all_q_t1)
    q_vi, lambdas_vi = vi.integrate()

    tic2 = t.time()
    print(tic2 - tic0)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(
        np.arange(0, time, time_step), q_vi[0, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[1, 0].plot(
        np.arange(0, time, time_step), q_vi[1, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[2, 0].plot(
        np.arange(0, time, time_step), q_vi[2, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[0, 1].plot(
        np.arange(0, time, time_step), lambdas_vi[0, :], label="Variational Integrator", color="red", linestyle="-"
    )
    axs[1, 1].plot(
        np.arange(0, time, time_step), lambdas_vi[1, :], label="Variational Integrator", color="red", linestyle="-"
    )

    axs[0, 0].set_title("q1")
    axs[1, 0].set_title("q2")
    axs[2, 0].set_title("q3")
    axs[0, 1].set_title("lambda1")
    axs[1, 1].set_title("lambda2")
    axs[0, 0].legend()

    # plot total energy for both methods
    plt.figure()
    plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), label="Variational Integrator", color="red")
    plt.title("Total energy comparison between RK45 and variational integrator")
    plt.legend()

    # verify the constraint respect
    plt.figure()
    plt.plot(fcn_constraint(q_vi).toarray()[0, :], label="Variational Integrator", color="red")
    plt.plot(fcn_constraint(q_vi).toarray()[1, :], label="Variational Integrator", color="red")
    plt.title("Constraint respect")

    plt.show()

    import bioviz

    b = bioviz.Viz(Models.ONE_PENDULUM.value)
    b.load_movement(q_vi)
    b.exec()
    return print("Hello World")


if __name__ == "__main__":
    # pendulum()
    # double_pendulum()
    # two_pendulum()
    one_pendulum()
