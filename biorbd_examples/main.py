"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
"""
from enum import Enum
import numpy as np
from casadi import MX, SX, jacobian, Function, rootfinder, transpose, vertcat
import biorbd_casadi
import biorbd

from variational_integrator import VariationalIntegrator, QuadratureRule


class Models(Enum):
    """
    The different models
    """

    PENDULUM = "models/pendulum.bioMod"
    DOUBLE_PENDULUM = "models/double_pendulum.bioMod"
    TWO_PENDULUMS = "models/two_pendulums.bioMod"
    ONE_PENDULUM = "models/one_pendulum.bioMod"


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
    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        q_init=np.array([[1.54, 1.54]]),
    )
    # vi.set_initial_values(q_prev=q_rk45[0, 0], q_cur=q_rk45[0, 1])
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
    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        q_init=np.concatenate((q_rk45[:2, 0:1], q_rk45[:2, 1:2]), axis=1),
    )
    # vi.set_initial_values(q_prev=1.54, q_cur=1.545)
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
    axs[0].set_title("q_prev")
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
    axs[1].set_title("q_cur")
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
        q_init=np.concatenate((all_q_t0[:, np.newaxis], all_q_t1[:, np.newaxis]), axis=1),
    )
    # vi.set_initial_values(q_prev=1.54, q_cur=1.545)
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

    axs[0, 0].set_title("q_prev")
    axs[1, 0].set_title("q_cur")
    axs[2, 0].set_title("q_next")
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
        discrete_lagrangian_approximation=QuadratureRule.TRAPEZOIDAL,
        q_init=np.concatenate((all_q_t0[:, np.newaxis], all_q_t1[:, np.newaxis]), axis=1),
    )
    # vi.set_initial_values(q_prev=1.54, q_cur=1.545)
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

    axs[0, 0].set_title("q_prev")
    axs[1, 0].set_title("q_cur")
    axs[2, 0].set_title("q_next")
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


def one_pendulum_force():
    biorbd_casadi_model = biorbd_casadi.Model(Models.ONE_PENDULUM.value)
    biorbd_model = biorbd.Model(Models.ONE_PENDULUM.value)

    import time as t

    time = 10
    time_step = 0.05
    nb_frames = int(time / time_step)

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

    # controls
    tau = np.zeros((biorbd_model.nbGeneralizedTorque(), nb_frames))
    tau[-1, :] = +0.25

    # variational integrator
    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        constraints=fcn_constraint,
        jac=fcn_jacobian,
        discrete_lagrangian_approximation=QuadratureRule.TRAPEZOIDAL,
        controls=tau,
        q_init=np.concatenate((all_q_t0[:, np.newaxis], all_q_t1[:, np.newaxis]), axis=1),
    )
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

    axs[0, 0].set_title("q_prev")
    axs[1, 0].set_title("q_cur")
    axs[2, 0].set_title("q_next")
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
    # one_pendulum()
    one_pendulum_force()
