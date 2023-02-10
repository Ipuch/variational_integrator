"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
"""
import biorbd_casadi

from casadi import MX, jacobian, Function

from varint.minimal_variational_integrator import VariationalIntegrator, QuadratureRule

from utils import *


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
    torque = 20
    tau = np.zeros((biorbd_model.nbGeneralizedTorque(), nb_frames))
    tau[-1, :] = torque

    # variational integrator
    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        constraints=fcn_constraint,
        jac=fcn_jacobian,
        discrete_approximation=QuadratureRule.TRAPEZOIDAL,
        controls=tau,
        q_init=np.concatenate((all_q_t0[:, np.newaxis], all_q_t1[:, np.newaxis]), axis=1),
    )
    q_vi, lambdas_vi = vi.integrate()

    tic2 = t.time()
    print(tic2 - tic0)

    import bioviz

    b = bioviz.Viz(Models.ONE_PENDULUM.value)
    b.load_movement(q_vi)
    b.exec()

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

    axs[0, 0].set_title("q0")
    axs[1, 0].set_title("q1")
    axs[2, 0].set_title("q2")
    axs[0, 1].set_title("lambda1")
    axs[1, 1].set_title("lambda2")

    # plot total energy for both methods
    W = torque * (q_vi[2, :] - q_vi[2, 0])

    CoM_marker = 2
    # Rotational kinetic energy
    I_G = biorbd_model.segments()[0].characteristics().inertia().to_array()
    q_coord_rel_dot = (q_vi[2, 1:] - q_vi[2, :-1]) / time_step
    Ec_rot = 1 / 2 * I_G[0, 0] * q_coord_rel_dot**2
    # Translational kinetic energy
    y_com = np.asarray([biorbd_model.markers(q_vi[:, i])[CoM_marker].to_array()[1] for i in range(len(q_vi[0, :]))])
    z_com = np.asarray([biorbd_model.markers(q_vi[:, i])[CoM_marker].to_array()[2] for i in range(len(q_vi[0, :]))])
    V_com_sq = (z_com[:-1] * q_coord_rel_dot) ** 2 + (y_com[:-1] * q_coord_rel_dot) ** 2
    Ec_trs = 1 / 2 * biorbd_model.segments()[0].characteristics().mass() * V_com_sq

    Ec = Ec_trs + Ec_rot

    # Potential energy
    Ep = biorbd_model.segments()[0].characteristics().mass() * 9.81 * z_com

    plt.figure()
    plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step).reshape(discrete_total_energy(biorbd_model, q_vi, time_step).shape[0]) - W[:-1], label="Total energy")
    # plt.plot(Ec, label="Amandine Ec")
    # plt.plot(Ep, label="Amandine Ep")
    plt.plot(Ec + Ep[:-1], label="Amandine Em")
    plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), label="RBDL")
    plt.legend()
    plt.title("Total energy")

    # verify the constraint respect
    plt.figure()
    plt.plot(fcn_constraint(q_vi).toarray()[0, :], label="constraint0")
    plt.plot(fcn_constraint(q_vi).toarray()[1, :], label="constraint1")
    plt.legend()
    plt.title("Constraint respect")

    plt.show()

    return print("Hello World")


if __name__ == "__main__":
    one_pendulum_force()
