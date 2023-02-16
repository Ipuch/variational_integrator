"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
This example is a triple pendulum but the second and third ones have no parent, they are constrained to be coincidents
with the first and the second ones.
"""
import biorbd_casadi

from casadi import MX, jacobian, Function, vertcat

from varint.minimal_variational_integrator import VariationalIntegrator

from biorbd_examples.utils import *


def three_pendulums(time: float = 20, time_step: float = 0.05, unit_test = False):
    biorbd_casadi_model = biorbd_casadi.Model(Models.THREE_PENDULUMS.value)
    biorbd_model = biorbd.Model(Models.THREE_PENDULUMS.value)

    import time as t

    tic0 = t.time()

    # Rotations at t0
    q_t0 = np.array([1.54, 1.54, 1.54])
    # Translations between Seg0 and Seg1 at t0, calculated with cos and sin as Seg1 has no parent
    t1_t0 = np.array([np.sin(q_t0[0]), -np.cos(q_t0[0])])
    # Translations between Seg1 and Seg2 at t0, calculated with cos and sin as Seg2 has no parent
    t2_t0 = np.array([2 * np.sin(q_t0[1]), - 2 * np.cos(q_t0[1])])
    # Rotations at t1
    q_t1 = np.array([1.545, 1.545, 1.545])
    # Translations between Seg0 and Seg1 at t1, calculated with cos and sin as Seg1 has no parent
    t1_t1 = np.array([np.sin(q_t1[0]), -np.cos(q_t1[0])])
    # Translations between Seg1 and Seg2 at t1, calculated with cos and sin as Seg2 has no parent
    t2_t1 = np.array([2 * np.sin(q_t1[1]), - 2 * np.cos(q_t1[1])])

    all_q_t0 = np.array([q_t0[0], t1_t0[0], t1_t0[1], q_t0[1], t2_t0[0], t2_t0[1], q_t0[2]])
    all_q_t1 = np.array([q_t1[0], t1_t1[0], t1_t1[1], q_t1[1], t2_t1[0], t2_t1[1], q_t1[2]])

    # Build  constraints
    q_sym = MX.sym("q", (biorbd_casadi_model.nbQ(), 1))
    # The origin of the second pendulum is constrained to the tip of the first pendulum
    constraint1 = (
        biorbd_casadi_model.markers(q_sym)[1].to_mx()[1:] - biorbd_casadi_model.globalJCS(q_sym, 1).to_mx()[1:3, 3]
    )
    # The origin of the third pendulum is constrained to the tip of the second pendulum
    constraint2 = (
        biorbd_casadi_model.markers(q_sym)[4].to_mx()[1:] - biorbd_casadi_model.globalJCS(q_sym, 2).to_mx()[1:3, 3]
    )
    constraint = vertcat(constraint1, constraint2)
    fcn_constraint = Function("constraint", [q_sym], [constraint], ["q"], ["constraint"]).expand()
    fcn_jacobian = Function("jacobian", [q_sym], [jacobian(constraint, q_sym)], ["q"], ["jacobian"]).expand()

    # Test the constraint
    print(fcn_constraint(all_q_t0))

    # Variational integrator
    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        constraints=fcn_constraint,
        jac=fcn_jacobian,
        q_init=np.concatenate((all_q_t0[:, np.newaxis], all_q_t1[:, np.newaxis]), axis=1),
    )

    q_vi, lambdas_vi = vi.integrate()

    tic2 = t.time()
    print(tic2 - tic0)

    if unit_test:
        import bioviz

        b = bioviz.Viz(Models.THREE_PENDULUMS.value)
        b.load_movement(q_vi)
        b.exec()

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(6, 2)
        axs[0, 0].plot(
            np.arange(0, time, time_step), q_vi[0, :], label="Variational Integrator", color="red", linestyle="-"
        )
        axs[0, 1].plot(
            np.arange(0, time, time_step), q_vi[1, :], label="Variational Integrator", color="red", linestyle="-"
        )
        axs[1, 0].plot(
            np.arange(0, time, time_step), q_vi[2, :], label="Variational Integrator", color="red", linestyle="-"
        )
        axs[1, 1].plot(
            np.arange(0, time, time_step), q_vi[3, :], label="Variational Integrator", color="red", linestyle="-"
        )
        axs[2, 0].plot(
            np.arange(0, time, time_step), q_vi[4, :], label="Variational Integrator", color="red", linestyle="-"
        )
        axs[2, 1].plot(
            np.arange(0, time, time_step), q_vi[5, :], label="Variational Integrator", color="red", linestyle="-"
        )
        axs[3, 0].plot(
            np.arange(0, time, time_step), q_vi[6, :], label="Variational Integrator", color="red", linestyle="-"
        )
        axs[3, 1].plot(
            np.arange(0, time, time_step), lambdas_vi[0, :], label="Variational Integrator", color="red", linestyle="-"
        )
        axs[4, 0].plot(
            np.arange(0, time, time_step), lambdas_vi[1, :], label="Variational Integrator", color="red", linestyle="-"
        )
        axs[4, 1].plot(
            np.arange(0, time, time_step), lambdas_vi[2, :], label="Variational Integrator", color="red", linestyle="-"
        )
        axs[5, 0].plot(
            np.arange(0, time, time_step), lambdas_vi[3, :], label="Variational Integrator", color="red", linestyle="-"
        )

        axs[0, 0].set_title("q0")
        axs[0, 1].set_title("q1")
        axs[1, 0].set_title("q2")
        axs[1, 1].set_title("q3")
        axs[2, 0].set_title("q4")
        axs[2, 1].set_title("q5")
        axs[3, 0].set_title("q6")
        axs[3, 1].set_title("lambda0")
        axs[4, 0].set_title("lambda1")
        axs[4, 1].set_title("lambda2")
        axs[5, 0].set_title("lambda3")

        # Plot total energy
        plt.figure()
        plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), color="red")
        plt.title("Total energy with variational integrator")

        # Verify the constraint respect
        plt.figure()
        plt.plot(fcn_constraint(q_vi).toarray()[0, :], label="constraint0")
        plt.plot(fcn_constraint(q_vi).toarray()[1, :], label="constraint1")
        plt.plot(fcn_constraint(q_vi).toarray()[2, :], label="constraint2")
        plt.plot(fcn_constraint(q_vi).toarray()[3, :], label="constraint3")
        plt.title("Constraint respect")
        plt.legend()

        plt.show()

    return q_vi

if __name__ == "__main__":
    three_pendulums(unit_test=True)
