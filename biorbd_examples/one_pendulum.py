"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
This example is a simple pendulum freed and constrained in translation on the y-axis and the z-axis.
"""
import biorbd_casadi

from casadi import MX, jacobian, Function

from varint.minimal_variational_integrator import VariationalIntegrator

from biorbd_examples.utils import *


def one_pendulum(time: float = 10, time_step: float = 0.05, unit_test: bool = False):
    biorbd_casadi_model = biorbd_casadi.Model(Models.ONE_PENDULUM.value)
    biorbd_model = biorbd.Model(Models.ONE_PENDULUM.value)

    import time as t

    tic0 = t.time()

    q_t0 = np.array([1.54])

    all_q_t0 = np.array([0, 0, q_t0[0]])

    # build  constraint
    # the origin of the second pendulum is constrained to the tip of the first pendulum
    q_sym = MX.sym("q", (biorbd_casadi_model.nbQ(), 1))
    constraint = biorbd_casadi_model.markers(q_sym)[0].to_mx()[1:] - MX.zeros((2, 1))
    fcn_constraint = Function("constraint", [q_sym], [constraint], ["q"], ["constraint"]).expand()
    fcn_jacobian = Function("jacobian", [q_sym], [jacobian(constraint, q_sym)], ["q"], ["jacobian"]).expand()

    # variational integrator
    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        nb_steps=int(time / time_step),
        time=time,
        constraints=fcn_constraint,
        jac=fcn_jacobian,
        q_init=all_q_t0[:, np.newaxis],
        q_dot_init=np.zeros((biorbd_casadi_model.nbQ(), 1)),
    )
    # vi.set_initial_values(q_prev=1.54, q_cur=1.545)
    q_vi, lambdas_vi, q_vi_dot = vi.integrate()

    tic2 = t.time()
    print(tic2 - tic0)

    if unit_test:
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
        plt.figure()
        plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), label="Variational Integrator", color="red")
        plt.title("Total energy")

        # verify the constraint respect
        plt.figure()
        plt.plot(fcn_constraint(q_vi).toarray()[0, :], label="constraint0")
        plt.plot(fcn_constraint(q_vi).toarray()[1, :], label="constraint1")
        plt.legend()
        plt.title("Constraint respect")

        plt.show()

        np.set_printoptions(formatter={"float": lambda x: "{0:0.15f}".format(x)})
        print(q_vi[:, -1], q_vi_dot)

    return q_vi, q_vi_dot


if __name__ == "__main__":
    one_pendulum(unit_test=True)
