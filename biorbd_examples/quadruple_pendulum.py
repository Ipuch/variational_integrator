"""
"""
import biorbd_casadi

from casadi import MX, jacobian, Function, vertcat

from varint.minimal_variational_integrator import VariationalIntegrator

from biorbd_examples.utils import *


def quadruple_pendulum(time: float = 5.0, time_step: float = 0.05):
    """ """
    biorbd_casadi_model = biorbd_casadi.Model(Models.QUADRUPLE_PENDULUM.value)

    alpha = 1
    q_t0 = np.array([alpha, -2 * alpha, -alpha, 2 * alpha])

    # Build  constraints
    q_sym = MX.sym("q", (biorbd_casadi_model.nbQ(), 1))
    # The origin of the second pendulum is constrained to the tip of the first pendulum
    constraint = (biorbd_casadi_model.markers(q_sym)[3].to_mx() - biorbd_casadi_model.markers(q_sym)[7].to_mx())[1:]
    fcn_constraint = Function("constraint", [q_sym], [constraint], ["q"], ["constraint"]).expand()
    fcn_jacobian = Function("jacobian", [q_sym], [jacobian(constraint, q_sym)], ["q"], ["jacobian"]).expand()

    # # build  constraint
    # # the end of the third pendulum is constrained at the same height as the first pendulum
    # q_sym = MX.sym("q", (biorbd_casadi_model.nbQ(), 1))
    # q_end = MX([0, 1, 0])
    # constraint = (biorbd_casadi_model.markers(q_sym)[7].to_mx() - q_end)[2]
    # fcn_constraint = Function("constraint", [q_sym], [constraint], ["q"], ["constraint"]).expand()
    # fcn_jacobian = Function("jacobian", [q_sym], [jacobian(constraint, q_sym)], ["q"], ["jacobian"]).expand()

    # variational integrator
    vi = VariationalIntegrator(
        biorbd_model=biorbd_casadi_model,
        time_step=time_step,
        time=time,
        constraints=fcn_constraint,
        jac=fcn_jacobian,
        q_init=q_t0[:, np.newaxis],
        q_dot_init=np.zeros((biorbd_casadi_model.nbQ(), 1)),
    )
    q_vi, *_ = vi.integrate()

    import bioviz

    b = bioviz.Viz(Models.QUADRUPLE_PENDULUM.value)
    b.load_movement(q_vi)
    b.exec()


if __name__ == "__main__":
    quadruple_pendulum()
