"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
"""
from typing import Tuple
from enum import Enum
import numpy as np
from casadi import MX, SX, jacobian, Function, rootfinder, transpose, vertcat
import biorbd_casadi


class QuadratureRule(Enum):
    """
    The different discrete methods
    """

    MIDPOINT = "midpoint"
    LEFT_APPROXIMATION = "left_approximation"
    RIGHT_APPROXIMATION = "right_approximation"
    TRAPEZOIDAL = "trapezoidal"


# class ControlType(Enum):
#     """
#     The different control types
#     """
#
#     CONSTANT = "constant"  # piecewise constant
#     LINEAR_CONTINUOUS = "linear_continuous"  # piecewise linear


class VariationalIntegratorType(Enum):
    """
    The different variational integrator types
    """

    DISCRETE_EULER_LAGRANGE = "discrete_euler_lagrange"
    CONSTRAINED_DISCRETE_EULER_LAGRANGE = "constrained_discrete_euler_lagrange"
    FORCED_DISCRETE_EULER_LAGRANGE = "forced_discrete_euler_lagrange"
    FORCED_CONSTRAINED_DISCRETE_EULER_LAGRANGE = "forced_constrained_discrete_euler_lagrange"


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
    forces: Callable
        The forces of the system only one callable for now, it needs to be a function of q, qdot
    controls: np.ndarray
        The controls of the system, it needs to be the size of the number of degrees of freedom
    """

    def __init__(
            self,
            biorbd_model: biorbd_casadi.Model,
            time_step: float,
            time: float,
            q_init: np.ndarray,
            constraints: Function = None,
            jac: Function = None,
            discrete_lagrangian_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
            controls: np.ndarray = None,
            # control_type: ControlType = ControlType.CONSTANT,
            forces: Function = None,
            # force_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
    ):

        # check q_init
        if q_init.shape[0] != biorbd_model.nbQ():
            raise RuntimeError("q_init must have the same number of rows as the number of degrees of freedom")
        if q_init.shape[1] != 2:
            raise RuntimeError("q_init must have two columns, one q0 and one q1")
        self.q_init = q_init
        self.q1_num = q_init[:, 0]
        self.q2_num = q_init[:, 1]

        self.biorbd_model = biorbd_model
        self.time_step = time_step
        self.time = time
        self.nb_steps = int(time / time_step)

        self.constraints = constraints
        self.jac = jac
        self.discrete_lagrangian_approximation = discrete_lagrangian_approximation

        if controls is None:
            controls = np.zeros((self.biorbd_model.nbQ(), self.nb_steps))
        elif controls.shape[0] != self.biorbd_model.nbQ():
            raise ValueError("The control must be of the same size as the number of degrees of freedom")
        elif controls.shape[1] != self.time / self.time_step:
            raise ValueError("The control must have the same number of time steps as the time of the simulation")
        self.controls = controls
        # self.control_type = control_type
        # self._controls_to_force_func()

        # Check the type of variational integrator
        if controls is None and forces is None and constraints is None:
            self.variational_integrator_type = VariationalIntegratorType.DISCRETE_EULER_LAGRANGE
        elif controls is None and forces is None and constraints is not None:
            self.variational_integrator_type = VariationalIntegratorType.CONSTRAINED_DISCRETE_EULER_LAGRANGE
        elif (controls is not None or forces is None) and constraints is None:
            self.variational_integrator_type = VariationalIntegratorType.FORCED_DISCRETE_EULER_LAGRANGE
        elif (controls is not None or forces is None) and constraints is not None:
            self.variational_integrator_type = VariationalIntegratorType.FORCED_CONSTRAINED_DISCRETE_EULER_LAGRANGE
        else:
            raise RuntimeError("The variational integrator type is not recognized")

        # self.force_approximation = force_approximation

        self._declare_mx()
        self._declare_discrete_euler_lagrange_equations()

    def _declare_mx(self):
        """
        Declare the MX variables
        """
        # declare q for each time step of the integration
        self.q_prev = MX.sym("q_prev", self.biorbd_model.nbQ(), 1)  # ti-1
        self.q_cur = MX.sym("q_cur", self.biorbd_model.nbQ(), 1)  # ti
        self.q_next = MX.sym("q_next", self.biorbd_model.nbQ(), 1)  # ti+1
        # declare lambda for each constraint
        if self.constraints is not None:
            self.lambdas = MX.sym("lambda", self.constraints.nnz_out(), 1)
        else:
            self.lambdas = MX.sym("lambda", (0, 0))

        self.control_minus = MX.sym("control_minus", self.biorbd_model.nbQ(), 1)
        self.control_plus = MX.sym("control_plus", self.biorbd_model.nbQ(), 1)

    def _declare_discrete_euler_lagrange_equations(self):
        """
        Declare the discrete Euler-Lagrange equations
        """
        # list of symbolic variables needed for the integration
        self.sym_list = [self.q_prev, self.q_cur, self.q_next, self.lambdas, self.control_minus, self.control_plus]

        # output of the discrete Euler-Lagrange equations
        output = [
            self.interface_discrete_euler_lagrange_equations(
                self.q_prev,
                self.q_cur,
                self.q_next,
                self.lambdas,
                self.control_minus,
                self.control_plus,
            )
        ]

        self.dela = Function(f"DEL", self.sym_list, output).expand()

    def _declare_residuals(self, q_prev, q_cur, control_minus, control_plus):
        """
        This function declares the residuals of the discrete Euler-Lagrange equations to be solved implicitly. All the
        entries are numerical values.
        """
        mx_residuals = self.dela(q_prev, q_cur, self.q_next, self.lambdas, control_minus, control_plus)
        decision_variables = self.q_next

        if self.variational_integrator_type == VariationalIntegratorType.CONSTRAINED_DISCRETE_EULER_LAGRANGE \
                or self.variational_integrator_type == VariationalIntegratorType.FORCED_CONSTRAINED_DISCRETE_EULER_LAGRANGE:

            decision_variables = vertcat(decision_variables, self.lambdas)
            mx_residuals = vertcat(mx_residuals, self.constraints(self.q_next))

        self.residuals = Function(
            "Residuals",
            [decision_variables],
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
        if self.discrete_lagrangian_approximation == QuadratureRule.MIDPOINT:
            q_discrete = (q1 + q2) / 2
            qdot_discrete = (q2 - q1) / self.time_step
            return MX(self.time_step) * self.lagrangian(q_discrete, qdot_discrete)
        elif self.discrete_lagrangian_approximation == QuadratureRule.LEFT_APPROXIMATION:
            q_discrete = q1
            qdot_discrete = (q2 - q1) / self.time_step
            return MX(self.time_step) * self.lagrangian(q_discrete, qdot_discrete)
        elif self.discrete_lagrangian_approximation == QuadratureRule.RIGHT_APPROXIMATION:
            q_discrete = q2
            qdot_discrete = (q2 - q1) / self.time_step
            return MX(self.time_step) * self.lagrangian(q_discrete, qdot_discrete)
        elif self.discrete_lagrangian_approximation == QuadratureRule.TRAPEZOIDAL:
            # from : M. West, “Variational integrators,” Ph.D. dissertation, California Inst.
            # Technol., Pasadena, CA, 2004. p 13
            qdot_discrete = (q2 - q1) / self.time_step
            return MX(self.time_step) / 2 * (self.lagrangian(q1, qdot_discrete) + self.lagrangian(q2, qdot_discrete))
        else:
            raise NotImplementedError(
                f"Discrete Lagrangian {self.discrete_lagrangian_approximation} is not implemented"
            )

    # def discrete_force(self, q_prev: MX | SX, q_cur: MX | SX) -> MX | SX:
    #     """
    #     Compute the discrete force of a biorbd model
    #
    #     Parameters
    #     ----------
    #     q_prev: MX | SX
    #         The generalized coordinates at the previous time step
    #     q_cur: MX | SX
    #         The generalized coordinates at the current time step
    #     if interested in force functions of f(q, qdot) then use the following function
    #     Returns
    #     -------
    #     The discrete force
    #     """
    #     # Note: not tested yet
    #     if self.force_approximation == QuadratureRule.MIDPOINT:
    #         q_discrete = (q_prev + q_cur) / 2
    #         qdot_discrete = (q_prev - q_cur) / self.time_step
    #         return MX(self.time_step) * self.force(q_discrete, qdot_discrete)
    #     elif self.force_approximation == QuadratureRule.LEFT_APPROXIMATION:
    #         q_discrete = q_prev
    #         qdot_discrete = (q_cur - q_prev) / self.time_step
    #         return MX(self.time_step) * self.force(q_discrete, qdot_discrete)
    #     elif self.force_approximation == QuadratureRule.RIGHT_APPROXIMATION:
    #         q_discrete = q_cur
    #         qdot_discrete = (q_cur - q_prev) / self.time_step
    #         return MX(self.time_step) * self.force(q_discrete, qdot_discrete)
    #     elif self.force_approximation == QuadratureRule.TRAPEZOIDAL:
    #         # from : M. West, “Variational integrators,” Ph.D. dissertation, California Inst.
    #         # Technol., Pasadena, CA, 2004. p 13
    #         qdot_discrete = (q_cur - q_prev) / self.time_step
    #         return MX(self.time_step) / 4 * (self.force(q_prev, qdot_discrete) + self.force(q_cur, qdot_discrete))
    #     else:
    #         raise NotImplementedError(
    #             f"Discrete Lagrangian {self.discrete_lagrangian_approximation} is not implemented"
    #         )

    def compute_p_current(self, q_prev: MX | SX, q_cur: MX | SX) -> MX | SX:
        """
        Compute the current p (momentum when there is no forces)

        Parameters
        ----------
        q_prev: MX | SX
            The generalized coordinates at the previous time step
        q_cur: MX | SX
            The generalized coordinates at the current time step

        Returns
        -------
        The current p
        """
        return transpose(jacobian(self.discrete_lagrangian(q_prev, q_cur), q_cur))

    def compute_pi_current(self, q_cur: MX | SX) -> MX | SX:
        """
        Compute the current pi, the constraint jacobian

        Parameters
        ----------
        q_cur: MX | SX
            The generalized coordinates at the current time step

        Returns
        -------
        The current pi, the constraint jacobian
        """
        return self.jac(q_cur)

    def interface_discrete_euler_lagrange_equations(self, q_prev, q_cur, q_next, lambdas, control_minus, control_plus) -> MX | SX:
        if self.variational_integrator_type == VariationalIntegratorType.DISCRETE_EULER_LAGRANGE:
            return self._discrete_euler_lagrange_equations(
                q_prev=q_prev,
                q_cur=q_cur,
                q_next=q_next,
            )
        elif self.variational_integrator_type == VariationalIntegratorType.CONSTRAINED_DISCRETE_EULER_LAGRANGE:
            return self._constrained_discrete_euler_lagrange_equations(
                q_prev=q_prev,
                q_cur=q_cur,
                q_next=q_next,
                lambdas=lambdas,
            )
        elif self.variational_integrator_type == VariationalIntegratorType.FORCED_DISCRETE_EULER_LAGRANGE:
            return self._forced_discrete_euler_lagrange_equations(
                q_prev=q_prev,
                q_cur=q_cur,
                q_next=q_next,
                control_minus=control_minus,
                control_plus=control_plus,
            )
        elif self.variational_integrator_type == VariationalIntegratorType.FORCED_CONSTRAINED_DISCRETE_EULER_LAGRANGE:
            return self._forced_constrained_discrete_euler_lagrange_equations(
                q_prev=q_prev,
                q_cur=q_cur,
                q_next=q_next,
                lambdas=lambdas,
                control_minus=control_minus,
                control_plus=control_plus,
            )
        else:
            raise NotImplementedError(f"Variational integrator type {self.variational_integrator_type} is not implemented")

    def _discrete_euler_lagrange_equations(
            self, q_prev: MX | SX, q_cur: MX | SX, q_next: MX | SX) -> MX | SX:
        """
        Compute the discrete Euler-Lagrange equations of a biorbd model

        Parameters
        ----------
        q_prev: MX | SX
            The generalized coordinates at the first time step
        q_cur: MX | SX
            The generalized coordinates at the second time step
        q_next: MX | SX
            The generalized coordinates at the third time step

        Returns
        -------
        The discrete Euler-Lagrange equations
        """
        p_current = self.compute_p_current(q_prev, q_cur)
        D1_Ld_q2_q3 = transpose(jacobian(self.discrete_lagrangian(q_cur, q_next), q_cur))
        return p_current + D1_Ld_q2_q3

    def _constrained_discrete_euler_lagrange_equations(
            self, q_prev: MX | SX, q_cur: MX | SX, q_next: MX | SX, lambdas: MX | SX = None) -> MX | SX:
        """
        Compute the discrete Euler-Lagrange equations of a biorbd model

        Parameters
        ----------
        q_prev: MX | SX
            The generalized coordinates at the first time step
        q_cur: MX | SX
            The generalized coordinates at the second time step
        q_next: MX | SX
            The generalized coordinates at the third time step
        lambdas: MX | SX
            The Lagrange multipliers of second current time step
        """
        p_current = self.compute_p_current(q_prev, q_cur)  # momentum at current time step
        pi_current = self.jac(q_cur)

        D1_Ld_qcur_qnext = transpose(jacobian(self.discrete_lagrangian(q_cur, q_next), q_cur))

        return p_current + D1_Ld_qcur_qnext - transpose(pi_current) @ lambdas

    def _forced_discrete_euler_lagrange_equations(
            self, q_prev: MX | SX, q_cur: MX | SX, q_next: MX | SX, control_minus: MX | SX, control_plus: MX | SX) -> MX | SX:
        """
        Compute the discrete Euler-Lagrange equations of a biorbd model

        Parameters
        ----------
        q_prev: MX | SX
            The generalized coordinates at the first time step
        q_cur: MX | SX
            The generalized coordinates at the second time step
        q_next: MX | SX
            The generalized coordinates at the third time step
        control_minus: MX | SX
            The generalized forces at the first time step
        control_plus: MX | SX
            The generalized forces at the second time step
        """
        p_current = transpose(jacobian(self.discrete_lagrangian(q_prev, q_cur),
                                       q_cur)) + control_plus  # momentum at current time step + force
        D1_Ld_qcur_qnext = transpose(jacobian(self.discrete_lagrangian(q_cur, q_next), q_cur))

        return p_current + D1_Ld_qcur_qnext + control_minus

    def _forced_constrained_discrete_euler_lagrange_equations(
            self, q_prev: MX | SX, q_cur: MX | SX, q_next: MX | SX, control_minus: MX | SX, control_plus: MX | SX,
            lambdas: MX | SX = None) -> MX | SX:
        """
        Compute the discrete Euler-Lagrange equations of a biorbd model

        Parameters
        ----------
        q_prev: MX | SX
            The generalized coordinates at the first time step
        q_cur: MX | SX
            The generalized coordinates at the second time step
        q_next: MX | SX
            The generalized coordinates at the third time step
        control_minus: MX | SX
            The generalized forces at the first time step
        control_plus: MX | SX
            The generalized forces at the second time step
        lambdas: MX | SX
            The Lagrange multipliers of second current time step
        """
        p_current = transpose(jacobian(self.discrete_lagrangian(q_prev, q_cur), q_cur)) + control_plus
        pi_current = self.jac(q_cur)

        D1_Ld_qcur_qnext = transpose(jacobian(self.discrete_lagrangian(q_cur, q_next), q_cur))

        return p_current + D1_Ld_qcur_qnext + control_minus - transpose(pi_current) @ lambdas

    def integrate(self):
        """
        Integrate the discrete euler lagrange over time
        """
        q_prev = self.q1_num
        q_cur = self.q2_num

        u_prev = self.controls[:, 0]
        u_cur = self.controls[:, 1]

        # initialize the outputs of the integrator
        q_all = np.zeros((self.biorbd_model.nbQ(), self.nb_steps))
        q_all[:, 0] = q_prev
        q_all[:, 1] = q_cur

        if self.constraints is not None:
            lambdas_all = np.zeros((self.constraints.nnz_out(), self.nb_steps))
            lambdas_num = lambdas_all[:, 0]
        else:
            lambdas_num = np.zeros((0, self.nb_steps))

        for i in range(2, self.nb_steps):

            # f(q_prev, q_cur, q_next) = 0, only q_next is unknown
            ifcn = self._declare_residuals(q_prev, q_cur, control_minus=u_prev, control_plus=u_cur)

            # q_cur as an initial guess
            v_init = self._dispatch_to_v(q_cur, lambdas_num)
            v_opt = ifcn(v_init)
            q_next, lambdas_num = self._dispatch_to_q_lambdas(v_opt)

            q_prev = q_cur
            q_cur = q_next

            # store the results
            if self.constraints is not None:
                lambdas_all[:, i] = lambdas_num.toarray().squeeze()
            q_all[:, i] = q_next.toarray().squeeze()

        return q_all, lambdas_all

    def _dispatch_to_v(self, q: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
        """
        Dispatch the q and lambdas to the correct format

        Parameters
        ----------
        q: np.ndarray
            The generalized coordinates
        lambdas: np.ndarray
            The Lagrange multipliers
        """
        v = q
        if self.constraints is not None:
            v = np.concatenate((v, lambdas), axis=0)
        return v

    def _dispatch_to_q_lambdas(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dispatch the v to the correct format

        Parameters
        ----------
        v: np.ndarray
            The generalized coordinates and Lagrange multipliers
        """
        q = v[: self.biorbd_model.nbQ()]
        if self.constraints is not None:
            lambdas = v[self.biorbd_model.nbQ():]
        else:
            lambdas = None
        return q, lambdas
