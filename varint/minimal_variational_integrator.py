"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
"""
from typing import Tuple
import numpy as np
from casadi import MX, SX, jacobian, Function, rootfinder, transpose, vertcat
import biorbd_casadi
from .enums import QuadratureRule, VariationalIntegratorType, ControlType


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
        q_dot_init: np.ndarray = None,
        constraints: Function = None,
        jac: Function = None,
        discrete_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
        controls: np.ndarray = None,
        control_type: ControlType = ControlType.PIECEWISE_CONSTANT,
        forces: Function = None,
        # force_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
        newton_descent_tolerance: float = 1e-14,
    ):
        # `check approximation` and `control_type`
        if discrete_approximation not in QuadratureRule:
            raise NotImplementedError(
                f"Discrete {self.discrete_approximation} is not implemented"
            )
        if control_type not in ControlType:
            raise NotImplementedError(
                f"Control {self.discrete_approximation} is not implemented"
            )

        self.biorbd_model = biorbd_model
        self.time_step = time_step
        self.time = time
        self.nb_steps = int(time / time_step)

        self.constraints = constraints
        self.jac = jac
        self.discrete_approximation = discrete_approximation
        self.control_type = control_type

        if controls is None:
            self.controls = np.zeros((self.biorbd_model.nbQ(), self.nb_steps))
        elif controls.shape[0] != self.biorbd_model.nbQ():
            raise ValueError("The control must be of the same size as the number of degrees of freedom")
        elif controls.shape[1] != self.time / self.time_step:
            raise ValueError("The control must have the same number of time steps as the time of the simulation")
        else:
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

        if jac is None:
            q_sym = MX.sym("q", (biorbd_model.nbQ(), 1))
            self.jac = Function("no_constraint", [q_sym], [MX.zeros(q_init.shape)], ["q"], ["zero"]).expand()

        self._declare_mx()
        self._declare_discrete_euler_lagrange_equations()
        self.newton_descent_tolerance = newton_descent_tolerance

        # check q_init
        if q_dot_init is not None:
            if q_init.shape[0] != biorbd_model.nbQ():
                raise RuntimeError("q_init must have the same number of rows as the number of degrees of freedom")
            if q_dot_init.shape[0] != biorbd_model.nbQ():
                raise RuntimeError("q_dot_init must have the same number of rows as the number of degrees of freedom")
            if q_init.shape[1] != 1:
                raise RuntimeError("If an initial velocity is given (q_dot_init), q_init must have one columns (q0)")
            if q_dot_init.shape[1] != 1:
                raise RuntimeError("q_dot_init must have one columns (q0_dot)")
            self.q_init = self._compute_initial_states(q_init, q_dot_init)
        else:
            if q_init.shape[0] != biorbd_model.nbQ():
                raise RuntimeError("q_init must have the same number of rows as the number of degrees of freedom")
            if q_init.shape[1] != 2:
                raise RuntimeError("q_init must have two columns, one q0 and one q1")
            self.q_init = q_init

        self.q1_num = self.q_init[:, 0]
        self.q2_num = self.q_init[:, 1]

    def _compute_initial_states(self, q_init, q_dot_init):
        # Declare the MX variables
        q0 = MX.sym("q0", self.biorbd_model.nbQ(), 1)
        q0_dot = MX.sym("q0_dot", self.biorbd_model.nbQ(), 1)
        q1 = MX.sym("q1", self.biorbd_model.nbQ(), 1)
        f0_minus = MX.sym("f0_minus", self.biorbd_model.nbQ(), 1)

        # Equation (3.15c) in the paper Discrete Mechanics and Optimal Control: an Analysis
        # (https://arxiv.org/pdf/0810.1386.pdf)
        D2_L_q0_q0dot = transpose(jacobian(self.lagrangian(q0, q0_dot), q0_dot))
        D1_Ld_q0_q1 = transpose(jacobian(self.discrete_lagrangian(q0, q1), q0))
        # The constraint term is added as in _discrete_euler_lagrange_equations
        constraint_term = transpose(self.jac(q0)) @ self.lambdas if self.constraints is not None else MX.zeros(
            self.biorbd_model.nbQ(), 1)
        output = [D2_L_q0_q0dot + D1_Ld_q0_q1 + f0_minus - constraint_term]

        initial_states_fun = Function("initial_states", [q0, q0_dot, q1, f0_minus, self.lambdas], output).expand()

        mx_residuals = initial_states_fun(q_init[:, 0], q_dot_init[:, 0], q1, self.controls[:, 0], self.lambdas)
        decision_variables = q1

        if (
            self.variational_integrator_type == VariationalIntegratorType.CONSTRAINED_DISCRETE_EULER_LAGRANGE
            or self.variational_integrator_type == VariationalIntegratorType.FORCED_CONSTRAINED_DISCRETE_EULER_LAGRANGE
        ):
            decision_variables = vertcat(decision_variables, self.lambdas)
            mx_residuals = vertcat(mx_residuals, self.constraints(q1))

        residuals = Function(
            "initial_states_residuals",
            [decision_variables],
            [mx_residuals],
        ).expand()

        # Create a implicit function instance to solve the system of equations
        opts = {}
        opts["abstol"] = self.newton_descent_tolerance
        ifcn = rootfinder("ifcn", "newton", residuals, opts)

        if self.constraints is not None:
            lambdas_all = np.zeros((self.constraints.nnz_out(), self.nb_steps))
            lambdas_num = lambdas_all[:, 0]
        else:
            lambdas_num = np.zeros((0, self.nb_steps))

        # q_cur as an initial guess
        v_init = self._dispatch_to_v(q_init[:, 0] + q_dot_init[:, 0] * self.time_step, lambdas_num)
        v_opt = ifcn(v_init)
        qdot_opt, _ = self._dispatch_to_q_lambdas(v_opt)

        return np.concatenate((q_init, qdot_opt), axis=1)

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

        self.control_prev = MX.sym("control_prev", self.biorbd_model.nbQ(), 1)
        self.control_cur = MX.sym("control_cur", self.biorbd_model.nbQ(), 1)
        self.control_next = MX.sym("control_next", self.biorbd_model.nbQ(), 1)

    def _declare_discrete_euler_lagrange_equations(self):
        """
        Declare the discrete Euler-Lagrange equations
        """
        # list of symbolic variables needed for the integration
        self.sym_list = [self.q_prev, self.q_cur, self.q_next, self.lambdas, self.control_prev, self.control_cur, self.control_next]

        # output of the discrete Euler-Lagrange equations
        output = [
            self._discrete_euler_lagrange_equations(
                self.q_prev,
                self.q_cur,
                self.q_next,
                self.control_prev,
                self.control_cur,
                self.control_next,
                self.lambdas,
            )
        ]

        self.dela = Function(f"DEL", self.sym_list, output).expand()

    def _declare_residuals(self, q_prev, q_cur, control_prev, control_cur, control_next):
        """
        This function declares the residuals of the discrete Euler-Lagrange equations to be solved implicitly. All the
        entries are numerical values.
        """
        mx_residuals = self.dela(q_prev, q_cur, self.q_next, self.lambdas, control_prev, control_cur, control_next)
        decision_variables = self.q_next

        if (
            self.variational_integrator_type == VariationalIntegratorType.CONSTRAINED_DISCRETE_EULER_LAGRANGE
            or self.variational_integrator_type == VariationalIntegratorType.FORCED_CONSTRAINED_DISCRETE_EULER_LAGRANGE
        ):
            decision_variables = vertcat(decision_variables, self.lambdas)
            mx_residuals = vertcat(mx_residuals, self.constraints(self.q_next))

        self.residuals = Function(
            "Residuals",
            [decision_variables],
            [mx_residuals],
        ).expand()

        # Create a implicit function instance to solve the system of equations
        opts = {}
        opts["abstol"] = self.newton_descent_tolerance
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

        return self.biorbd_model.KineticEnergy(q, qdot).to_mx() - self.biorbd_model.PotentialEnergy(q).to_mx()

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
        if self.discrete_approximation == QuadratureRule.MIDPOINT:
            q_discrete = (q1 + q2) / 2
            qdot_discrete = (q2 - q1) / self.time_step
            return MX(self.time_step) * self.lagrangian(q_discrete, qdot_discrete)
        elif self.discrete_approximation == QuadratureRule.LEFT_APPROXIMATION:
            q_discrete = q1
            qdot_discrete = (q2 - q1) / self.time_step
            return MX(self.time_step) * self.lagrangian(q_discrete, qdot_discrete)
        elif self.discrete_approximation == QuadratureRule.RIGHT_APPROXIMATION:
            q_discrete = q2
            qdot_discrete = (q2 - q1) / self.time_step
            return MX(self.time_step) * self.lagrangian(q_discrete, qdot_discrete)
        elif self.discrete_approximation == QuadratureRule.TRAPEZOIDAL:
            # from : M. West, “Variational integrators,” Ph.D. dissertation, California Inst.
            # Technol., Pasadena, CA, 2004. p 13
            qdot_discrete = (q2 - q1) / self.time_step
            return MX(self.time_step) / 2 * (self.lagrangian(q1, qdot_discrete) + self.lagrangian(q2, qdot_discrete))
        else:
            raise NotImplementedError(
                f"Discrete Lagrangian {self.discrete_approximation} is not implemented"
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
    #             f"Discrete Lagrangian {self.discrete_approximation} is not implemented"
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

    def _discrete_euler_lagrange_equations(
        self,
        q_prev: MX | SX,
        q_cur: MX | SX,
        q_next: MX | SX,
        control_prev: MX | SX,
        control_cur: MX | SX,
        control_next: MX | SX,
        lambdas: MX | SX = None,
    ) -> MX | SX:
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
        control_prev: MX | SX
            The generalized forces at the first time step
        control_cur: MX | SX
            The generalized forces at the second time step
        control_next: MX | SX
            The generalized forces at the third time step
        lambdas: MX | SX
            The Lagrange multipliers of second current time step
        """
        p_current = self.compute_p_current(q_prev, q_cur)  # momentum at current time step

        D1_Ld_qcur_qnext = transpose(jacobian(self.discrete_lagrangian(q_cur, q_next), q_cur))
        constraint_term = transpose(self.jac(q_cur)) @ lambdas if self.constraints is not None else MX.zeros(p_current.shape)

        return p_current + D1_Ld_qcur_qnext - constraint_term + self.control_approximation(control_prev, control_cur) + self.control_approximation(control_cur, control_next)

    def control_approximation(
        self,
        control_minus: MX | SX,
        control_plus: MX | SX,
    ):
        """
        Compute the term associated to the discrete forcing.

        Parameters
        ----------
        control_minus: MX | SX
            Control at t_k (or t{k-1})
        control_plus: MX | SX
            Control at t_{k+1} (or tk)
        Returns
        ----------
        The term associated to the controls in the Lagrangian equations.
        Johnson, E. R., & Murphey, T. D. (2009).
        Scalable Variational Integrators for Constrained Mechanical Systems in Generalized Coordinates.
        IEEE Transactions on Robotics, 25(6), 1249–1261. doi:10.1109/tro.2009.2032955
        """
        if self.control_type == ControlType.PIECEWISE_CONSTANT:
            return 1 / 2 * control_minus * self.time_step

        elif self.control_type == ControlType.PIECEWISE_LINEAR:
            if self.discrete_approximation == QuadratureRule.MIDPOINT:
                return 1 / 2 * (control_minus + control_plus) / 2 * self.time_step
            elif self.discrete_approximation == QuadratureRule.LEFT_APPROXIMATION:
                return 1 / 2 * control_minus * self.time_step
            elif self.discrete_approximation == QuadratureRule.RIGHT_APPROXIMATION:
                return 1 / 2 * control_plus * self.time_step
            elif self.discrete_approximation == QuadratureRule.TRAPEZOIDAL:
                raise NotImplementedError(
                    f"Discrete {self.discrete_approximation} is not implemented for {self.control_type}"
                )

    def integrate(self):
        """
        Integrate the discrete euler lagrange over time
        """
        q_prev = self.q1_num
        q_cur = self.q2_num

        # initialize the outputs of the integrator
        q_all = np.zeros((self.biorbd_model.nbQ(), self.nb_steps))
        q_all[:, 0] = q_prev
        q_all[:, 1] = q_cur

        if self.constraints is not None:
            lambdas_all = np.zeros((self.constraints.nnz_out(), self.nb_steps))
            lambdas_num = lambdas_all[:, 0]
        else:
            lambdas_all = None
            lambdas_num = np.zeros((0, self.nb_steps))

        for i in range(2, self.nb_steps):

            if self.controls is not None:
                u_prev = self.controls[:, i - 2]
                u_cur = self.controls[:, i - 1]
                u_next = self.controls[:, i]
            else:
                u_prev = None
                u_cur = None
                u_next = None

            # f(q_prev, q_cur, q_next) = 0, only q_next is unknown
            ifcn = self._declare_residuals(q_prev, q_cur, control_prev=u_prev, control_cur=u_cur, control_next=u_next)

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
            lambdas = v[self.biorbd_model.nbQ() :]
        else:
            lambdas = None
        return q, lambdas
