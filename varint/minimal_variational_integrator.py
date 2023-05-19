"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
"""
from typing import Tuple
import numpy as np
from casadi import DM, MX, SX, jacobian, Function, rootfinder, transpose, vertcat
import biorbd_casadi
import biorbd
from .enums import QuadratureRule, VariationalIntegratorType, ControlType, InitialGuessApproximation


class VariationalIntegrator:
    """
    This class to build a variational integrator based on the discrete Lagrangian and a first order quadrature method.

    Attributes
    ----------
    biorbd_model: biorbd_casadi.Model
        The biorbd model.
    nb_steps: int
        The number of steps of the integration.
    time: float
        The duration of the integration.
    nb_steps: int
        The number of steps of the integration.
    constraints: Callable
        The constraints of the system only one callable for now.
    jac: Callable
        The jacobian of the constraints of the system only one callable for now.
    discrete_approximation: QuadratureRule
        The quadrature rule used to approximate the discrete Lagrangian.
    controls: np.ndarray
        The controls of the system, it needs to be the size of the number of degrees of freedom.
    control_type: ControlType
        The type of control used.
    # forces: Callable
    #     The forces of the system only one callable for now, it needs to be a function of q, qdot.
    newton_descent_tolerance: float
        The tolerance of the Newton descent.
    ignore_initial_constraints: bool
        If the initial constraints are ignored or not.
    variational_integrator_type: VariationalIntegratorType
        The type of variational integrator used.
    q0_num: np.ndarray
        The position at the first (0) time step.
    q1_num: np.ndarray
        The position at the second (1) time step.
    lambdas_0: np.ndarray
        The initial value of the Lagrange multipliers.
    lambdas: MX
        The Lagrange multipliers.
    q_prev: MX
        The position at the previous time step.
    q_cur: MX
        The position at the current time step.
    q_next: MX
        The position at the next time step.
    control_prev: MX
        The control at the previous time step.
    control_cur: MX
        The control at the current time step.
    control_next: MX
        The control at the next time step.
    dela: Function
        The discrete Lagrangian callable function.
    sym_list: list
        The list of symbolic variables needed for the integration.
    residuals: Function
        The residuals of the Newton descent.
    debug_mode: bool
        If the debug mode is activated the integrator will indicate where it stopped if it is because of the
        Newton's method.

    Methods
    -------
    lagrangian(self, q: MX | SX, qdot: MX | SX) -> MX | SX
        Compute the Lagrangian of a biorbd model.
    discrete_lagrangian(self, q1: MX | SX, q2: MX | SX) -> MX | SX
        Compute the discrete Lagrangian of a biorbd model.
    control_approximation(self, control_minus: MX | SX, control_plus: MX | SX) -> MX | SX
        Compute the term associated to the discrete forcing. The term associated to the controls in the Lagrangian
        equations is homogeneous to a force or a torque multiplied by a time.
    integrate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        Integrate the discrete euler lagrange over time.
    """

    def __init__(
        self,
        biorbd_model: biorbd_casadi.Model,
        nb_steps: int,
        time: float,
        q_init: np.ndarray,
        q_dot_init: np.ndarray,
        constraints: Function = None,
        jac: Function = None,
        discrete_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
        controls: np.ndarray = None,
        control_type: ControlType = ControlType.PIECEWISE_CONSTANT,
        forces: Function = None,
        # force_approximation: QuadratureRule = QuadratureRule.TRAPEZOIDAL,
        newton_descent_tolerance: float = 1e-14,
        ignore_initial_constraints=False,
        initial_guess_approximation: InitialGuessApproximation = InitialGuessApproximation.CURRENT,
        initial_guess_custom: np.ndarray = None,
        debug_mode: bool = False,
    ):
        """
        Parameters
        ----------
        biorbd_model: biorbd_casadi.Model
            The biorbd model.
        nb_steps: int
            The number of steps of the integration.
        time: float
            The duration of the integration.
        q_init: np.ndarray
            The initial position of the system. q_init must have the same number of rows as the number of degrees of
            freedom. It needs to have only one column.
        q_dot_init: np.ndarray
            The initial velocity of the system. q_dot_init must have the same number of rows as the number of degrees of
            freedom. It needs to have only one column.
        constraints: Callable
            The constraints of the system only one callable for now.
        jac: Callable
            The jacobian of the constraints of the system only one callable for now.
        discrete_approximation: QuadratureRule
            The quadrature rule used to approximate the discrete Lagrangian.
        controls: np.ndarray
            The controls of the system, it needs to be the size of the number of degrees of freedom.
        control_type: ControlType
            The type of control used.
        forces: Callable
            The forces of the system only one callable for now, it needs to be a function of q, qdot.
        newton_descent_tolerance: float
            The tolerance of the newton descent method.
        initial_guess_approximation: InitialGuessApproximation
            The initial guess approximation used.
        initial_guess_custom: np.ndarray | None
            The custom initial guess custom used if initial_guess_approximation == InitialGuessApproximation.CUSTOM.
        debug_mode: bool
            If the debug mode is activated the integrator will indicate where it stopped if it is because of the
            Newton's method.
        """
        # Check `discrete_approximation` and `control_type`
        if discrete_approximation not in QuadratureRule:
            raise NotImplementedError(f"Discrete {discrete_approximation} is not implemented")
        if control_type not in ControlType:
            raise NotImplementedError(f"Control {control_type} is not implemented")

        self.biorbd_model = biorbd_model
        self.biorbd_numpy_model = biorbd.Model(biorbd_model.path().absolutePath().to_string())
        self.time = time
        self.nb_steps = nb_steps
        self.time_step = time / nb_steps

        self.constraints = constraints
        self.jac = jac
        self.discrete_approximation = discrete_approximation
        self.control_type = control_type
        self.initial_guess_approximation = initial_guess_approximation
        if initial_guess_approximation == InitialGuessApproximation.CUSTOM:
            if initial_guess_custom is None:
                raise ValueError("The initial guess is set to CUSTOM but no initial guess was given.")
            if initial_guess_custom.shape != (self.biorbd_model.nbQ(), self.nb_steps):
                raise ValueError(
                    f"The initial guess must be of the same size as the number of degrees of freedom "
                    f"and the number of time steps. The initial guess's shape is "
                    f"{initial_guess_custom.shape} and it should be ({self.biorbd_model.nbQ()}, "
                    f"{self.nb_steps})"
                )

        self.initial_guess_custom = initial_guess_custom

        if controls is None:
            self.controls = np.zeros((self.biorbd_model.nbQ(), self.nb_steps))
        elif controls.shape[0] != self.biorbd_model.nbQ():
            raise ValueError("The control must be of the same size as the number of degrees of freedom")
        elif controls.shape[1] != self.nb_steps:
            raise ValueError(
                f"The control must have the same number of time steps as the time of the simulation"
                f"The number of step is {self.nb_steps} and the control has {controls.shape[1]} steps."
            )
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
        self.debug_mode = debug_mode

        # check q_init
        if q_init.shape[0] != biorbd_model.nbQ():
            raise ValueError("q_init must have the same number of rows as the number of degrees of freedom")
        if q_dot_init.shape[0] != biorbd_model.nbQ():
            raise ValueError("q_dot_init must have the same number of rows as the number of degrees of freedom")
        if q_init.shape[1] != 1:
            raise ValueError("If an initial velocity is given (q_dot_init), q_init must have one columns (q0)")
        if q_dot_init.shape[1] != 1:
            raise ValueError("q_dot_init must have one columns (q0_dot)")

        # `check constraints`
        if constraints is not None and not ignore_initial_constraints:
            try:
                np.testing.assert_almost_equal(
                    constraints(q_init),
                    np.zeros((constraints.nnz_out(), 1)),
                    decimal=15,
                )
            except:
                raise ValueError("The initial position does not respect the constraints.")

        self.q0_num, self.q1_num, self.lambdas_0 = self._compute_initial_states(q_init.squeeze(), q_dot_init.squeeze())

    def _compute_initial_states(self, q_init: np.ndarray, q_dot_init: np.ndarray) -> Tuple[DM, DM, DM | None]:
        """
        Compute the initial states of the system from the initial position and velocity.

        Parameters
        ----------
        q_init: np.ndarray
            The initial position of the system.
        q_dot_init: np.ndarray
            The initial velocity of the system.

        Returns
        -------
        (q0, q1, lambdas0): Tuple[DM, DM, DM | None]
            `q0` is the initial position of the system. `q1` is the initial velocity of the system. `lambdas0` are the
            initial lagrange multipliers of the system if there are constraints, None otherwise.
        """
        # Declare the MX variables
        q0 = MX.sym("q0", self.biorbd_model.nbQ(), 1)
        q0_dot = MX.sym("q0_dot", self.biorbd_model.nbQ(), 1)
        q1 = MX.sym("q1", self.biorbd_model.nbQ(), 1)
        f0_minus = MX.sym("f0_minus", self.biorbd_model.nbQ(), 1)

        # The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
        # constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912), equations (14) and the
        # indications given just before the equation (18) for p0 and pN.
        D2_L_q0_q0dot = transpose(jacobian(self.lagrangian(q0, q0_dot), q0_dot))
        D1_Ld_q0_q1 = transpose(jacobian(self.discrete_lagrangian(q0, q1), q0))
        # The constraint term is added as in _discrete_euler_lagrange_equations
        constraint_term = (
            1 / 2 * transpose(self.jac(q0)) @ self.lambdas
            if self.constraints is not None
            else MX.zeros(self.biorbd_model.nbQ(), 1)
        )
        output = [D2_L_q0_q0dot + D1_Ld_q0_q1 + f0_minus - constraint_term]

        initial_states_fun = Function("initial_states", [q0, q0_dot, q1, f0_minus, self.lambdas], output).expand()

        mx_residuals = initial_states_fun(
            q_init, q_dot_init, q1, self.control_approximation(self.controls[:, 0], self.controls[:, 1]), self.lambdas
        )
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

        # Create an implicit function instance to solve the system of equations
        opts = {"abstol": self.newton_descent_tolerance}
        ifcn = rootfinder("ifcn", "newton", residuals, opts)

        if self.constraints is not None:
            lambdas_all = np.zeros((self.constraints.nnz_out(), self.nb_steps))
            lambdas_num = lambdas_all[:, 0]
        else:
            lambdas_num = np.zeros((0, self.nb_steps))

        v_init = self._dispatch_to_v(q_init + q_dot_init * self.time_step, lambdas_num)
        v_opt = ifcn(v_init)
        q1_opt, lambdas_0_opt = self._dispatch_to_q_lambdas(v_opt)

        return DM(q_init), q1_opt, lambdas_0_opt

    def _compute_final_velocity(self, q_penultimate: np.ndarray, q_ultimate: np.ndarray) -> Tuple[DM, DM | None]:
        """
        Compute the final velocity of the system from the initial position and velocity.

        Parameters
        ----------
        q_penultimate: np.ndarray
            The penultimate position of the system.
        q_ultimate: np.ndarray
            The ultimate position of the system.

        Returns
        -------
        (qN_dot, lambdasN): Tuple[DM, DM | None]
            `qNdot` is the final velocity of the system. `lambdasN` are he final lagrange multipliers of the system.
        """
        # Declare the MX variables
        qN = MX.sym("qN", self.biorbd_model.nbQ(), 1)
        qN_dot = MX.sym("qN_dot", self.biorbd_model.nbQ(), 1)
        qN_minus_1 = MX.sym("qN_minus_1", self.biorbd_model.nbQ(), 1)
        fd_plus = MX.sym("fd_plus", self.biorbd_model.nbQ(), 1)

        # The following equation as been calculated thanks to the paper "Discrete mechanics and optimal control for
        # constrained systems" (https://onlinelibrary.wiley.com/doi/epdf/10.1002/oca.912), equations (14) and the
        # indications given just before the equation (18) for p0 and pN.
        D2_L_qN_qN_dot = transpose(jacobian(self.lagrangian(qN, qN_dot), qN_dot))
        D2_Ld_qN_minus_1_qN = transpose(jacobian(self.discrete_lagrangian(qN_minus_1, qN), qN))
        # The constraint term is added as in _discrete_euler_lagrange_equations
        constraint_term = (
            1 / 2 * transpose(self.jac(qN)) @ self.lambdas
            if self.constraints is not None
            else MX.zeros(self.biorbd_model.nbQ(), 1)
        )
        output = [-D2_L_qN_qN_dot + D2_Ld_qN_minus_1_qN - constraint_term + fd_plus]

        final_states_fun = Function("final_velocity", [qN, qN_dot, qN_minus_1, fd_plus, self.lambdas], output).expand()

        mx_residuals = final_states_fun(
            q_ultimate,
            qN_dot,
            q_penultimate,
            self.control_approximation(self.controls[:, -2], self.controls[:, -1]),
            self.lambdas,
        )
        decision_variables = qN_dot

        if (
            self.variational_integrator_type == VariationalIntegratorType.CONSTRAINED_DISCRETE_EULER_LAGRANGE
            or self.variational_integrator_type == VariationalIntegratorType.FORCED_CONSTRAINED_DISCRETE_EULER_LAGRANGE
        ):
            decision_variables = vertcat(decision_variables, self.lambdas)
            mx_residuals = vertcat(mx_residuals, self.constraints(qN_dot))

        residuals = Function(
            "final_states_residuals",
            [decision_variables],
            [mx_residuals],
        ).expand()

        # Create an implicit function instance to solve the system of equations
        opts = {"abstol": self.newton_descent_tolerance}
        ifcn = rootfinder("ifcn", "newton", residuals, opts)

        if self.constraints is not None:
            lambdas_all = np.zeros((self.constraints.nnz_out(), self.nb_steps))
            lambdas_num = lambdas_all[:, 0]
        else:
            lambdas_num = np.zeros((0, self.nb_steps))

        v_init = self._dispatch_to_v((q_ultimate - q_penultimate) / self.time_step, lambdas_num)
        v_opt = ifcn(v_init)
        qN_dot_opt, lambdasN_opt = self._dispatch_to_q_lambdas(v_opt)

        return qN_dot_opt, lambdasN_opt

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
        self.sym_list = [
            self.q_prev,
            self.q_cur,
            self.q_next,
            self.lambdas,
            self.control_prev,
            self.control_cur,
            self.control_next,
        ]

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

        # Create an implicit function instance to solve the system of equations
        opts = {"abstol": self.newton_descent_tolerance}
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
            raise NotImplementedError(f"Discrete Lagrangian {self.discrete_approximation} is not implemented")

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

    def _compute_p_current(self, q_prev: MX | SX, q_cur: MX | SX) -> MX | SX:
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
        p_current = self._compute_p_current(q_prev, q_cur)  # momentum at current time step

        D1_Ld_qcur_qnext = transpose(jacobian(self.discrete_lagrangian(q_cur, q_next), q_cur))
        constraint_term = (
            transpose(self.jac(q_cur)) @ lambdas if self.constraints is not None else MX.zeros(p_current.shape)
        )

        return (
            p_current
            + D1_Ld_qcur_qnext
            - constraint_term
            + self.control_approximation(control_prev, control_cur)
            + self.control_approximation(control_cur, control_next)
        )

    def control_approximation(
        self,
        control_minus: MX | SX,
        control_plus: MX | SX,
    ):
        """
        Compute the term associated to the discrete forcing. The term associated to the controls in the Lagrangian
        equations is homogeneous to a force or a torque multiplied by a time.

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

    def _compute_initial_guess(self, q_prev: DM, q_cur: DM, u_cur: np.ndarray, time_step: int) -> DM:
        """
        Compute the initial guess for the next time step.

        Parameters
        ----------
        q_prev: DM
            The coordinates at the previous time step.
        q_cur: DM
            The coordinates at the current time step.
        u_cur: np.ndarray
            The controls at the current time step.
        time_step: int
            The current time step, i.e. iteration number.

        Returns
        -------
        q_guess: DM
            The initial guess for the next time step.
        """
        # The first three initial guesses are issued from https://arxiv.org/pdf/1609.02898.pdf (3.3).
        if self.initial_guess_approximation == InitialGuessApproximation.CURRENT:
            q_guess = q_cur
        elif self.initial_guess_approximation == InitialGuessApproximation.EXPLICIT_EULER:
            qdot_cur_array = 1 / self.time_step * (q_cur - q_prev)
            q_guess = q_cur + self.time_step * qdot_cur_array
        elif self.initial_guess_approximation == InitialGuessApproximation.SEMI_IMPLICIT_EULER:
            q_cur_array = q_cur.toarray().squeeze()
            q_prev_array = q_prev.toarray().squeeze()
            qdot_cur_array = 1 / self.time_step * (q_cur_array - q_prev_array)
            u_cur_array = u_cur
            qddot_cur_array = self.biorbd_numpy_model.ForwardDynamics(
                q_cur_array, qdot_cur_array, u_cur_array
            ).to_array()
            qdot_next_array = qdot_cur_array + self.time_step * qddot_cur_array
            q_guess = DM(q_cur_array + self.time_step * qdot_next_array)
        # The following initial guess is issued from http://journals.cambridge.org/abstract_S096249290100006X (2.1.1).
        elif self.initial_guess_approximation == InitialGuessApproximation.LAGRANGIAN:
            q_cur_array = q_cur.toarray().squeeze()
            q_prev_array = q_prev.toarray().squeeze()
            D2_Ld_qprev_qcur = Function(
                "D2_Ld_qprev_qcur",
                [self.q_prev, self.q_cur],
                [transpose(jacobian(self.discrete_lagrangian(self.q_prev, self.q_cur), self.q_cur))],
            ).expand()
            pk = D2_Ld_qprev_qcur(q_prev_array, q_cur_array)
            M_inv = self.biorbd_numpy_model.massMatrixInverse(q_cur_array).to_array()
            q_guess = DM(q_cur + (M_inv @ pk * self.time_step).toarray())
        # It is also possible to give an array of initial guesses.
        elif self.initial_guess_approximation == InitialGuessApproximation.CUSTOM:
            q_guess = DM(self.initial_guess_custom[:, time_step])
        else:
            raise NotImplementedError(f"Initial guess {self.initial_guess_approximation} is not implemented yet.")
        return q_guess

    def integrate(self):
        """
        Integrate the discrete euler lagrange over time.
        """
        q_prev = self.q0_num
        q_cur = self.q1_num

        # initialize the outputs of the integrator
        q_all = np.zeros((self.biorbd_model.nbQ(), self.nb_steps))
        q_all[:, 0] = q_prev.toarray().squeeze()
        q_all[:, 1] = q_cur.toarray().squeeze()

        if self.constraints is not None:
            lambdas_all = np.zeros((self.constraints.nnz_out(), self.nb_steps))
            lambdas_all[:, 0] = self.lambdas_0.toarray().squeeze()
            lambdas_num = DM(lambdas_all[:, 0])
        else:
            lambdas_all = None
            lambdas_num = DM(np.zeros((0, self.nb_steps)))

        if self.debug_mode:
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
                ifcn = self._declare_residuals(
                    q_prev, q_cur, control_prev=u_prev, control_cur=u_cur, control_next=u_next
                )

                q_guess = self._compute_initial_guess(q_prev, q_cur, u_cur, i)

                v_init = self._dispatch_to_v(q_guess, lambdas_num)
                try:
                    v_opt = ifcn(v_init)
                except RuntimeError:
                    print(f"The integration crashed at the {i}th time step because of the rootfinding.")
                    break
                q_next, lambdas_num = self._dispatch_to_q_lambdas(v_opt)

                q_prev = q_cur
                q_cur = q_next

                # store the results
                if self.constraints is not None:
                    lambdas_all[:, i - 1] = lambdas_num.toarray().squeeze()
                    q_all[:, i] = q_next.toarray().squeeze()
                else:
                    q_all[:, i] = q_next.toarray().squeeze()
        else:
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
                ifcn = self._declare_residuals(
                    q_prev, q_cur, control_prev=u_prev, control_cur=u_cur, control_next=u_next
                )

                q_guess = self._compute_initial_guess(q_prev, q_cur, u_cur, i)

                v_init = self._dispatch_to_v(q_guess, lambdas_num)

                v_opt = ifcn(v_init)

                q_next, lambdas_num = self._dispatch_to_q_lambdas(v_opt)

                q_prev = q_cur
                q_cur = q_next

                # store the results
                if self.constraints is not None:
                    lambdas_all[:, i - 1] = lambdas_num.toarray().squeeze()
                    q_all[:, i] = q_next.toarray().squeeze()
                else:
                    q_all[:, i] = q_next.toarray().squeeze()

        q_dot_final, lambda_final = self._compute_final_velocity(q_all[:, -2], q_all[:, -1])

        if self.constraints is not None:
            lambdas_all[:, -1] = lambda_final.toarray().squeeze()

        return q_all, lambdas_all, q_dot_final.toarray().squeeze()

    def _dispatch_to_v(self, q: np.ndarray | DM, lambdas: np.ndarray | DM) -> np.ndarray | DM:
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

    def _dispatch_to_q_lambdas(self, v: np.ndarray | DM) -> Tuple[np.ndarray | DM, np.ndarray | DM | None]:
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
