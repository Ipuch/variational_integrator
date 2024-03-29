from enum import Enum


class QuadratureRule(Enum):
    """
    The different discrete methods
    """

    MIDPOINT = "midpoint"
    LEFT_APPROXIMATION = "left_approximation"
    RIGHT_APPROXIMATION = "right_approximation"
    TRAPEZOIDAL = "trapezoidal"


class VariationalIntegratorType(Enum):
    """
    The different variational integrator types
    """

    DISCRETE_EULER_LAGRANGE = "discrete_euler_lagrange"
    CONSTRAINED_DISCRETE_EULER_LAGRANGE = "constrained_discrete_euler_lagrange"
    FORCED_DISCRETE_EULER_LAGRANGE = "forced_discrete_euler_lagrange"
    FORCED_CONSTRAINED_DISCRETE_EULER_LAGRANGE = "forced_constrained_discrete_euler_lagrange"


class ControlType(Enum):
    """
    The different control types
    """

    PIECEWISE_CONSTANT = "piecewise_constant"
    PIECEWISE_LINEAR = "piecewise_linear"


class InitialGuessApproximation(Enum):
    """
    The different initial guess approximations available for the Newton's method
    """

    LAGRANGIAN = "lagrangian"
    CURRENT = "current"
    EXPLICIT_EULER = "explicit_euler"
    SEMI_IMPLICIT_EULER = "semi_implicit_euler"
    CUSTOM = "custom"
