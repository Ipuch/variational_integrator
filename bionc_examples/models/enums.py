"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
"""
from enum import Enum


class Models(Enum):
    """
    The different models
    """

    ONE_PENDULUM = "models/pendulum.nmod"
    TWENTY_PENDULUM = "models/20_link_pendulum.nMod"
