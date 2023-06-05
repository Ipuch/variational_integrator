"""
This script is testing all the examples in the biorbd examples folder to verify the integrity of the code and the main
class VariationalIntegrator.
"""

import numpy as np
import importlib.util
from pathlib import Path


def biorbd_examples_folder() -> str:
    """return the path to the biorbd examples folder"""
    return str(Path(__file__).parent.parent / "biorbd_examples")


def load_module(path: str):
    """Load a module from a path"""
    module_name = path.split("/")[-1].split(".")[0]
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pendulum():
    """Test the pendulum example"""
    module = load_module(biorbd_examples_folder() + "/pendulum.py")

    q_vi, q_vi_dot = module.pendulum(time=1, time_step=0.01)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [-1.479865618493051],
        decimal=15,
    )

    np.testing.assert_almost_equal(
        q_vi_dot,
        [-1.999894365826913],
        decimal=15,
    )


def test_one_pendulum():
    module = load_module(biorbd_examples_folder() + "/one_pendulum.py")

    q_vi, _ = module.one_pendulum(time=10, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.0, 0.0, 0.743415690332072],
        decimal=13,
    )


def test_one_pendulum_force():
    module = load_module(biorbd_examples_folder() + "/one_pendulum_force.py")

    q_vi, _ = module.one_pendulum_force(time=10, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.0, 0.0, -0.177363694419747],
        decimal=13,
    )


def test_pendulum_control():
    module = load_module(biorbd_examples_folder() + "/pendulum_control.py")

    q_vi, q_vi_dot = module.pendulum()

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [3.129742547604385],
        decimal=15,
    )

    np.testing.assert_almost_equal(
        q_vi_dot,
        [0.574833259022850],
        decimal=15,
    )


def test_double_pendulum():
    module = load_module(biorbd_examples_folder() + "/double_pendulum.py")

    q_vi, q_vi_dot = module.double_pendulum(time=1, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.176934499338377, -1.630924739364348],
        decimal=13,
    )
    np.testing.assert_almost_equal(
        q_vi_dot,
        [-1.717140638154052, -5.855445602253749],
        decimal=13,
    )


def test_two_pendulums():
    module = load_module(biorbd_examples_folder() + "/two_pendulums.py")

    q_vi, _ = module.two_pendulums(time=10, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.035018794176383, 0.035011637264280, -0.999386904685105, -1.402946923108905],
        decimal=11,
    )


def test_three_pendulums():
    module = load_module(biorbd_examples_folder() + "/three_pendulums.py")

    q_vi, _ = module.three_pendulums(time=1, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [
            -0.073570544396147,
            -0.073504194060489,
            -0.997294907966304,
            -0.059725961312502,
            -0.133194652725969,
            -1.995511842877902,
            -0.757376202440198,
        ],
        decimal=6,
    )
