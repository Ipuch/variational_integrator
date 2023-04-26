import numpy as np
import importlib.util
from pathlib import Path


def biorbd_examples_folder() -> str:
    """ return the path to the biorbd examples folder"""
    return str(Path(__file__).parent.parent / "biorbd_examples")


def load_module(path: str):
    module_name = path.split("/")[-1].split(".")[0]
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pendulum():
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

    q_vi, q_vi_dot = module.one_pendulum(time=10, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.0, 0.0, 0.743415690332072],
        decimal=15,
    )

    np.testing.assert_almost_equal(
        q_vi_dot,
        [0.0, 0.0, -3.774176684457794],
        decimal=15,
    )


def test_one_pendulum_force():
    module = load_module(biorbd_examples_folder() + "/one_pendulum_force.py")

    q_vi, q_vi_dot = module.one_pendulum_force(time=10, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.0, 0.0, -0.177363694419747],
        decimal=15,
    )

    np.testing.assert_almost_equal(
        q_vi_dot,
        [0.0, 0.0, -4.084879154646298],
        decimal=15,
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

    q_vi, q_vi_dot = module.double_pendulum(time=60, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [-4.750533490616283, -8.262599424254001],
        decimal=15,
    )
    np.testing.assert_almost_equal(
        q_vi_dot,
        [1.379398727905979, -7.117106031979558],
        decimal=15,
    )


def test_two_pendulums():
    module = load_module(biorbd_examples_folder() + "/two_pendulums.py")

    q_vi, q_vi_dot = module.two_pendulums(time=10, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.035018794176383, 0.035011637264280, -0.999386904685105, -1.402946923108905],
        decimal=15,
    )

    np.testing.assert_almost_equal(
        q_vi_dot,
        [-5.763506918114953, 0.496601012013394, -0.867978936879964, -6.511856255514687],
        decimal=15,
    )


def test_three_pendulums():
    module = load_module(biorbd_examples_folder() + "/three_pendulums.py")

    q_vi, q_vi_dot = module.three_pendulums(time=20, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [
            1.651920786352850,
            0.996711215297497,
            0.081035506416568,
            7.205510421104589,
            1.793719284375165,
            -0.522933149910146,
            -1.928639911050799,
        ],
        decimal=15,
    )

    np.testing.assert_almost_equal(
        q_vi_dot,
        [
            3.233086046242024,
            -0.091365796914086,
            0.995817398499489,
            -4.850137395466243,
            0.899161882189885,
            0.858504191090646,
            3.539749724317079,
        ],
        decimal=15,
    )
