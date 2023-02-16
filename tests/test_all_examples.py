import numpy as np
import importlib.util
from pathlib import Path


def biorbd_examples_folder() -> str:
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

    q_vi = module.pendulum(time=600, time_step=0.01)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [-0.408176118310164],
        decimal=15,
    )


def test_one_pendulum():
    module = load_module(biorbd_examples_folder() + "/one_pendulum.py")

    q_vi = module.one_pendulum(time=10, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.0, 0.0, 0.925589212069481],
        decimal=15,
    )


def test_one_pendulum_force():
    module = load_module(biorbd_examples_folder() + "/one_pendulum_force.py")

    q_vi = module.one_pendulum_force(time=10, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.0, 0.0, 1001.856438367177134],
        decimal=15,
    )


def test_pendulum_control():
    module = load_module(biorbd_examples_folder() + "/pendulum_control.py")

    q_vi = module.pendulum()

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [3.129467086425164],
        decimal=15,
    )


def test_double_pendulum():
    module = load_module(biorbd_examples_folder() + "/double_pendulum.py")

    q_vi = module.double_pendulum(time=60, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [-0.002664549613123, -39.618243621555521],
        decimal=15,
    )


def test_two_pendulums():
    module = load_module(biorbd_examples_folder() + "/two_pendulums.py")

    q_vi = module.two_pendulums(time=10, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.161169726990580, 0.160472883755656, -0.987040249219426, -1.124584348566156],
        decimal=15,
    )


def test_three_pendulums():
    module = load_module(biorbd_examples_folder() + "/three_pendulums.py")

    q_vi = module.three_pendulums(time=20, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [1.101954274798004, 0.892092109121184, - 0.451853592266031, - 4.750585386762840,
         1.891362715077611, - 0.490040711448757, - 2.457996763996095],
        decimal=15,
    )
