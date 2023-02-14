import numpy as np


def test_pendulum():
    from biorbd_examples import pendulum

    q_vi = pendulum.pendulum(time=600, time_step=0.01)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [-0.408176118310164],
        decimal=15,
    )


def test_one_pendulum():
    from biorbd_examples import one_pendulum

    q_vi = one_pendulum.one_pendulum(time=10, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.0, 0.0, 0.925589212069481],
        decimal=15,
    )


def test_one_pendulum_force():
    from biorbd_examples import one_pendulum_force

    q_vi = one_pendulum_force.one_pendulum_force(time=10, time_step=0.05)

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.0, 0.0, 1001.856438367177134],
        decimal=15,
    )


def test_pendulum_control():
    from biorbd_examples import pendulum_control

    q_vi = pendulum_control.pendulum()

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [3.196070289107271],
        decimal=15,
    )


def test_double_pendulum(time=60, time_step=0.05):
    from biorbd_examples import douple_pendulum

    q_vi = douple_pendulum.double_pendulum()

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [-0.002664549613123, -39.618243621555521],
        decimal=15,
    )


def test_two_pendulums(time=10, time_step=0.05):
    from biorbd_examples import two_pendulums

    q_vi = two_pendulums.two_pendulums()

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [0.161169726990580, 0.160472883755656, -0.987040249219426, -1.124584348566156],
        decimal=15,
    )


def test_three_pendulums(time=20, time_step=0.05):
    from biorbd_examples import three_pendulums

    q_vi = three_pendulums.three_pendulums()

    np.testing.assert_almost_equal(
        q_vi[:, -1],
        [1.101954274798004, 0.892092109121184, - 0.451853592266031, - 4.750585386762840,
         1.891362715077611, - 0.490040711448757, - 2.457996763996095],
        decimal=15,
    )
