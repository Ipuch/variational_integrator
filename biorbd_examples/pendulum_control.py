"""
This script is used to integrate the motion with a variational integrator based on the discrete Lagrangian,
and a first order quadrature method.
This example is a simple pendulum controlled with a torque calculated by bioptim to move the pendulum from 0 rad to pi
rad in 1 sec and 30 frames.
"""
import biorbd_casadi
from varint.minimal_variational_integrator import VariationalIntegrator

from biorbd_examples.utils import *

# `tau_optimal_control` and `q_optimal_control` where calculated with the example `pendulum.py` from `bioptim` with a
# model with only one dof (Rotx), `final_time=1`, `n_shooting=30` and without preventing the model from actively rotate
# (l.81).

tau_optimal_control = np.asarray(
    [
        [
            18.39841936,
            18.56183951,
            18.52361453,
            18.28517695,
            17.85120277,
            17.2300812,
            16.43428248,
            15.48044982,
            14.38906786,
            13.18363429,
            11.88936988,
            10.5316134,
            9.13412692,
            7.71755652,
            6.29825044,
            4.88754928,
            3.49156338,
            2.1113697,
            0.74351136,
            -0.61932967,
            -1.98760312,
            -3.37419226,
            -4.79380495,
            -6.26248594,
            -7.79726463,
            -9.41594264,
            -11.13702162,
            -12.97977061,
            -14.96443099,
            -17.11254998,
        ]
    ]
)

q_optimal_control = [
    0.0,
    0.01030821,
    0.04121574,
    0.09245982,
    0.16332607,
    0.25266637,
    0.35893181,
    0.48022312,
    0.61435943,
    0.75896352,
    0.91155867,
    1.06966929,
    1.23091611,
    1.39309694,
    1.55424584,
    1.712667,
    1.86694296,
    2.01591981,
    2.15867424,
    2.29446765,
    2.42269275,
    2.54281691,
    2.65432548,
    2.75666724,
    2.8492029,
    2.93115699,
    3.00157283,
    3.05926975,
    3.10280139,
    3.1304136,
    3.14,
]


def pendulum(
    tau_optimal_control: np.ndarray = tau_optimal_control,
    q_optimal_control: np.ndarray = q_optimal_control,
    unit_test: bool = False,
):
    biorbd_casadi_model = biorbd_casadi.Model(Models.PENDULUM.value)
    biorbd_model = biorbd.Model(Models.PENDULUM.value)

    import time as t

    time = 1
    nb_steps = 30

    tic0 = t.time()

    # variational integrator
    vi = VariationalIntegrator(
        biorbd_casadi_model=biorbd_casadi_model,
        nb_steps=30,
        time=time,
        q_init=np.array([[q_optimal_control[0]]]),
        q_dot_init=np.array([[0]]),
        controls=tau_optimal_control,
    )

    q_vi, _, q_vi_dot = vi.integrate()

    print(q_vi)

    tic1 = t.time()
    print(tic1 - tic0)

    if unit_test:
        import bioviz

        b = bioviz.Viz(Models.PENDULUM.value)
        b.load_movement(q_vi)
        b.exec()

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(q_vi[0, :], label="Variational Integrator")
        plt.plot(q_optimal_control, label="Optimal control")
        plt.title(f"Generalized coordinates")
        plt.legend()

        # Plot total energy for both methods
        plt.figure()
        plt.plot(discrete_total_energy(biorbd_model, q_vi, time_step), label="Mechanical energy", color="red")
        plt.title("Total energy")
        plt.legend()

        plt.show()

        np.set_printoptions(formatter={"float": lambda x: "{0:0.15f}".format(x)})
        print(q_vi[:, -1], q_vi_dot)

    return q_vi, q_vi_dot


if __name__ == "__main__":
    pendulum(tau_optimal_control, q_optimal_control, unit_test=True)
