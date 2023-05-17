"""
This script is used to reintegrate the data for the somersault example and compare the variational integrator with
different initial guesses, RK4 and RK45.
"""
import scipy
import pickle
import numpy as np
import matplotlib.pyplot as plt

import biorbd_casadi

from varint import InitialGuessApproximation
from varint.minimal_variational_integrator import VariationalIntegrator

from biorbd_examples.utils import *


def get_created_data_from_pickle(file: str):
    """
    Creates data from pickle

    Parameters
    ----------
    file: str
        File where data has been saved.

    Returns
    -------
    Data
    """
    with open(file, "rb") as f:
        data_tmp = pickle.load(f)

        print(
            f"{file}: status: {data_tmp['status']}, cost:{data_tmp['cost']}, time to optimize: "
            f"{data_tmp['real_time_to_optimize']}, "
            f"nb_it: {data_tmp['iterations']}"
        )
        print(
            f"1ère phase : {data_tmp['time'][0][-1] - data_tmp['time'][0][0]}, "
            f"{data_tmp['states_no_intermediate'][0]['q'].shape[1]} nodes"
        )
        print(
            f"2ère phase : {data_tmp['time'][1][-1] - data_tmp['time'][1][0]}, "
            f"{data_tmp['states_no_intermediate'][1]['q'].shape[1]} nodes"
        )

        shape_0_1 = data_tmp["states_no_intermediate"][0]["q"].shape[1] - 1

        datas_shape = (
            data_tmp["states_no_intermediate"][0]["q"].shape[0],
            shape_0_1 + data_tmp["states_no_intermediate"][1]["q"].shape[1],
        )

        # q
        datas_q = np.zeros(datas_shape)
        datas_q[:, :shape_0_1] = data_tmp["states_no_intermediate"][0]["q"][:, :-1]
        datas_q[:, shape_0_1:] = data_tmp["states_no_intermediate"][1]["q"]

        # qdot
        datas_qdot = np.zeros(datas_shape)
        datas_qdot[:, :shape_0_1] = data_tmp["states_no_intermediate"][0]["qdot"][:, :-1]
        datas_qdot[:, shape_0_1:] = data_tmp["states_no_intermediate"][1]["qdot"]

        # Time
        datas_time = np.zeros(datas_shape[1])
        if data_tmp["states_no_intermediate"][0]["q"].shape[1] == data_tmp["states"][0]["q"].shape[1]:
            step = 1
        else:
            step = 5
        datas_time[:shape_0_1] = data_tmp["time"][0][:-1:step]
        datas_time[shape_0_1:] = data_tmp["time"][1][::step]

        tau_shape = (
            data_tmp["controls"][0]["tau"].shape[0],
            data_tmp["controls"][0]["tau"].shape[1] - 1 + data_tmp["controls"][1]["tau"].shape[1],
        )
        datas_tau = np.zeros((tau_shape[0], tau_shape[1]))
        datas_tau[:, : data_tmp["controls"][0]["tau"].shape[1] - 1] = data_tmp["controls"][0]["tau"][:, :-1]
        datas_tau[:, data_tmp["controls"][0]["tau"].shape[1] - 1 :] = data_tmp["controls"][1]["tau"]

        return np.asarray(datas_q), np.asarray(datas_qdot), np.asarray(datas_time), np.asarray(datas_tau)


def RK4(t, f, y0: np.ndarray, u):
    """
    Runge-Kutta 4th order method

    Parameters
    ----------
    t : array_like
        Time steps.
    f : Callable
        Function to be integrated in the form f(t, y, *args).
    y0 : np.ndarray
        Initial conditions of states.
    u : np.ndarray
        Controls for all the `t` time steps.

    Returns
    -------
    y : array_like
        States for each time step.

    """
    n = len(t)
    y = np.zeros((len(y0), n))
    y[:, 0] = y0
    for it in range(n - 1):
        h = t[it + 1] - t[it]
        y_it = np.squeeze(y[:, it])
        k1 = f(t[it], y_it, u)
        k2 = f(t[it] + h / 2.0, y_it + k1 * h / 2.0, u)
        k3 = f(t[it] + h / 2.0, y_it + k2 * h / 2.0, u)
        k4 = f(t[it] + h, y_it + k3 * h, u)
        y[:, it + 1] = y_it + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


if __name__ == "__main__":
    fig, axs = plt.subplots(3, 5)

    biorbd_casadi_model = biorbd_casadi.Model(Models.ACROBAT.value)
    biorbd_model = biorbd.Model(Models.ACROBAT.value)

    q, qdot, time, tau = get_created_data_from_pickle(
        "/home/mickaelbegon/Documents/Stage_Amandine/energetic_somersault/examples/20m_RK4_0317"
    )

    # Reintegration with RK4
    def dynamics_for_RK4(t, y, u):
        """
        Dynamics of the system for the RK4 method.

        Parameters
        ----------
        t : float
            Time.
        y : np.ndarray
            States.
        u : np.ndarray
            Controls.
        """
        q = y[:15]
        qdot = y[15:]
        return np.concatenate((qdot, biorbd_model.ForwardDynamics(q, qdot, u).to_array()))

    qRK4 = np.zeros(q.shape)
    for i in range(len(time) - 1):
        discretized_time = np.linspace(time[i], time[i + 1], 5)
        qRK4[:, i] = RK4(
            discretized_time,
            dynamics_for_RK4,
            np.concatenate((q[:, i], qdot[:, i])),
            np.concatenate((np.zeros(6), tau[:, i]), axis=0),
        )[:15, -1]

    # Reintegration with RK45
    q_RK45 = np.zeros(q.shape)
    q_RK45[:, 0] = q[:, 0]
    yi = np.concatenate((q[:, 0], qdot[:, 0]))
    for i in range(len(time) - 1):
        ui = np.concatenate((np.zeros(6), tau[:, i]), axis=0)

        def dynamics_for_RK45(t, y):
            """
            Dynamics of the system for the RK45 method.

            Parameters
            ----------
            t : float
                Time.
            y : np.ndarray
                States.
            """
            q = y[:15]
            qdot = y[15:]
            return np.concatenate((qdot, biorbd_model.ForwardDynamics(q, qdot, ui).to_array()))

        y_next = scipy.integrate.RK45(dynamics_for_RK45, time[i], yi, time[i + 1])
        while y_next.t < time[i + 1]:
            y_next.step()
        q_RK45[:, i + 1] = y_next.y[:15]
        yi = y_next.y

    # Plot
    for i in range(3):
        for j in range(5):
            axs[i, j].plot(q[5 * i + j, :], label=f"original")
            axs[i, j].plot(q_RK45[5 * i + j, :], label=f"RK45")
            axs[i, j].plot(qRK4[5 * i + j, :], label=f"RK4")
            axs[i, j].set_title(f"q{5 * i + j + 1}")
            axs[i, j].set_xlabel("Time step")

    for initial_guess_approximation in [
        InitialGuessApproximation.SEMI_IMPLICIT_EULER,
        InitialGuessApproximation.EXPLICIT_EULER,
        InitialGuessApproximation.CURRENT,
        InitialGuessApproximation.LAGRANGIAN,
        InitialGuessApproximation.CUSTOM,
    ]:
        print(initial_guess_approximation)
        vi = VariationalIntegrator(
            biorbd_casadi_model=biorbd_casadi_model,
            nb_steps=229,
            time=time[229],
            q_init=q[:, 0][:, np.newaxis],
            q_dot_init=qdot[:, 0][:, np.newaxis],
            controls=np.concatenate((np.zeros((6, len(time))), tau), axis=0)[:, :229],
            discrete_approximation=QuadratureRule.TRAPEZOIDAL,
            initial_guess_approximation=initial_guess_approximation,
            initial_guess_custom=q[:, :229],
        )

        q_vi, *_ = vi.integrate()

        for time_step in range(3):
            for j in range(5):
                axs[time_step, j].plot(q_vi[5 * time_step + j, :], label=f"{initial_guess_approximation.value}")

    fig.suptitle("Comparison of different initial guess approximations for the 20m somersault")
    # Create a new axis for the legend
    legend_ax = fig.add_subplot(111)

    # Hide the new axis
    legend_ax.set_frame_on(False)
    legend_ax.axis("off")

    # Create the legend outside the subplots
    legend = legend_ax.legend(*axs[0, 0].get_legend_handles_labels(), loc="center", ncol=2, bbox_to_anchor=(0.5, -0.15))

    # Adjust the position of the legend
    plt.subplots_adjust(top=0.9, bottom=0.17, left=0.06, right=0.94, hspace=0.4, wspace=0.2)

    plt.show()
