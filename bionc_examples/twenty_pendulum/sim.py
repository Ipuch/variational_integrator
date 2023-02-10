from bionc import (
    BiomechanicalModel,
    SegmentNaturalCoordinates,
    NaturalCoordinates,
    NaturalVelocities,
    SegmentNaturalVelocities,
)
from casadi import Function
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import time as t
from .utils import RK4
from varint import NaturalVariationalIntegrator as VariationalIntegrator
from varint import QuadratureRule


class StandardSim:
    def __init__(
        self,
        biomodel,
        final_time,
        dt,
        RK: str = "RK4",
    ):
        self.biomodel = biomodel
        self.final_time = final_time
        self.dt = dt

        tuple_of_Q = [
            SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, -i, 0], rd=[0, -i - 1, 0], w=[0, 0, 1])
            for i in range(0, self.biomodel.nb_segments)
        ]
        Q = NaturalCoordinates.from_qi(tuple(tuple_of_Q))
        self.all_q_t0 = Q

        tuple_of_Qdot = [
            SegmentNaturalVelocities.from_components(udot=[0, 0, 0], rpdot=[0, 0, 0], rddot=[0, 0, 0], wdot=[0, 0, 0])
            for i in range(0, self.biomodel.nb_segments)
        ]
        Qdot = NaturalVelocities.from_qdoti(tuple(tuple_of_Qdot))
        self.all_qdot_t0 = Qdot

        results_col = [
            "time",
            "time_steps",
            "states",
            "q",
            "qdot",
            "lagrange_multipliers",
            "Etot",
            "Ekin",
            "Epot",
            "Phi_r",
            "Phi_j",
            "Phi_rdot",
            "Phi_jdot",
            "Phi_rddot",
            "Phi_jddot",
        ]
        self.results = {d: None for d in results_col}

        self.run_discrete_ode_solver(
            RK=RK,
        )
        self.compute_q_and_qdot()
        self.compute_energy()
        self.compute_lagrange_multipliers()
        self.compute_constraints()
        self.compute_constraint_derivative()

    def run_discrete_ode_solver(
        self,
        RK: str = "RK4",
    ):
        tic0 = t.time()
        self.results["time_steps"], self.results["states"], self.results["dynamics"] = self.drop_the_pendulum(
            model=self.biomodel,
            Q_init=self.all_q_t0,
            Qdot_init=self.all_qdot_t0,
            t_final=self.final_time,
            steps_per_second=self.final_time / self.dt,
            integrator=RK,
        )

        tic_end = t.time()
        self.results["time"] = tic_end - tic0
        print(f"RK4 took {self.results['time']} seconds")

    @staticmethod
    def drop_the_pendulum(
        model: BiomechanicalModel,
        Q_init: NaturalCoordinates,
        Qdot_init: NaturalVelocities,
        t_final: float = 2,
        steps_per_second: int = 50,
        integrator: str = "RK4",
    ):
        """
        This function simulates the dynamics of a natural segment falling from 0m during 2s

        Parameters
        ----------
        model : BiomechanicalModel
            The model to be simulated
        Q_init : SegmentNaturalCoordinates
            The initial natural coordinates of the segment
        Qdot_init : SegmentNaturalVelocities
            The initial natural velocities of the segment
        t_final : float, optional
            The final time of the simulation, by default 2
        steps_per_second : int, optional
            The number of steps per second, by default 50
        integrator : str
            The integrator to be used, by default "RK4"

        Returns
        -------
        tuple:
            time_steps : np.ndarray
                The time steps of the simulation
            all_states : np.ndarray
                The states of the system at each time step X = [Q, Qdot]
            dynamics : Callable
                The dynamics of the system, f(t, X) = [Xdot, lambdas]
        """

        print("Evaluate Rigid Body Constraints:")
        print(model.rigid_body_constraints(Q_init))
        print("Evaluate Rigid Body Constraints Jacobian Derivative:")
        print(model.rigid_body_constraint_jacobian_derivative(Qdot_init))

        if (model.rigid_body_constraints(Q_init) > 1e-6).any():
            print(model.rigid_body_constraints(Q_init))
            raise ValueError(
                "The segment natural coordinates don't satisfy the rigid body constraint, at initial conditions."
            )

        t_final = t_final  # [s]
        steps_per_second = steps_per_second
        time_steps = np.linspace(0, t_final, int(steps_per_second + 1))

        # initial conditions, x0 = [Qi, Qidot]
        states_0 = np.concatenate((Q_init.to_array(), Qdot_init.to_array()), axis=0)

        # Create the forward dynamics function Callable (f(t, x) -> xdot)
        def dynamics(t, states):
            idx_coordinates = slice(0, model.nb_Q)
            idx_velocities = slice(model.nb_Q, model.nb_Q + model.nb_Qdot)

            qddot, lambdas = model.forward_dynamics(
                NaturalCoordinates(states[idx_coordinates]),
                NaturalVelocities(states[idx_velocities]),
            )
            return np.concatenate((states[idx_velocities], qddot.to_array()), axis=0), lambdas

        # Solve the Initial Value Problem (IVP) for each time step
        if integrator == "RK4":
            all_states = RK4(t=time_steps, f=lambda t, states: dynamics(t, states)[0], y0=states_0)
        else:
            all_states = np.zeros((len(states_0), len(time_steps)))
            all_states[:, 0] = states_0
            for i in range(len(time_steps) - 1):
                sol = solve_ivp(
                    fun=lambda t, states: dynamics(t, states)[0],
                    t_span=(time_steps[i], time_steps[i + 1]),
                    y0=all_states[:, i],
                    method="RK45",
                )
                all_states[:, i + 1] = sol.y[:, -1]

        return time_steps, all_states, dynamics

    def compute_energy(self):
        """
        Compute the energy of the system at each time step
        """
        self.results["Etot"] = np.zeros(len(self.results["time_steps"]))
        self.results["Ekin"] = np.zeros(len(self.results["time_steps"]))
        self.results["Epot"] = np.zeros(len(self.results["time_steps"]))

        for i in range(len(self.results["time_steps"])):
            self.results["Etot"][i] = self.biomodel.energy(
                NaturalCoordinates(self.results["states"][0 : self.biomodel.nb_Q, i]),
                NaturalVelocities(self.results["states"][self.biomodel.nb_Q :, i]),
            )
            self.results["Ekin"][i] = self.biomodel.kinetic_energy(
                NaturalVelocities(self.results["states"][self.biomodel.nb_Q :, i]),
            )
            self.results["Epot"][i] = self.biomodel.potential_energy(
                NaturalCoordinates(self.results["states"][0 : self.biomodel.nb_Q, i]),
            )

    def plot_energy(self):
        """
        Plot the energy of the system at each time step
        """
        plt.figure()
        plt.plot(self.results["time_steps"], self.results["Etot"], label="Etot")
        plt.plot(self.results["time_steps"], self.results["Ekin"], label="Ekin")
        plt.plot(self.results["time_steps"], self.results["Epot"], label="Epot")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Energy [J]")
        plt.title("Energy of the system")
        plt.show()

    def compute_lagrange_multipliers(self):
        """
        Compute the lagrange_multipliers of the system at each time step
        """
        self.results["lagrange_multipliers"] = np.zeros(
            (self.biomodel.nb_holonomic_constraints, len(self.results["time_steps"]))
        )
        for i in range(len(self.results["time_steps"])):
            self.results["lagrange_multipliers"][:, i] = self.results["dynamics"](
                self.results["time_steps"][i], self.results["states"][:, i]
            )[1].squeeze()

    def compute_q_and_qdot(self):
        """
        Compute the q and qdot of the system at each time step
        """
        self.results["q"] = np.zeros((self.biomodel.nb_Q, len(self.results["time_steps"])))
        self.results["qdot"] = np.zeros((self.biomodel.nb_Qdot, len(self.results["time_steps"])))
        for i in range(len(self.results["time_steps"])):
            self.results["q"][:, i] = self.results["states"][0 : self.biomodel.nb_Q, i]
            self.results["qdot"][:, i] = self.results["states"][self.biomodel.nb_Q :, i]

    def compute_constraints(self):
        """
        Compute the constraints of the system at each time step
        """
        self.results["Phi_r"] = np.zeros((self.biomodel.nb_rigid_body_constraints, len(self.results["time_steps"])))
        self.results["Phi_j"] = np.zeros((self.biomodel.nb_joint_constraints, len(self.results["time_steps"])))
        for i in range(len(self.results["time_steps"])):
            self.results["Phi_r"][:, i] = self.biomodel.rigid_body_constraints(
                NaturalCoordinates(self.results["q"][:, i])
            )
            self.results["Phi_j"][:, i] = self.biomodel.joint_constraints(NaturalCoordinates(self.results["q"][:, i]))

    def compute_constraint_derivative(self):
        self.results["Phi_r_dot"] = np.zeros((self.biomodel.nb_rigid_body_constraints, len(self.results["time_steps"])))
        self.results["Phi_j_dot"] = np.zeros((self.biomodel.nb_joint_constraints, len(self.results["time_steps"])))
        for i in range(len(self.results["time_steps"])):
            self.results["Phi_r_dot"][:, i] = self.biomodel.rigid_body_constraints_derivative(
                NaturalCoordinates(self.results["q"][:, i]), NaturalVelocities(self.results["qdot"][:, i])
            )
            # self.results["Phi_j_dot"][:, i] = self.biomodel.joints_constraints_derivative(
            #     NaturalCoordinates(self.results["q"][:, i]), NaturalVelocities(self.results["qdot"][:, i])
            # )

    def plot_Q(self):
        q = NaturalCoordinates(self.results["q"])
        fig, axs = plt.subplots(4, self.biomodel.nb_segments)
        for i in range(self.biomodel.nb_segments):
            for iu in range(3):
                axs[0, i].plot(
                    self.results["time_steps"], q.vector(i).u[iu, :], label="Variational Integrator", linestyle="-"
                )

            for irp in range(3):
                axs[1, i].plot(
                    self.results["time_steps"], q.vector(i).rp[irp, :], label="Variational Integrator", linestyle="-"
                )

            for ird in range(3):
                axs[2, i].plot(
                    self.results["time_steps"], q.vector(i).rd[ird, :], label="Variational Integrator", linestyle="-"
                )

            for iw in range(3):
                axs[3, i].plot(
                    self.results["time_steps"], q.vector(i).w[iw, :], label="Variational Integrator", linestyle="-"
                )

        axs[0, 0].set_ylabel("u")
        axs[1, 0].set_ylabel("rp")
        axs[2, 0].set_ylabel("rd")
        axs[3, 0].set_ylabel("w")

        # for i in range(biomodel.nb_rigid_body_constraints):
        #     axs[0, 1].plot(
        #         time_step, lambdas_vi[i, :], label="Variational Integrator", color="red", linestyle="-"
        #     )
        # for i in range(biomodel.nb_rigid_body_constraints,
        #                biomodel.nb_rigid_body_constraints + biomodel.nb_joint_constraints):
        #     axs[1, 1].plot(
        #         time_step, lambdas_vi[i, :], label="Variational Integrator", color="green",
        #         linestyle="-"
        #     )

        # axs[0, 0].legend()


class VariationalSim:
    def __init__(
        self,
        biomodel,
        final_time,
        dt,
        all_q_t0=None,
        all_q_t1=None,
    ):
        self.biomodel = biomodel
        self.final_time = final_time
        self.dt = dt

        if all_q_t0 is None:
            tuple_of_Q = [
                SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, -i, 0], rd=[0, -i - 1, 0], w=[0, 0, 1])
                for i in range(0, self.biomodel.nb_segments)
            ]
            Q = NaturalCoordinates.from_qi(tuple(tuple_of_Q))
            self.all_q_t0 = Q
        else:
            self.all_q_t0 = all_q_t0

        if all_q_t1 is None:
            tuple_of_Q = [
                SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, -i, 0], rd=[0, -i - 1, 0], w=[0, 0, 1])
                for i in range(0, self.biomodel.nb_segments)
            ]
            Q = NaturalCoordinates.from_qi(tuple(tuple_of_Q))
            self.all_q_t1 = Q
        else:
            self.all_q_t1 = all_q_t1

        results_col = [
            "time",
            "time_steps",
            "states",
            "q",
            "qdot",
            "lagrange_multipliers",
            "Etot",
            "Ekin",
            "Epot",
            "Phi_r",
            "Phi_j",
            "Phi_rdot",
            "Phi_jdot",
            "Phi_rddot",
            "Phi_jddot",
        ]
        self.results = {d: None for d in results_col}
        self.results["time_steps"] = np.arange(0, self.final_time, self.dt)

        vi = VariationalIntegrator(
            biomodel=self.biomodel,
            time_step=self.dt,
            time=self.final_time,
            discrete_lagrangian_approximation=QuadratureRule.TRAPEZOIDAL,
            q_init=np.concatenate((self.all_q_t0[:, np.newaxis], self.all_q_t1[:, np.newaxis]), axis=1),
        )
        tic0 = t.time()
        self.results["q"], self.results["lagrange_multipliers"] = vi.integrate()
        tic_end = t.time()
        print("Variational Integrator took {} seconds".format(tic_end - tic0))
        self.results["time"] = tic_end - tic0

        self.discrete_energy_function = Function(
            "energy",
            [vi.q_cur, vi.q_next],
            [
                vi.discrete_function(function=self.biomodel.energy, q1=vi.q_cur, q2=vi.q_next),
            ],
        ).expand()

        self.discrete_kinetic_energy_function = Function(
            "kinetic_energy",
            [vi.q_cur, vi.q_next],
            [
                vi.discrete_function(
                    function=lambda q, qdot: self.biomodel.kinetic_energy(qdot), q1=vi.q_cur, q2=vi.q_next
                ),
            ],
        ).expand()

        self.discrete_potential_energy_function = Function(
            "potential_energy",
            [vi.q_cur, vi.q_next],
            [
                vi.discrete_function(
                    function=lambda q, qdot: self.biomodel.potential_energy(q), q1=vi.q_cur, q2=vi.q_next
                ),
            ],
        ).expand()

        self.discrete_rigid_body_constraint_function = Function(
            "Phi_r",
            [vi.q_cur, vi.q_next],
            [
                vi.discrete_function(
                    function=lambda q, qdot: self.biomodel.rigid_body_constraints(q), q1=vi.q_cur, q2=vi.q_next
                ),
            ],
        ).expand()
        self.discrete_joint_constraint_function = Function(
            "Phi_j",
            [vi.q_cur, vi.q_next],
            [
                vi.discrete_function(
                    function=lambda q, qdot: self.biomodel.joint_constraints(q), q1=vi.q_cur, q2=vi.q_next
                ),
            ],
        ).expand()

        self.compute_energy()
        self.compute_constraints()
        # self.compute_constraint_derivative()

    def compute_energy(self):
        """
        Compute the energy of the system at each time step
        """
        self.results["Etot"] = np.zeros(len(self.results["time_steps"]))
        self.results["Ekin"] = np.zeros(len(self.results["time_steps"]))
        self.results["Epot"] = np.zeros(len(self.results["time_steps"]))

        for i in range(len(self.results["time_steps"])):
            self.results["Etot"][i] = self.discrete_energy_function(
                self.results["q"][:, i], self.results["q"][:, i + 1]
            )
            self.results["Ekin"][i] = self.discrete_kinetic_energy_function(
                self.results["q"][:, i], self.results["q"][:, i + 1]
            )
            self.results["Epot"][i] = self.discrete_potential_energy_function(
                self.results["q"][:, i], self.results["q"][:, i + 1]
            )

    def plot_energy(self):
        """
        Plot the energy of the system at each time step
        """
        plt.figure()
        plt.plot(self.results["time_steps"], self.results["Etot"], label="Etot")
        plt.plot(self.results["time_steps"], self.results["Ekin"], label="Ekin")
        plt.plot(self.results["time_steps"], self.results["Epot"], label="Epot")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Energy [J]")
        plt.title("Energy of the system")
        plt.show()

    def compute_constraints(self):
        """
        Compute the constraints of the system at each time step
        """
        self.results["Phi_r"] = np.zeros((self.biomodel.nb_rigid_body_constraints, len(self.results["time_steps"])))
        self.results["Phi_j"] = np.zeros((self.biomodel.nb_joint_constraints, len(self.results["time_steps"])))
        for i in range(len(self.results["time_steps"])):
            self.results["Phi_r"][:, i] = (
                self.discrete_rigid_body_constraint_function(
                    NaturalCoordinates(self.results["q"][:, i]), NaturalCoordinates(self.results["q"][:, i + 1])
                )
                .toarray()
                .squeeze()
            )
            self.results["Phi_j"][:, i] = (
                self.discrete_joint_constraint_function(
                    NaturalCoordinates(self.results["q"][:, i]), NaturalCoordinates(self.results["q"][:, i + 1])
                )
                .toarray()
                .squeeze()
            )

    def plot_Q(self):
        q = NaturalCoordinates(self.results["q"])
        fig, axs = plt.subplots(4, self.biomodel.nb_segments)
        for i in range(self.biomodel.nb_segments):
            for iu in range(3):
                axs[0, i].plot(
                    self.results["time_steps"], q.vector(i).u[iu, :], label="Variational Integrator", linestyle="-"
                )

            for irp in range(3):
                axs[1, i].plot(
                    self.results["time_steps"], q.vector(i).rp[irp, :], label="Variational Integrator", linestyle="-"
                )

            for ird in range(3):
                axs[2, i].plot(
                    self.results["time_steps"], q.vector(i).rd[ird, :], label="Variational Integrator", linestyle="-"
                )

            for iw in range(3):
                axs[3, i].plot(
                    self.results["time_steps"], q.vector(i).w[iw, :], label="Variational Integrator", linestyle="-"
                )

        axs[0, 0].set_ylabel("u")
        axs[1, 0].set_ylabel("rp")
        axs[2, 0].set_ylabel("rd")
        axs[3, 0].set_ylabel("w")

        # for i in range(biomodel.nb_rigid_body_constraints):
        #     axs[0, 1].plot(
        #         time_step, lambdas_vi[i, :], label="Variational Integrator", color="red", linestyle="-"
        #     )
        # for i in range(biomodel.nb_rigid_body_constraints,
        #                biomodel.nb_rigid_body_constraints + biomodel.nb_joint_constraints):
        #     axs[1, 1].plot(
        #         time_step, lambdas_vi[i, :], label="Variational Integrator", color="green",
        #         linestyle="-"
        #     )

        # axs[0, 0].legend()
