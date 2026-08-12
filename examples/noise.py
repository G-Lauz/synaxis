import jax
import matplotlib.pyplot as plt
import numpy

from synaxis.blocks import WhiteNoise
from synaxis.controller import ProportionalController
from synaxis.core import Input, Noise, Output, System
from synaxis.dynamics import LTISystem
from synaxis.solvers import Euler
from synaxis.systems import StaticSystem


class NoisyPropController(StaticSystem):
    z = Noise()
    proportional = ProportionalController()
    u = Output()

    def compute_outputs(self):
        return self.proportional.u + self.z


class ClosedLoopSystem(System):
    plant = LTISystem(direct_feedthrough=False)
    controller = NoisyPropController()

    reference = Input()

    def __init__(self) -> None:
        super().__init__(name="closed-loop model")

        self.connect(self.reference, self.controller.proportional.reference)
        self.connect(self.controller.u, self.plant.u)
        self.connect(self.plant.y, self.controller.proportional.observation)


def main():
    noise = WhiteNoise()
    compiled_system = noise.compile()

    compiled_system[noise.mean] = 0.0
    compiled_system[noise.stddev] = 1.0

    T = 10.0
    dt = 0.01
    n_steps = int(T / dt)
    time = numpy.linspace(0, T, n_steps + 1)

    solver = Euler(dt=dt)

    result = solver.rollout(compiled_system, n_steps=n_steps)

    plt.figure()
    plt.plot(time, result[noise.y], label="White Noise")
    plt.xlabel("Time [s]")
    plt.ylabel("Noise value")
    plt.title("White Noise Signal")
    plt.legend()
    plt.show()

    model = ClosedLoopSystem()
    compiled_model = model.compile()

    compiled_model[model.plant.x] = jax.numpy.array([0.0])  # initial state

    compiled_model[model.plant.A] = -1.0
    compiled_model[model.plant.B] = 1.0
    compiled_model[model.plant.C] = 1.0
    compiled_model[model.plant.D] = 0.0

    compiled_model[model.controller.proportional.kp] = 2.0

    reference = 5.0
    compiled_model[model.reference] = reference

    n_runs = 100
    rng_keys = jax.random.split(jax.random.PRNGKey(0), n_runs)
    vmapped_rollout = jax.jit(
        jax.vmap(lambda rng_key: solver.rollout(compiled_model, n_steps=n_steps, rng_key=rng_key))
    )
    results = vmapped_rollout(rng_keys)

    control_trajectories = results[model.plant.u][..., 0]
    mean = jax.numpy.mean(control_trajectories, axis=0)
    variance = jax.numpy.var(control_trajectories, axis=0)
    std = jax.numpy.sqrt(variance)

    sample_to_show = min(10, n_runs)

    plt.figure()

    plt.plot(
        time,
        control_trajectories[:sample_to_show].T,
        color="tab:blue",
        alpha=0.15,
        linewidth=0.8,
    )
    plt.plot(time, mean, color="tab:blue", label="mean")
    plt.fill_between(time, mean - std, mean + std, color="tab:blue", alpha=0.20, label="+/- 1 std")

    plt.xlabel("Time [s]")
    plt.ylabel("Output value")
    plt.title("Closed-Loop System Responses")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
