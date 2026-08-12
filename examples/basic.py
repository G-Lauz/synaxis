import jax
import matplotlib.pyplot as plt
import numpy

from synaxis.controller import ProportionalController
from synaxis.core import Input, System
from synaxis.diagram import to_dot
from synaxis.dynamics import LTISystem
from synaxis.solvers import Euler


class ClosedLoopModel(System):
    plant = LTISystem(name="plant", direct_feedthrough=False)

    def __init__(self, name=None) -> None:
        name = name if name is not None else "closed_loop_model"
        super().__init__(name=name)

        self.reference = Input()

        self.controller = ProportionalController()

        self.connect(self.reference, self.controller.reference)
        self.connect(self.controller.u, self.plant.u)
        self.connect(self.plant.y, self.controller.observation)


def main():
    model = ClosedLoopModel()
    compiled_model = model.compile()

    # model.pretty_print()

    # dot_text = to_dot(compiled_model.graph, explicit_connection=True)
    # with open("closed_loop_model.dot", "w") as f:
    #     f.write(dot_text)

    T = 10.0
    dt = 0.01
    n_steps = int(T / dt)
    time = numpy.linspace(0, T, n_steps + 1)

    solver = Euler(dt=dt)

    compiled_model[model.plant.x] = jax.numpy.array([0.0])  # initial state

    compiled_model[model.plant.A] = -1.0
    compiled_model[model.plant.B] = 1.0
    compiled_model[model.plant.C] = 1.0
    compiled_model[model.plant.D] = 0.0

    compiled_model[model.controller.kp] = 2.0

    reference = 5.0
    compiled_model[model.reference] = reference

    trajectory = solver.rollout(compiled_model, n_steps=n_steps)

    print("Final state:", trajectory[model.plant.x][-1])
    print("Trajectory shape:", trajectory[model.plant.x].shape)

    # jitted
    jitted_rolout = jax.jit(lambda: solver.rollout(compiled_model, n_steps=n_steps))
    jit_trajectory = jitted_rolout()

    # vmap
    @compiled_model.vary
    def rollout_given_gain(kp):
        compiled_model[model.controller.kp] = kp
        return solver.rollout(compiled_model, n_steps=n_steps)

    vmapped_rollout = jax.jit(jax.vmap(rollout_given_gain))

    gains = jax.numpy.array([0.5, 1.0, 2.0, 5.0])
    vmap_trajectories = vmapped_rollout(gains)

    print(vmap_trajectories[model.plant.x].shape)

    # grad
    @compiled_model.vary
    def loss(kp):
        compiled_model[model.controller.kp] = kp
        traj = solver.rollout(compiled_model, n_steps=n_steps)
        return jax.numpy.sum((traj[model.plant.x][-1] - reference) ** 2)

    grad_loss = jax.jit(jax.vmap(jax.grad(loss)))
    gains_grads = grad_loss(gains)

    print(f"Gradients wrt. gains: {gains_grads}")

    plt.figure()
    plt.plot(time, trajectory[model.plant.x], label="plant.x")
    plt.plot(time, jit_trajectory[model.plant.x], label="jit plant.x", linestyle="--")
    for gain, traj in zip(gains, vmap_trajectories[model.plant.x]):
        plt.plot(time, traj, label=f"kp={gain}")
    plt.xlabel("Time [s]")
    plt.ylabel("State value")
    plt.title("Closed-loop system state trajectory")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
