import jax
import matplotlib.pyplot as plt
import numpy

from synaxis.core import (
    DynamicSystem,
    Input,
    Output,
    Param,
    State,
    StateDerivative,
    StaticSystem,
    System,
)
from synaxis.diagram import to_dot
from synaxis.solvers import Euler


def bmm(matrix, vector) -> jax.Array:
    """
    Batch matrix-vector multiplication.
    """
    jnp_matrix = jax.numpy.asarray(matrix)
    jnp_vector = jax.numpy.asarray(vector)

    if jnp_matrix.ndim == 0:
        jnp_matrix = jnp_matrix[None, None]

    if jnp_vector.ndim == 0:
        jnp_vector = jnp_vector[None]

    return jax.numpy.matmul(jnp_matrix, jnp_vector[..., None])[..., 0]


class LTISystem(DynamicSystem):
    A = Param()
    B = Param()
    C = Param()
    D = Param()

    x = State()
    dx = StateDerivative()

    u = Input()
    y = Output()

    def compute_outputs(self):
        return bmm(self.C, self.x)  # + bmm(self.D, self.u)

    def compute_dynamics(self):
        return bmm(self.A, self.x) + bmm(self.B, self.u)


class ProportionalController(StaticSystem):
    def __init__(self, name=None) -> None:
        name = name if name is not None else "proportional controller"
        super().__init__(name=name)

        self.kp = Param()

        self.u = Output()
        self.observation = Input()
        self.reference = Input()

    def compute_outputs(self):
        return bmm(-self.kp, self.observation - self.reference)


class ClosedLoopModel(System):
    plant = LTISystem()

    def __init__(self, name=None) -> None:
        name = name if name is not None else "closed-loop model"
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
