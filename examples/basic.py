from typing import Optional

import jax
import matplotlib.pyplot as plt
import numpy

from minilink.core import (
    DynamicSystem,
    Input,
    Output,
    Param,
    SignalLike,
    State,
    StateDerivative,
    StaticSystem,
    System,
)

from minilink.diagram import to_dot
from minilink.solver import Euler


def bmm(
    matrix: SignalLike,
    vector: SignalLike,
) -> jax.Array:
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
        return bmm(self.C, self.x)# + bmm(self.D, self.u)

    def compute_dynamics(self):
        return bmm(self.A, self.x) + bmm(self.B, self.u)


class ProportionalController(StaticSystem):
    def __init__(self, name: Optional[str] = None) -> None:
        name = name if name is not None else "proportional controller"
        super().__init__(name)

        self.kp = Param()

        self.u = Output()
        self.observation = Input()
        self.reference = Input()

    def compute_outputs(self):
        return bmm(-self.kp, self.observation - self.reference)


class ClosedLoopModel(System):
    plant = LTISystem()

    def __init__(self, name: Optional[str] = None) -> None:
        name = name if name is not None else "closed-loop model"
        super().__init__(name)

        self.reference = Input()

        self.controller = ProportionalController()

        self.connect(self.reference, self.controller.reference)
        self.connect(self.controller.u, self.plant.u)
        self.connect(self.plant.y, self.controller.observation)


def main():
    model = ClosedLoopModel()
    compiled_model = model.compile()

    # dot_text = to_dot(compiled_model.graph, explicit_connection=False)
    # with open("closed_loop_model.dot", "w") as f:
    #     f.write(dot_text)

    T = 10.0
    dt = 0.01
    n_steps = int(T / dt)
    time = numpy.linspace(0, T, n_steps + 1)

    solver = Euler(dt=dt)

    initial_states = (jax.numpy.array([0.0]),)
    params = (
        jax.numpy.array([-1.0]),  # A
        jax.numpy.array([1.0]),  # B
        jax.numpy.array([1.0]),  # C
        jax.numpy.array([0.0]),  # D
        jax.numpy.array([2.0]),  # kp
    )
    inputs = (
        jax.numpy.array([5.0]),  # reference
    )

    finale_state, trajectory = solver.rollout(
        compiled_model, initial_states, inputs, params, n_steps=n_steps
    )

    print("Final state:", finale_state)
    print("Trajectory shape:", trajectory[0].shape)

    # jitted
    jitted_rolout = jax.jit(
        lambda states, inputs, params: solver.rollout(
            compiled_model, states, inputs, params, n_steps=n_steps
        )
    )
    jit_final_state, jit_trajectory = jitted_rolout(initial_states, inputs, params)

    # vmap
    def rollout_given_gain(kp):
        params = (
            jax.numpy.array([-1.0]),  # A
            jax.numpy.array([1.0]),  # B
            jax.numpy.array([1.0]),  # C
            jax.numpy.array([0.0]),  # D
            jax.numpy.array([kp,]),  # kp
        )
        _, traj = solver.rollout(compiled_model, initial_states, inputs, params, n_steps=n_steps)
        return traj

    vmapped_rollout = jax.jit(jax.vmap(rollout_given_gain))

    gains = jax.numpy.array([0.5, 1.0, 2.0, 5.0])
    vmap_trajectories = vmapped_rollout(gains)

    print(len(vmap_trajectories), vmap_trajectories[0].shape)

    # grad
    def loss(kp):
        params = (
            jax.numpy.array([-1.0]),  # A
            jax.numpy.array([1.0]),  # B
            jax.numpy.array([1.0]),  # C
            jax.numpy.array([0.0]),  # D
            jax.numpy.array([kp,]),  # kp
        )
        end_states, _ = solver.rollout(
            compiled_model, initial_states, inputs, params, n_steps=n_steps
        )
        return jax.numpy.sum((end_states[0] - inputs[0]) ** 2)

    grad_loss = jax.jit(jax.vmap(jax.grad(loss)))
    gains_grads = grad_loss(gains)

    print(f"Gradients wrt. gains: {gains_grads}")

    plt.figure()
    plt.plot(time, trajectory[0], label="plant.x")
    for gain, traj in zip(gains, vmap_trajectories[0]):
        plt.plot(time, traj, label=f"kp={gain}")
    plt.xlabel("Time [s]")
    plt.ylabel("State value")
    plt.title("Closed-loop system state trajectory")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
