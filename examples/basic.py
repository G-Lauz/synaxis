from typing import Optional

import jax
import numpy

from minilink.core import (
    DynamicSystem,
    Input,
    Output,
    Param,
    SignalLike,
    State,
    StaticSystem,
    System,
)


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

    u = Input()
    y = Output()

    def compute_outputs(self):
        return bmm(self.C, self.x) + bmm(self.D, self.u)

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

    model.pretty_print_identity()

    print(model.plant._compute_dynamics)

    # T = 10.0
    # dt = 0.01
    # n_steps = int(T / dt)
    # time = numpy.linspace(0, T, n_steps)

    # solver = Euler(dt=dt)

    # initial_states = model.states()
    # params = model.params()
    # inputs = model.inputs()

    # finale_state, trajectory = solver.rollout(
    #     compiled_model, initial_states, params, inputs, n_steps
    # )

    # # jitted
    # jitted_rolout = jax.jit(
    #     lambda states, inputs, params: solver.rollout(
    #         compiled_model, initial_states, params, inputs, n_steps
    #     )
    # )
    # jit_final_state, jit_trajectory = jitted_rolout(initial_states, inputs, params)

    # # vmap
    # def rollout_given_gain(kp):
    #     _, traj = solver.rollout(compiled_model, initial_states, kp, inputs, n_steps)
    #     return traj

    # vmapped_rollout = jax.jit(jax.vmap(rollout_given_gain))

    # gains = jax.numpy.array([0.5, 1.0, 2.0, 5.0])
    # vmap_trajectories = vmapped_rollout(gains)

    # # grad
    # def loss(kp):
    #     end_states, _ = solver.rollout(
    #         compiled_model, initial_states, kp, inputs, n_steps
    #     )
    #     return jax.numpy.sum((end_states.plant.x - model.reference) ** 2)

    # grad_loss = jax.jit(jax.grad(loss))
    # gain_grad = grad_loss(1.0)


if __name__ == "__main__":
    main()
