"""JAX-backed simulation solvers."""

import dataclasses
import uuid
from typing import Any

import jax

from .core import CompiledSystem, Noise, Signal, Time


# TODO: refactor this in cleaner pattern
@dataclasses.dataclass(frozen=True)
class ResultWrapper:
    """Signal-indexed trajectories returned by a solver."""

    trajectories: dict[uuid.UUID, Any]

    def __getitem__(self, signal: Signal):
        if signal.id not in self.trajectories:
            raise KeyError(f"signal {signal.name} not found in trajectory")
        return self.trajectories[signal.id]


jax.tree_util.register_dataclass(ResultWrapper)


@dataclasses.dataclass(frozen=True)
class Euler:
    """Explicit Euler integration backed by JAX."""

    dt: float

    def rollout(
        self,
        system: CompiledSystem,
        *,
        t0: float = 0.0,
        n_steps: int,
        rng_key: Any = None,
    ) -> ResultWrapper:
        """Return final states and a trajectory that includes the initial states.

        State and derivative pairs are provided by the compiled system.
        """
        initial_values = system.initial_values()

        state_pairs = system.state_pairs
        state_ids = {state_id for state_id, _ in state_pairs}
        noise_ids = system.signal_ids(Noise)
        time_ids = system.signal_ids(Time)

        solver_driven = state_ids | set(noise_ids) | set(time_ids)

        x0 = {state_id: initial_values[state_id] for state_id in state_ids}
        fixed_values = {
            signal_id: value for signal_id, value in initial_values.items() if signal_id not in solver_driven
        }

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        def time_values(time):
            return {time_id: time for time_id in time_ids}

        def sample_noise(key):
            keys = jax.random.split(key, len(noise_ids) + 1)
            noise_values = {
                signal_id: jax.random.normal(
                    sample_key,
                    shape=jax.numpy.shape(initial_values[signal_id]),
                )
                for signal_id, sample_key in zip(noise_ids, keys[1:])
            }
            return keys[0], noise_values

        def advance(carry, time):
            states, key = carry

            key, noise_values = sample_noise(key)

            results_by_id = system.evaluate({**fixed_values, **noise_values, **time_values(time), **states})

            next_states = {
                state_id: jax.tree.map(
                    lambda state, derivative: state + self.dt * derivative,
                    states[state_id],
                    results_by_id[derivative_id],
                )
                for state_id, derivative_id in state_pairs
            }
            return (next_states, key), results_by_id

        times = t0 + self.dt * jax.numpy.arange(n_steps)

        (final_states, rng_key), trajectory = jax.lax.scan(
            advance,
            (x0, rng_key),
            xs=times,
            length=n_steps,
        )

        _, noise_values = sample_noise(rng_key)

        final_result = system.evaluate(
            {**fixed_values, **noise_values, **time_values(t0 + self.dt * n_steps), **final_states}
        )

        trajectories = jax.tree.map(
            lambda history, final_value: jax.numpy.concatenate(
                (history, jax.numpy.expand_dims(final_value, axis=0)), axis=0
            ),
            trajectory,
            final_result,
        )
        return ResultWrapper(trajectories=trajectories)
