import dataclasses
import uuid
from typing import Any

import jax

from .core import CompiledSystem, Signal
from .core.signals import SignalKind


# TODO: refactor this in cleaner pattern
@dataclasses.dataclass(frozen=True)
class ResultWrapper:
    trajectories: dict[uuid.UUID, Any]

    def __getitem__(self, signal: Signal):
        if signal.id not in self.trajectories:
            raise KeyError(f"signal {signal.name} not found in trajectory")
        return self.trajectories[signal.id]


jax.tree_util.register_dataclass(ResultWrapper)


@dataclasses.dataclass(frozen=True)
class Euler:
    dt: float

    def rollout(
        self,
        system: CompiledSystem,
        *,
        n_steps: int,
        rng_key: Any = None,
    ):
        """Return final states and a trajectory that includes the initial states.

        State derivatives are paired with states by tuple position.
        """
        initial_values = system.initial_values()

        state_ids = system.get_signal_by_kind(SignalKind.STATE)
        derivative_ids = system.get_signal_by_kind(SignalKind.STATE_DERIVATIVE)
        noise_ids = system.get_signal_by_kind(SignalKind.NOISE)

        # TODO: this check should happen at system compilation time
        if len(derivative_ids) != len(state_ids):
            raise ValueError(
                f"expected {len(state_ids)} state derivatives, got {len(derivative_ids)}. "
                "Check that the system has the same number of state and state derivative signals."
            )

        x0 = {id: initial_values[id] for id in state_ids}
        fixed_values = {id: initial_values[id] for id in initial_values if id not in state_ids}

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        def advance(carry, _):
            states, key = carry

            key, rng = jax.random.split(key)
            noise_values = {id: jax.random.normal(rng, shape=jax.numpy.shape(initial_values[id])) for id in noise_ids}

            results_by_id = system.evaluate({**fixed_values, **noise_values, **states})

            next_states = {
                state_id: jax.tree.map(
                    lambda state, derivative: state + self.dt * derivative,
                    states[state_id],
                    results_by_id[derivative_id],
                )
                for state_id, derivative_id in zip(state_ids, derivative_ids)
            }
            return (next_states, rng), results_by_id

        (final_states, rng_key), trajectory = jax.lax.scan(
            advance,
            (x0, rng_key),
            xs=None,
            length=n_steps,
        )

        noise_values = {id: jax.random.normal(rng_key, shape=jax.numpy.shape(initial_values[id])) for id in noise_ids}

        final_result = system.evaluate({**fixed_values, **noise_values, **final_states})

        trajectories = jax.tree.map(
            lambda history, final_value: jax.numpy.concatenate(
                (history, jax.numpy.expand_dims(final_value, axis=0)), axis=0
            ),
            trajectory,
            final_result,
        )
        return ResultWrapper(trajectories=trajectories)
