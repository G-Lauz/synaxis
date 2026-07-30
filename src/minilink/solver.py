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
        if signal._uuid not in self.trajectories:
            raise KeyError(f"signal {signal.name} not found in trajectory")
        return self.trajectories[signal._uuid]


jax.tree_util.register_dataclass(ResultWrapper)


@dataclasses.dataclass(frozen=True)
class Euler:
    dt: float

    def rollout(
        self,
        system: CompiledSystem,
        *,
        n_steps: int,
    ):
        """Return final states and a trajectory that includes the initial states.

        State derivatives are paired with states by tuple position.
        """
        initial_values = system.initial_values()
        state_ids = system._signal_ids_by_kind.get(SignalKind.STATE, ())
        derivative_ids = system._signal_ids_by_kind.get(SignalKind.STATE_DERIVATIVE, ())

        # TODO: this check should happen at system compilation time
        # if len(derivative_ids) != len(state_ids):
        #     raise ValueError(
        #         f"expected {len(state_ids)} state derivatives, got {len(derivative_ids)}. "
        #         "Check that the system has the same number of state and state derivative signals."
        #     )

        x0 = {id: initial_values[id] for id in state_ids}
        fixed_values = {id: initial_values[id] for id in initial_values if id not in state_ids}

        def advance(states, _):
            results_by_id = system.evaluate({**fixed_values, **states})

            next_states = {
                state_id: jax.tree.map(
                    lambda state, derivative: state + self.dt * derivative,
                    states[state_id],
                    results_by_id[derivative_id],
                )
                for state_id, derivative_id in zip(state_ids, derivative_ids)
            }
            return next_states, results_by_id

        final_states, trajectory = jax.lax.scan(
            advance,
            x0,
            xs=None,
            length=n_steps,
        )

        final_result = system.evaluate({**fixed_values, **final_states})

        trajectories = jax.tree.map(
            lambda history, final_value: jax.numpy.concatenate(
                (history, jax.numpy.expand_dims(final_value, axis=0)), axis=0
            ),
            trajectory,
            final_result,
        )
        return ResultWrapper(trajectories=trajectories)
