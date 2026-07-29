import dataclasses

import jax

from .core import CompiledSystem, Signal
from .core.signals import SignalKind


# TODO: refactor this in cleaner pattern
@dataclasses.dataclass(frozen=True)
class ResultWrapper:
    trajectories: dict[str, jax.Array]

    def __getitem__(self, signal: Signal):
        trajectory = self.trajectories.get(signal._uuid, None)
        if trajectory is None:
            raise KeyError(f"signal {signal.name} not found in trajectory")
        return trajectory


jax.tree_util.register_dataclass(ResultWrapper)


@dataclasses.dataclass(frozen=True)
class Euler:
    dt: float

    def rollout(
        self,
        system: CompiledSystem,
        # x0=None,
        # inputs=None,
        # params=None,
        *,
        n_steps: int,
    ):
        """Return final states and a trajectory that includes the initial states.

        State derivatives are paired with states by tuple position.
        """
        x0, inputs, params = system.initial_values()

        derivatives_ids = [node.id for node in system._signal_nodes if node.kind == SignalKind.STATE_DERIVATIVE]
        if len(derivatives_ids) != len(x0):
            raise ValueError(
                f"expected {len(x0)} state derivatives, got {len(derivatives_ids)}. "
                "Check that the system has the same number of state and state derivative signals."
            )

        def advance(states, _):
            results_by_id = system.evaluate(states, inputs, params)

            derivatives = tuple(results_by_id[id] for id in derivatives_ids)

            if len(states) != len(derivatives):
                raise ValueError(f"expected {len(states)} state derivatives, got {len(derivatives)}")
            next_states = jax.tree.map(
                lambda state, derivative: state + self.dt * derivative,
                states,
                derivatives,
            )
            return next_states, results_by_id

        final_states, trajectory = jax.lax.scan(
            advance,
            x0,
            xs=None,
            length=n_steps,
        )

        final_result = system.evaluate(final_states, inputs, params)

        trajectories = jax.tree.map(
            lambda history, final_value: jax.numpy.concatenate(
                (history, jax.numpy.expand_dims(final_value, axis=0)), axis=0
            ),
            trajectory,
            final_result,
        )
        return ResultWrapper(trajectories=trajectories)
