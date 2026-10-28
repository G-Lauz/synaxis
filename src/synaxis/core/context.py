from __future__ import annotations

import contextlib
import contextvars
import uuid
from collections.abc import Generator, Mapping
from typing import Any

from .signals import Signal, _ComputedSignal

_TraceState = tuple[list[Signal[Any]], set[uuid.UUID]]

_TRACE: contextvars.ContextVar[_TraceState | None] = contextvars.ContextVar("synaxis_trace", default=None)
_VALUES: contextvars.ContextVar[Mapping[uuid.UUID, Any] | None] = contextvars.ContextVar("synaxis_values", default=None)

_SAMPLES: contextvars.ContextVar[dict[uuid.UUID, Any] | None] = contextvars.ContextVar("synaxis_samples", default=None)
_SAMPLING: contextvars.ContextVar[frozenset[uuid.UUID]] = contextvars.ContextVar(
    "synaxis_sampling", default=frozenset()
)


def is_signal_in_context() -> bool:
    return _TRACE.get() is not None or _VALUES.get() is not None


def read_signal(signal: Signal[Any]) -> Any:
    trace_state = _TRACE.get()

    if trace_state is not None:
        ordered, seen = trace_state
        if signal.id not in seen:
            ordered.append(signal)
            seen.add(signal.id)

    values = _VALUES.get()

    if values is not None:
        try:
            return values[signal.id]
        except KeyError as exception:
            raise RuntimeError(
                f"signal {signal.name} was not available in  the compiled execution order"
            ) from exception

    if trace_state is not None:
        return _sample(
            signal
        )  # sample the signal value from computed signals (cached) or default value (nominal or zero)

    return signal.get_value()  # the default value (nominal or zero)


def _sample(signal: Signal[Any]) -> Any:
    samples = _SAMPLES.get()
    equation = signal.equation if isinstance(signal, _ComputedSignal) else None

    if equation is None or samples is None:
        return signal.get_value()  # the default value (nominal or zero)

    if signal.id in samples:
        return samples[signal.id]

    if signal.id in _SAMPLING.get():
        return signal.get_value()  # cycle: fall back; the graph check reports the algebraic loop later

    sampling_token = _SAMPLING.set(_SAMPLING.get() | {signal.id})
    trace_token = _TRACE.set(([], set()))  # nested frame: inner reads are not the outer equation's inputs

    try:
        value = equation.function()
    except Exception as exception:
        raise RuntimeError(
            f"failed to sample {signal.name} (id={signal.id.hex[:5]}) (read by the equation being traced)"
        ) from exception
    finally:
        _TRACE.reset(trace_token)
        _SAMPLING.reset(sampling_token)

    samples[signal.id] = value
    return value


@contextlib.contextmanager
def sample_cache() -> Generator[None]:
    token = _SAMPLES.set({})

    try:
        yield
    finally:
        _SAMPLES.reset(token)


@contextlib.contextmanager
def trace() -> Generator[list[Signal[Any]]]:
    signals: list[Signal[Any]] = []
    token = _TRACE.set((signals, set()))

    try:
        yield signals
    finally:
        _TRACE.reset(token)


@contextlib.contextmanager
def evaluate(values: Mapping[uuid.UUID, Any]) -> Generator[None]:
    token = _VALUES.set(values)

    try:
        yield
    finally:
        _VALUES.reset(token)
