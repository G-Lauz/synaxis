from __future__ import annotations

import contextlib
import contextvars
import uuid
from collections.abc import Generator, Mapping
from typing import Any

from .signals import Signal

_TraceState = tuple[list[Signal[Any]], set[uuid.UUID]]

_TRACE: contextvars.ContextVar[_TraceState | None] = contextvars.ContextVar("synaxis_trace", default=None)
_VALUES: contextvars.ContextVar[Mapping[uuid.UUID, Any] | None] = contextvars.ContextVar("synaxis_values", default=None)


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

    if values is None:
        return signal.get_value()  # the default value (nominal or zero)

    try:
        return values[signal.id]
    except KeyError as exception:
        raise RuntimeError(f"signal {signal.name} was not available in  the compiled execution order") from exception


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
