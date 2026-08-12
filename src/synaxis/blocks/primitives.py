from __future__ import annotations

from synaxis.core import (
    DynamicSystem,
    Input,
    Output,
    Param,
    State,
    StateDerivative,
    StaticSystem,
)


class Gain(StaticSystem):
    u = Input()
    y = Output()

    k = Param()

    def __init__(self, name: str | None = None):
        name = name if name is not None else "gain"
        super().__init__(name=name)

    def compute_outputs(self):
        return self.k * self.u


class Sum(StaticSystem):
    a = Input()
    b = Input()
    y = Output()

    def __init__(self, name: str | None = None):
        name = name if name is not None else "sum"
        super().__init__(name=name)

    def compute_outputs(self):
        return self.a + self.b


class Integrator(DynamicSystem):
    u = Input()
    y = Output()

    x = State()
    dx = StateDerivative()

    def __init__(self, name: str | None = None):
        name = name if name is not None else "integrator"
        super().__init__(name=name)

    def compute_outputs(self):
        return self.x

    def compute_dynamics(self):
        return self.u
