from __future__ import annotations

from synaxis.core import Input, Output, Param, State, System


class Gain(System):
    default_name = "gain"

    u = Input()
    k = Param()

    @Output
    def y(self):
        return self.k * self.u


class Sum(System):
    default_name = "sum"

    a = Input()
    b = Input()

    @Output
    def y(self):
        return self.a + self.b


class Integrator(System):
    default_name = "integrator"

    u = Input()
    x = State()

    @x.derivative
    def dx(self):
        return self.u

    @Output
    def y(self):
        return self.x
