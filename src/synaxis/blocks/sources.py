from __future__ import annotations

from synaxis.core import Noise, Output, Param, StaticSystem


class Constant(StaticSystem):
    y = Output()
    value = Param()

    def __init__(self, name: str | None = None):
        name = name if name is not None else "constant"
        super().__init__(name=name)

    def compute_outputs(self):
        return self.value


class WhiteNoise(StaticSystem):
    y = Output()

    z = Noise()

    mean = Param()
    stddev = Param()

    def __init__(self, name: str | None = None):
        name = name if name is not None else "white_noise"
        super().__init__(name=name)

    def compute_outputs(self):
        return self.mean + self.stddev * self.z
