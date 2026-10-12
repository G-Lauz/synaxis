from __future__ import annotations

import math

from synaxis.core import Noise, Output, Param, System, Time


class Constant(System):
    default_name = "constant"

    value = Param()

    @Output
    def y(self):
        return self.value


class Sine(System):
    default_name = "sine"

    time = Time()
    amplitude = Param()
    frequency = Param()

    @Output
    def y(self):
        # TODO: take into account the backend, e.g. numpy, jax, etc.
        return self.amplitude * math.sin(2 * math.pi * self.frequency * self.time)


class WhiteNoise(System):
    default_name = "white_noise"

    z = Noise()

    mean = Param()
    stddev = Param()

    @Output
    def y(self):
        return self.mean + self.stddev * self.z
