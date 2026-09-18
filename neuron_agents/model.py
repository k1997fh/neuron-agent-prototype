"""Izhikevich equations with simultaneous forward Euler integration."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Parameters:
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0

    def __post_init__(self):
        if not all(isfinite(x) for x in (self.a, self.b, self.c, self.d)):
            raise ValueError("Neuron parameters must be finite")
        if self.a <= 0 or self.c >= 30:
            raise ValueError("Require a > 0 and reset voltage c < 30")


class IzhikevichNeuron:
    def __init__(self, parameters: Parameters = Parameters()):
        self.parameters = parameters
        self.v = parameters.c
        self.u = parameters.b * self.v

    def step(self, current: float, dt_ms: float) -> bool:
        """Advance one step; return a spike flag and retain post-reset state."""
        if not isfinite(current) or not isfinite(dt_ms) or not 0 < dt_ms <= 0.1:
            raise ValueError("Require finite current and 0 < dt_ms <= 0.1")
        p, v, u = self.parameters, self.v, self.u
        next_v = v + dt_ms * (0.04 * v * v + 5 * v + 140 - u + current)
        next_u = u + dt_ms * p.a * (p.b * v - u)
        if not isfinite(next_v) or not isfinite(next_u):
            raise ValueError("Integration diverged; reduce input magnitude or dt_ms")
        fired = next_v >= 30.0
        self.v = p.c if fired else next_v
        self.u = next_u + p.d if fired else next_u
        return fired
