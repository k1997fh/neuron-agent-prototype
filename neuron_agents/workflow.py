"""Typed spike messages, independent agent state, and clocked routing."""

from collections import defaultdict
from dataclasses import dataclass, field
from math import exp, isclose, isfinite

from .model import IzhikevichNeuron, Parameters


@dataclass(frozen=True)
class SpikeEvent:
    source: str
    target: str
    emitted_ms: float
    delivered_ms: float
    weight: float


@dataclass(frozen=True)
class Sample:
    time_ms: float
    agent: str
    v: float
    u: float
    external_current: float
    synaptic_current: float
    fired: bool


@dataclass(frozen=True)
class Config:
    duration_ms: float = 300.0
    dt_ms: float = 0.1
    delay_ms: float = 1.0
    tau_ms: float = 5.0
    current_a: float = 10.0
    current_b: float = 0.0
    stimulus_start_ms: float = 10.0
    stimulus_end_ms: float = 250.0
    weight_ab: float = 20.0
    weight_ba: float = 5.0
    parameters_a: Parameters = field(default_factory=Parameters)
    parameters_b: Parameters = field(default_factory=Parameters)

    def __post_init__(self):
        values = (self.duration_ms, self.dt_ms, self.delay_ms, self.tau_ms,
                  self.current_a, self.current_b, self.stimulus_start_ms,
                  self.stimulus_end_ms, self.weight_ab, self.weight_ba)
        if not all(isfinite(x) for x in values):
            raise ValueError("Simulation settings must be finite")
        if not 0 < self.dt_ms <= 0.1 or self.duration_ms <= 0 or self.tau_ms <= 0:
            raise ValueError("Require 0 < dt_ms <= 0.1, duration_ms > 0, tau_ms > 0")
        if self.delay_ms < self.dt_ms:
            raise ValueError("Delay must be at least one timestep")
        for value in (self.duration_ms, self.delay_ms):
            if not isclose(value / self.dt_ms, round(value / self.dt_ms), abs_tol=1e-9, rel_tol=0):
                raise ValueError("Duration and delay must be integer multiples of dt_ms")
        if not 0 <= self.stimulus_start_ms <= self.stimulus_end_ms:
            raise ValueError("Require 0 <= stimulus_start_ms <= stimulus_end_ms")


class NeuronAgent:
    def __init__(self, name: str, parameters: Parameters):
        self.name = name
        self.neuron = IzhikevichNeuron(parameters)
        self.synaptic_current = 0.0

    def receive(self, event: SpikeEvent):
        if event.target != self.name:
            raise ValueError("Spike addressed to a different agent")
        self.synaptic_current += event.weight

    def advance(self, time_ms: float, external: float, dt_ms: float, tau_ms: float) -> Sample:
        current = self.synaptic_current
        fired = self.neuron.step(external + current, dt_ms)
        sample = Sample(time_ms, self.name, self.neuron.v, self.neuron.u,
                        external, current, fired)
        self.synaptic_current *= exp(-dt_ms / tau_ms)
        return sample


@dataclass
class Result:
    samples: list[Sample]
    messages: list[SpikeEvent]


class NeuronWorkflow:
    """Exactly two computational agents; coordinator is ordinary Python."""

    def __init__(self, config: Config = Config()):
        self.config = config

    def run(self) -> Result:
        # Fresh state on every run, including pending messages.
        c = self.config
        agents = (NeuronAgent("A", c.parameters_a), NeuronAgent("B", c.parameters_b))
        pending = defaultdict(list)
        result = Result([], [])
        delay_steps = round(c.delay_ms / c.dt_ms)
        for tick in range(round(c.duration_ms / c.dt_ms)):
            start_ms, end_ms = tick * c.dt_ms, (tick + 1) * c.dt_ms
            for event in pending.pop(tick, []):
                agents[0 if event.target == "A" else 1].receive(event)
            active = c.stimulus_start_ms <= start_ms < c.stimulus_end_ms
            # Both agents advance before any new output is scheduled.
            samples = [agent.advance(end_ms, drive if active else 0.0, c.dt_ms, c.tau_ms)
                       for agent, drive in zip(agents, (c.current_a, c.current_b))]
            result.samples.extend(samples)
            for sample, target, weight in zip(samples, ("B", "A"), (c.weight_ab, c.weight_ba)):
                if sample.fired:
                    delivery_tick = tick + 1 + delay_steps
                    event = SpikeEvent(sample.agent, target, end_ms,
                                       delivery_tick * c.dt_ms, weight)
                    pending[delivery_tick].append(event)
                    result.messages.append(event)
        return result
