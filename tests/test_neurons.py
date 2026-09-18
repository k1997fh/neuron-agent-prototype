import unittest
from dataclasses import replace
from math import exp

from neuron_agents.model import IzhikevichNeuron, Parameters
from neuron_agents.workflow import Config, NeuronAgent, NeuronWorkflow, SpikeEvent


class ModelTests(unittest.TestCase):
    def test_equations_and_reset(self):
        neuron = IzhikevichNeuron()
        self.assertFalse(neuron.step(10, 0.1))
        self.assertAlmostEqual(neuron.v, -64.3)
        self.assertAlmostEqual(neuron.u, -13)
        neuron.v, neuron.u = 29, -13
        self.assertTrue(neuron.step(10, 0.1))
        self.assertEqual(neuron.v, -65)
        self.assertAlmostEqual(neuron.u, -13 + 0.1 * 0.02 * (0.2 * 29 + 13) + 8)

    def test_quiet_without_input_and_repetitive_firing_with_input(self):
        for current, expected in ((0, False), (10, True)):
            neuron = IzhikevichNeuron()
            spikes = sum(neuron.step(current, 0.1) for _ in range(3000))
            self.assertEqual(spikes > 2, expected)
            if current == 0:
                self.assertEqual(spikes, 0)

    def test_incoming_signals_sum_and_decay(self):
        agent = NeuronAgent("B", Parameters())
        agent.receive(SpikeEvent("A", "B", 0, 1, 20))
        agent.receive(SpikeEvent("A", "B", 0, 1, -5))
        sample = agent.advance(1.1, 0, 0.1, 5)
        self.assertEqual(sample.synaptic_current, 15)
        self.assertAlmostEqual(agent.synaptic_current, 15 * exp(-0.1 / 5))
        with self.assertRaises(ValueError):
            agent.receive(SpikeEvent("B", "A", 0, 1, 1))


class WorkflowTests(unittest.TestCase):
    def test_causal_bidirectional_communication(self):
        c = Config()
        result = NeuronWorkflow(c).run()
        self.assertEqual({(m.source, m.target) for m in result.messages}, {("A", "B"), ("B", "A")})
        # Independently reconstruct every received current from event timestamps.
        for agent in ("A", "B"):
            current = 0.0
            incoming = {}
            for event in result.messages:
                if event.target == agent:
                    tick = round(event.delivered_ms / c.dt_ms)
                    incoming[tick] = incoming.get(tick, 0) + event.weight
                    self.assertAlmostEqual(event.delivered_ms - event.emitted_ms, c.delay_ms)
            for tick, sample in enumerate(s for s in result.samples if s.agent == agent):
                current += incoming.get(tick, 0)
                self.assertAlmostEqual(sample.synaptic_current, current)
                current *= exp(-c.dt_ms / c.tau_ms)
        disconnected = NeuronWorkflow(replace(c, weight_ab=0, weight_ba=0)).run()
        self.assertFalse(any(s.fired for s in disconnected.samples if s.agent == "B"))
        one_way = NeuronWorkflow(replace(c, weight_ba=0)).run()
        self.assertNotEqual([s.v for s in result.samples if s.agent == "A"],
                            [s.v for s in one_way.samples if s.agent == "A"])

    def test_repeatable_and_bounded(self):
        workflow = NeuronWorkflow(Config(duration_ms=20))
        first = workflow.run()
        self.assertEqual(first, workflow.run())
        self.assertEqual(len(first.samples), 400)
        self.assertEqual(first.samples[-1].time_ms, 20)

    def test_timestep_refinement_preserves_single_neuron_spike_count(self):
        counts = []
        for dt in (0.1, 0.05, 0.025):
            result = NeuronWorkflow(Config(dt_ms=dt, weight_ab=0, weight_ba=0)).run()
            counts.append(sum(s.fired for s in result.samples if s.agent == "A"))
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_invalid_settings(self):
        for kwargs in ({"dt_ms": 0}, {"dt_ms": 1}, {"duration_ms": -1},
                       {"delay_ms": 0}, {"delay_ms": 0.15}, {"duration_ms": 0.15},
                       {"tau_ms": 0}, {"weight_ab": float("nan")},
                       {"stimulus_start_ms": -1}, {"stimulus_end_ms": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
