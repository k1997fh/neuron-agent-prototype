import argparse
import csv
import json
from dataclasses import asdict, fields
from pathlib import Path

from .workflow import Config, NeuronWorkflow, Sample, SpikeEvent


def main():
    parser = argparse.ArgumentParser(description="Run two communicating Izhikevich neuron agents")
    for name in ("duration_ms", "dt_ms", "delay_ms", "tau_ms", "current_a", "current_b",
                 "stimulus_start_ms", "stimulus_end_ms", "weight_ab", "weight_ba"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(Config(), name))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        config = Config(**args)
        result = NeuronWorkflow(config).run()
    except ValueError as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)
    for name, kind, rows in (("traces.csv", Sample, result.samples),
                             ("messages.csv", SpikeEvent, result.messages)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[f.name for f in fields(kind)])
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    for agent in ("A", "B"):
        spikes = sum(s.fired for s in result.samples if s.agent == agent)
        print(f"Agent {agent}: {spikes} spikes")
    print(f"Scheduled {len(result.messages)} spike messages. Results: {output.resolve()}")


if __name__ == "__main__":
    main()
