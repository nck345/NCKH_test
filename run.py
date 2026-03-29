"""
Application entrypoint for the turn-based tkinter traffic simulator.
"""

from __future__ import annotations

import sys

from config_loader import load_config
from simulation_core import TrafficModel
from visualizer import run_visualization


def main() -> None:
    try:
        config = load_config("config.json")
    except Exception as exc:  # noqa: BLE001 - show clear startup failure
        print(f"Failed to load config.json: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("=" * 60)
    print(" Traffic Simulator - Turn-Based (Mesa + tkinter)")
    print("=" * 60)
    print(f" Grid size: {config.grid.size}x{config.grid.size}")
    print(f" Cars: {config.cars.total_cars}")
    print(f" Epoch max turns: {config.simulation.max_turns_per_epoch}")
    print(" Keys: [Space]=Pause/Resume  [N]=Next Epoch  [Esc]=Quit")
    print("=" * 60)

    model = TrafficModel(config)
    run_visualization(model)


if __name__ == "__main__":
    main()
