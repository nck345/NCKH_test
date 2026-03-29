"""
Load and validate runtime configuration from config.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any


class CarPriority(str, Enum):
    NO_PRIORITY = "NoPriority"
    GREEN_LIGHT = "GreenLight"
    LOWER_TRAFFIC = "LowerTraffic"


class TrafficLightIntelligence(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


@dataclass(frozen=True)
class SimulationConfig:
    max_turns_per_epoch: int
    ms_per_turn: int
    manual_epoch_advance: bool


@dataclass(frozen=True)
class GridConfig:
    size: int
    no_cars_per_cell: int
    max_cars_per_cell: int


@dataclass(frozen=True)
class CarsConfig:
    total_cars: int
    spawn_rates: tuple[int, ...]
    multiple_cars_per_turn: bool


@dataclass(frozen=True)
class TrafficLightsConfig:
    switch_interval: int
    initial_state: str
    intelligence: TrafficLightIntelligence


@dataclass(frozen=True)
class PolicyConfig:
    priority: CarPriority


@dataclass(frozen=True)
class QLearningConfig:
    learning_rate: float
    discount_factor: float
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float


@dataclass(frozen=True)
class RewardsConfig:
    reach_destination: float
    time_penalty: float
    wait_penalty: float
    blocked_penalty: float


@dataclass(frozen=True)
class UIConfig:
    window_width: int
    window_height: int
    title: str


@dataclass(frozen=True)
class AppConfig:
    simulation: SimulationConfig
    grid: GridConfig
    cars: CarsConfig
    traffic_lights: TrafficLightsConfig
    policy: PolicyConfig
    q_learning: QLearningConfig
    rewards: RewardsConfig
    ui: UIConfig

    @property
    def num_starting_points(self) -> int:
        return (self.grid.size + 1) // 2

    @property
    def starting_point_x(self) -> tuple[int, ...]:
        return tuple(i * 2 for i in range(self.num_starting_points))

    @property
    def destination_x(self) -> tuple[int, ...]:
        return self.starting_point_x


def _require_dict(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config.{key} must be an object.")
    return value


def _require_int(section: dict[str, Any], key: str, minimum: int = 0) -> int:
    value = section.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}.")
    return value


def _require_float(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number.")
    return float(value)


def _require_bool(section: dict[str, Any], key: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _require_str(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def load_config(path: str | Path = "config.json") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Root JSON must be an object.")

    sim = _require_dict(raw, "simulation")
    grid = _require_dict(raw, "grid")
    cars = _require_dict(raw, "cars")
    lights = _require_dict(raw, "traffic_lights")
    policy = _require_dict(raw, "policy")
    ql = _require_dict(raw, "q_learning")
    rewards = _require_dict(raw, "rewards")
    ui = _require_dict(raw, "ui")

    simulation = SimulationConfig(
        max_turns_per_epoch=_require_int(sim, "max_turns_per_epoch", minimum=1),
        ms_per_turn=_require_int(sim, "ms_per_turn", minimum=1),
        manual_epoch_advance=_require_bool(sim, "manual_epoch_advance"),
    )

    grid_cfg = GridConfig(
        size=_require_int(grid, "size", minimum=3),
        no_cars_per_cell=_require_int(grid, "no_cars_per_cell", minimum=1),
        max_cars_per_cell=_require_int(grid, "max_cars_per_cell", minimum=1),
    )
    if grid_cfg.size % 2 == 0:
        raise ValueError("grid.size must be odd (7, 9, ...).")
    if grid_cfg.max_cars_per_cell > grid_cfg.no_cars_per_cell ** 2:
        raise ValueError("grid.max_cars_per_cell cannot exceed no_cars_per_cell^2.")

    spawn_rates = cars.get("spawn_rates")
    if not isinstance(spawn_rates, list) or not spawn_rates:
        raise ValueError("cars.spawn_rates must be a non-empty array of integers.")
    if not all(isinstance(v, int) and v >= 0 for v in spawn_rates):
        raise ValueError("cars.spawn_rates must contain integers >= 0.")

    cars_cfg = CarsConfig(
        total_cars=_require_int(cars, "total_cars", minimum=1),
        spawn_rates=tuple(spawn_rates),
        multiple_cars_per_turn=_require_bool(cars, "multiple_cars_per_turn"),
    )

    expected_spawn_len = (grid_cfg.size + 1) // 2
    if len(cars_cfg.spawn_rates) != expected_spawn_len:
        raise ValueError(
            f"cars.spawn_rates must have {expected_spawn_len} values for grid.size={grid_cfg.size}."
        )

    initial_state = _require_str(lights, "initial_state")
    if initial_state not in {"Green", "Red"}:
        raise ValueError("traffic_lights.initial_state must be 'Green' or 'Red'.")

    try:
        intelligence = TrafficLightIntelligence(_require_str(lights, "intelligence"))
    except ValueError as exc:
        raise ValueError(
            "traffic_lights.intelligence must be one of: L0, L1, L2, L3."
        ) from exc

    lights_cfg = TrafficLightsConfig(
        switch_interval=_require_int(lights, "switch_interval", minimum=1),
        initial_state=initial_state,
        intelligence=intelligence,
    )

    try:
        priority = CarPriority(_require_str(policy, "priority"))
    except ValueError as exc:
        raise ValueError(
            "policy.priority must be one of: NoPriority, GreenLight, LowerTraffic."
        ) from exc
    policy_cfg = PolicyConfig(priority=priority)

    ql_cfg = QLearningConfig(
        learning_rate=_require_float(ql, "learning_rate"),
        discount_factor=_require_float(ql, "discount_factor"),
        epsilon_start=_require_float(ql, "epsilon_start"),
        epsilon_min=_require_float(ql, "epsilon_min"),
        epsilon_decay=_require_float(ql, "epsilon_decay"),
    )

    if not (0.0 <= ql_cfg.learning_rate <= 1.0):
        raise ValueError("q_learning.learning_rate must be in [0, 1].")
    if not (0.0 <= ql_cfg.discount_factor <= 1.0):
        raise ValueError("q_learning.discount_factor must be in [0, 1].")
    if not (0.0 <= ql_cfg.epsilon_min <= ql_cfg.epsilon_start <= 1.0):
        raise ValueError("q_learning epsilon values must satisfy 0 <= min <= start <= 1.")
    if not (0.0 < ql_cfg.epsilon_decay <= 1.0):
        raise ValueError("q_learning.epsilon_decay must be in (0, 1].")

    rewards_cfg = RewardsConfig(
        reach_destination=_require_float(rewards, "reach_destination"),
        time_penalty=_require_float(rewards, "time_penalty"),
        wait_penalty=_require_float(rewards, "wait_penalty"),
        blocked_penalty=_require_float(rewards, "blocked_penalty"),
    )

    ui_cfg = UIConfig(
        window_width=_require_int(ui, "window_width", minimum=320),
        window_height=_require_int(ui, "window_height", minimum=320),
        title=_require_str(ui, "title"),
    )

    return AppConfig(
        simulation=simulation,
        grid=grid_cfg,
        cars=cars_cfg,
        traffic_lights=lights_cfg,
        policy=policy_cfg,
        q_learning=ql_cfg,
        rewards=rewards_cfg,
        ui=ui_cfg,
    )
