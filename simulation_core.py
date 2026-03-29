"""
Turn-based traffic simulation core powered by Mesa.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Iterable

import mesa

from config_loader import AppConfig, CarPriority


A_LEFT = "left"
A_UP = "up"
A_RIGHT = "right"
A_WAIT = "wait"
MOVE_ACTIONS = (A_LEFT, A_UP, A_RIGHT)
ALL_ACTIONS = (A_LEFT, A_UP, A_RIGHT, A_WAIT)


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    turns: int
    reached: int
    total: int
    avg_reward: float


class CarAgent(mesa.Agent):
    """Discrete-turn car agent with Q-learning + priority fallback."""

    def __init__(
        self,
        model: "TrafficModel",
        car_id: int,
        spawn_turn: int,
        start_x: int,
        destination_x: int,
    ) -> None:
        super().__init__(model)
        self.car_id = car_id

        self.spawn_turn = spawn_turn
        self.start_x = start_x
        self.destination_x = destination_x

        self.x = start_x
        self.y = self.model.size
        self.started = False
        self.reached = False

        self.accumulated_reward = 0.0
        self.blocked_count = 0
        self.wait_count = 0
        self.turns_alive = 0

        self.q_table: defaultdict[tuple[tuple, str], float] = defaultdict(float)
        self.epsilon = self.model.config.q_learning.epsilon_start

    @property
    def is_active(self) -> bool:
        return self.started and not self.reached

    @property
    def cell(self) -> tuple[int, int]:
        return self.x, self.y

    def reset_for_epoch(self, spawn_turn: int, start_x: int, destination_x: int) -> None:
        self.spawn_turn = spawn_turn
        self.start_x = start_x
        self.destination_x = destination_x
        self.x = start_x
        self.y = self.model.size
        self.started = False
        self.reached = False
        self.accumulated_reward = 0.0
        self.blocked_count = 0
        self.wait_count = 0
        self.turns_alive = 0

    def step(self) -> None:
        if self.reached:
            return

        if not self.started:
            if self.model.turn_count < self.spawn_turn:
                return
            self.started = True
            self.model.add_to_occupancy(self.cell)

        self.turns_alive += 1
        state = self._get_state()
        valid_actions = self.model.get_valid_actions(self)
        action = self._choose_action(state, valid_actions)

        next_pos, moved = self.model.try_move(self, action)
        reward = self.model.config.rewards.time_penalty

        if not moved:
            if action == A_WAIT:
                self.wait_count += 1
                reward += self.model.config.rewards.wait_penalty
            else:
                self.blocked_count += 1
                reward += self.model.config.rewards.blocked_penalty
        else:
            self.model.move_agent(self, next_pos)
            if self.y == 0 and self.x == self.destination_x:
                self.reached = True
                self.model.remove_from_occupancy(next_pos)
                reward += self.model.config.rewards.reach_destination

        self.accumulated_reward += reward
        next_state = self._get_state()
        next_valid_actions = self.model.get_valid_actions(self)
        self._update_q(state, action, reward, next_state, next_valid_actions)
        self._decay_epsilon()

    def _get_state(self) -> tuple:
        left_info = self.model.peek_cell(self, A_LEFT)
        up_info = self.model.peek_cell(self, A_UP)
        right_info = self.model.peek_cell(self, A_RIGHT)
        return (
            self.x,
            self.y,
            self.destination_x,
            left_info,
            up_info,
            right_info,
        )

    def _choose_action(self, state: tuple, valid_actions: list[str]) -> str:
        if not valid_actions:
            return A_WAIT

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_values = {action: self.q_table[(state, action)] for action in valid_actions}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        if len(best_actions) == 1:
            return best_actions[0]

        fallback = self.model.priority_fallback(self, valid_actions)
        if fallback is not None:
            return fallback
        return random.choice(best_actions)

    def _update_q(
        self,
        state: tuple,
        action: str,
        reward: float,
        next_state: tuple,
        next_valid_actions: Iterable[str],
    ) -> None:
        alpha = self.model.config.q_learning.learning_rate
        gamma = self.model.config.q_learning.discount_factor

        old_q = self.q_table[(state, action)]
        if self.reached:
            next_max = 0.0
        else:
            candidates = [self.q_table[(next_state, a)] for a in next_valid_actions]
            next_max = max(candidates) if candidates else 0.0

        self.q_table[(state, action)] = old_q + alpha * (reward + gamma * next_max - old_q)

    def _decay_epsilon(self) -> None:
        cfg = self.model.config.q_learning
        self.epsilon = max(cfg.epsilon_min, self.epsilon * cfg.epsilon_decay)


class TrafficModel(mesa.Model):
    """Environment-managed traffic simulation with epoch support."""

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.size = config.grid.size
        self.turn_count = 0
        self.epoch = 1
        self.epoch_done = False

        self.last_epoch_summary: EpochSummary | None = None
        self.epoch_summaries: list[EpochSummary] = []

        self.traffic_lights: dict[tuple[int, int], str] = {}
        self._light_timer = 0
        self._occupancy: dict[tuple[int, int], int] = {}

        self._spawn_schedule: list[tuple[int, int, int]] = self._build_spawn_schedule()
        self._build_lights()
        self._create_cars()
        self._rebuild_occupancy()

    def _create_cars(self) -> None:
        for car_id, (spawn_turn, start_x, dest_x) in enumerate(self._spawn_schedule):
            CarAgent(self, car_id, spawn_turn, start_x, dest_x)

    def _build_spawn_schedule(self) -> list[tuple[int, int, int]]:
        schedule: list[tuple[int, int, int]] = []
        no_cars_left = self.config.cars.total_cars
        turn = 1
        start_points = self.config.starting_point_x
        spawn_rates = self.config.cars.spawn_rates

        while no_cars_left > 0:
            for idx, start_x in enumerate(start_points):
                if no_cars_left <= 0:
                    break
                rate = spawn_rates[idx]
                for _ in range(rate):
                    if no_cars_left <= 0:
                        break
                    dest_x = random.choice(self.config.destination_x)
                    schedule.append((turn, start_x, dest_x))
                    no_cars_left -= 1
                    if not self.config.cars.multiple_cars_per_turn:
                        turn += 1
            if self.config.cars.multiple_cars_per_turn:
                turn += 1
        return schedule

    def _build_lights(self) -> None:
        self.traffic_lights.clear()
        next_state = self.config.traffic_lights.initial_state
        for y in range(2, self.size - 1, 2):
            for x in range(0, self.size, 2):
                self.traffic_lights[(x, y)] = next_state
                next_state = "Red" if next_state == "Green" else "Green"
        self._light_timer = 0

    def step(self) -> None:
        if self.epoch_done:
            return

        self.turn_count += 1
        self._update_traffic_lights()
        self.agents.shuffle_do("step")

        if self.all_reached() or self.turn_count >= self.config.simulation.max_turns_per_epoch:
            self._finish_epoch()

    def _update_traffic_lights(self) -> None:
        self._light_timer += 1
        if self._light_timer < self.config.traffic_lights.switch_interval:
            return
        self._light_timer = 0
        for pos, state in list(self.traffic_lights.items()):
            self.traffic_lights[pos] = "Red" if state == "Green" else "Green"

    def _finish_epoch(self) -> None:
        self.epoch_done = True
        self.last_epoch_summary = EpochSummary(
            epoch=self.epoch,
            turns=self.turn_count,
            reached=self.get_reached_count(),
            total=self.config.cars.total_cars,
            avg_reward=self.get_avg_reward(),
        )
        self.epoch_summaries.append(self.last_epoch_summary)

    def reset_epoch(self) -> None:
        self.epoch += 1
        self.turn_count = 0
        self.epoch_done = False

        self._spawn_schedule = self._build_spawn_schedule()
        self._build_lights()
        self._occupancy.clear()

        for agent, schedule in zip(self.agents, self._spawn_schedule):
            spawn_turn, start_x, dest_x = schedule
            agent.reset_for_epoch(spawn_turn, start_x, dest_x)

    def all_reached(self) -> bool:
        return all(agent.reached for agent in self.agents)

    def get_reached_count(self) -> int:
        return sum(1 for agent in self.agents if agent.reached)

    def get_avg_reward(self) -> float:
        rewards = [agent.accumulated_reward for agent in self.agents]
        return sum(rewards) / len(rewards) if rewards else 0.0

    def is_unavailable(self, x: int, y: int) -> bool:
        if y >= self.size:
            return False
        return x % 2 == 1 and y % 2 == 1

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y <= self.size

    def get_light(self, x: int, y: int) -> str | None:
        return self.traffic_lights.get((x, y))

    def can_enter_cell(self, agent: CarAgent, x: int, y: int, action: str) -> bool:
        if not self.in_bounds(x, y):
            return False

        if y == self.size:
            return action == A_WAIT and (x, y) == agent.cell

        if self.is_unavailable(x, y):
            return False

        if not self._light_allows_move(action, x, y):
            return False

        if self.get_occupancy((x, y)) >= self.config.grid.max_cars_per_cell:
            return False
        return True

    def _light_allows_move(self, action: str, x: int, y: int) -> bool:
        light_state = self.get_light(x, y)
        if light_state is None:
            return True
        if action == A_UP:
            return light_state == "Green"
        if action in (A_LEFT, A_RIGHT):
            return light_state == "Red"
        return True

    def get_valid_actions(self, agent: CarAgent) -> list[str]:
        if not agent.started or agent.reached:
            return [A_WAIT]

        actions: list[str] = [A_WAIT]
        for action in MOVE_ACTIONS:
            nx, ny = self._next_pos(agent.cell, action)
            if ny == self.size:
                continue
            if self.can_enter_cell(agent, nx, ny, action):
                actions.append(action)
        return actions

    def _next_pos(self, pos: tuple[int, int], action: str) -> tuple[int, int]:
        x, y = pos
        if action == A_LEFT:
            return x - 1, y
        if action == A_RIGHT:
            return x + 1, y
        if action == A_UP:
            return x, y - 1
        return x, y

    def try_move(self, agent: CarAgent, action: str) -> tuple[tuple[int, int], bool]:
        if action == A_WAIT:
            return agent.cell, False

        nx, ny = self._next_pos(agent.cell, action)
        if not self.can_enter_cell(agent, nx, ny, action):
            return agent.cell, False
        return (nx, ny), True

    def move_agent(self, agent: CarAgent, new_pos: tuple[int, int]) -> None:
        old_pos = agent.cell
        self.remove_from_occupancy(old_pos)
        agent.x, agent.y = new_pos
        self.add_to_occupancy(new_pos)

    def _rebuild_occupancy(self) -> None:
        self._occupancy.clear()
        for agent in self.agents:
            if agent.started and not agent.reached:
                self.add_to_occupancy(agent.cell)

    def add_to_occupancy(self, pos: tuple[int, int]) -> None:
        self._occupancy[pos] = self._occupancy.get(pos, 0) + 1

    def remove_from_occupancy(self, pos: tuple[int, int]) -> None:
        if pos not in self._occupancy:
            return
        self._occupancy[pos] -= 1
        if self._occupancy[pos] <= 0:
            del self._occupancy[pos]

    def get_occupancy(self, pos: tuple[int, int]) -> int:
        return self._occupancy.get(pos, 0)

    def peek_cell(self, agent: CarAgent, action: str) -> tuple[int, int]:
        nx, ny = self._next_pos(agent.cell, action)
        if not self.in_bounds(nx, ny) or ny == self.size or self.is_unavailable(nx, ny):
            return -1, 0
        light = self.get_light(nx, ny)
        light_code = 0 if light is None else (1 if light == "Green" else 2)
        return min(self.get_occupancy((nx, ny)), self.config.grid.max_cars_per_cell), light_code

    def priority_fallback(self, agent: CarAgent, valid_actions: list[str]) -> str | None:
        moves = [a for a in valid_actions if a in MOVE_ACTIONS]
        if not moves:
            return A_WAIT if A_WAIT in valid_actions else None

        priority = self.config.policy.priority
        order = self._directional_preference(agent)

        if priority in (CarPriority.NO_PRIORITY, CarPriority.GREEN_LIGHT):
            for action in order:
                if action in moves:
                    return action
            return moves[0]

        if priority == CarPriority.LOWER_TRAFFIC:
            scored: list[tuple[int, int, str]] = []
            for action in moves:
                nx, ny = self._next_pos(agent.cell, action)
                occupancy = self.get_occupancy((nx, ny))
                direction_bias = order.index(action) if action in order else len(order)
                scored.append((occupancy, direction_bias, action))
            scored.sort()
            return scored[0][2]

        return moves[0]

    def _directional_preference(self, agent: CarAgent) -> list[str]:
        dx = agent.x - agent.destination_x
        if dx > 0:
            return [A_LEFT, A_UP, A_RIGHT]
        if dx < 0:
            return [A_RIGHT, A_UP, A_LEFT]
        return [A_UP, A_LEFT, A_RIGHT]
