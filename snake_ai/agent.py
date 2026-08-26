import random
from collections import deque

import numpy as np
import torch

from snake_ai.game import Direction, Point, BLOCK_SIZE
from snake_ai.model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
INPUT_SIZE = 27
HIDDEN_SIZE = 256
TARGET_UPDATE = 10
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY_GAMES = 2000

_CLOCKWISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
_DELTAS = {
    Direction.RIGHT: (BLOCK_SIZE, 0),
    Direction.DOWN: (0, BLOCK_SIZE),
    Direction.LEFT: (-BLOCK_SIZE, 0),
    Direction.UP: (0, -BLOCK_SIZE),
}


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = EPS_START
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, 3)
        self.target_model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.head
        rel_dirs = self._relative_dirs(game.direction)

        danger_1 = [game.is_collision(self._offset(head, d, 1)) for d in rel_dirs]
        danger_2 = [game.is_collision(self._offset(head, d, 2)) for d in rel_dirs]
        danger_3 = [game.is_collision(self._offset(head, d, 3)) for d in rel_dirs]
        free_space = [self._free_steps(game, head, d) for d in rel_dirs]
        reach_tail = [float(self._can_reach_tail(game, self._offset(head, d, 1))) for d in rel_dirs]

        food_dx = (game.food.x - head.x) / game.w
        food_dy = (game.food.y - head.y) / game.h
        tail = game.snake[-1]
        tail_dx = (tail.x - head.x) / game.w
        tail_dy = (tail.y - head.y) / game.h

        state = [
            *danger_1,
            *danger_2,
            *danger_3,
            *free_space,
            game.direction == Direction.LEFT,
            game.direction == Direction.RIGHT,
            game.direction == Direction.UP,
            game.direction == Direction.DOWN,
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,
            food_dx,
            food_dy,
            tail_dx,
            tail_dy,
            *reach_tail,
        ]
        return np.array(state, dtype=float)

    def _relative_dirs(self, direction):
        idx = _CLOCKWISE.index(direction)
        straight = _CLOCKWISE[idx]
        right = _CLOCKWISE[(idx + 1) % 4]
        left = _CLOCKWISE[(idx - 1) % 4]
        return straight, right, left

    def _offset(self, head, direction, steps):
        dx, dy = _DELTAS[direction]
        return Point(head.x + dx * steps, head.y + dy * steps)

    def _free_steps(self, game, head, direction):
        limit = max(game.w, game.h) // BLOCK_SIZE
        for steps in range(1, limit + 1):
            if game.is_collision(self._offset(head, direction, steps)):
                return (steps - 1) / limit
        return 1.0

    def _can_reach_tail(self, game, start):
        if game.is_collision(start):
            return False
        tail = game.snake[-1]
        if start == tail:
            return True
        blocked = set(game.snake[1:-1])
        queue = deque([start])
        seen = {start}
        while queue:
            pt = queue.popleft()
            if pt == tail:
                return True
            for dx, dy in _DELTAS.values():
                nxt = Point(pt.x + dx, pt.y + dy)
                if nxt in seen:
                    continue
                if nxt.x < 0 or nxt.x >= game.w or nxt.y < 0 or nxt.y >= game.h:
                    continue
                if nxt in blocked:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def update_target(self):
        if self.n_games % TARGET_UPDATE == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, explore=True):
        t = min(1.0, self.n_games / EPS_DECAY_GAMES)
        self.epsilon = EPS_END + (EPS_START - EPS_END) * (1.0 - t) if explore else 0.0
        final_move = [0, 0, 0]
        if explore and random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
