from enum import Enum
from collections import namedtuple
import random

import pygame
import numpy as np

pygame.init()

Point = namedtuple("Point", "x, y")

BLOCK_SIZE = 20
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
HEAD = (0, 180, 90)
HEAD_INNER = (80, 240, 140)
EYE = (20, 20, 20)
BLACK = (0, 0, 0)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeGameAI:
    def __init__(self, w=640, h=480, render=True, speed=40):
        self.w = w
        self.h = h
        self.render = render
        self.speed = speed
        self.display = None
        self.clock = None
        self.font = None

        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 25)

        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.visited_since_food = set()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.visited_since_food = set()
        else:
            self.snake.pop()
            if self.head in self.visited_since_food:
                reward -= 1.0 / len(self.snake)
            else:
                self.visited_since_food.add(self.head)

        if self.render:
            self._update_ui()
            self.clock.tick(self.speed)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake[1:]:
            pygame.draw.rect(
                self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )
        self._draw_head()
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _draw_head(self):
        pt = self.head
        pygame.draw.rect(
            self.display,
            HEAD,
            pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
            border_radius=7,
        )
        pygame.draw.rect(
            self.display,
            HEAD_INNER,
            pygame.Rect(pt.x + 3, pt.y + 3, 14, 14),
            border_radius=5,
        )

        if self.direction == Direction.RIGHT:
            eyes = ((pt.x + 14, pt.y + 5), (pt.x + 14, pt.y + 15))
        elif self.direction == Direction.LEFT:
            eyes = ((pt.x + 6, pt.y + 5), (pt.x + 6, pt.y + 15))
        elif self.direction == Direction.UP:
            eyes = ((pt.x + 5, pt.y + 6), (pt.x + 15, pt.y + 6))
        else:
            eyes = ((pt.x + 5, pt.y + 14), (pt.x + 15, pt.y + 14))

        for x, y in eyes:
            pygame.draw.circle(self.display, WHITE, (int(x), int(y)), 3)
            pygame.draw.circle(self.display, EYE, (int(x), int(y)), 1)

    def _move(self, action):
        # action is one-hot [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
