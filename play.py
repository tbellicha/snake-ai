import os
import sys

from snake_ai.agent import Agent
from snake_ai.game import SnakeGameAI


def play():
    model_path = os.path.join("./model", "model.pth")
    if not os.path.exists(model_path):
        print("No trained model found at model/model.pth. Run python train.py first.")
        sys.exit(1)

    agent = Agent()
    agent.model.load()
    game = SnakeGameAI(render=True, speed=20)

    while True:
        state = agent.get_state(game)
        final_move = agent.get_action(state, explore=False)
        _, done, score = game.play_step(final_move)
        if done:
            print(f"Game over  Score {score}")
            game.reset()


if __name__ == "__main__":
    play()
