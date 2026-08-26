import argparse

from snake_ai.agent import Agent
from snake_ai.game import SnakeGameAI


def train(render=True, speed=40, max_games=None):
    plot_fn = None
    if render:
        from snake_ai.helper import plot as plot_fn

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(render=render, speed=speed)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            agent.update_target()

            if score > record:
                record = score
                agent.model.save()

            print(f"Game {agent.n_games}  Score {score}  Record {record}", flush=True)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            if plot_fn is not None:
                plot_fn(plot_scores, plot_mean_scores)

            if max_games is not None and agent.n_games >= max_games:
                return record, plot_mean_scores[-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Snake.")
    parser.add_argument(
        "--no-render",
        "--fast",
        dest="no_render",
        action="store_true",
        help="Train headless at maximum speed (no pygame window or live plot).",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=40,
        help="Render FPS cap when watching training (default: 40).",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Stop after this many games (default: train forever).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(render=not args.no_render, speed=args.speed, max_games=args.games)
