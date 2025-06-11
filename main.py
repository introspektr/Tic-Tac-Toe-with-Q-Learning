import argparse
import sys
from train import train
from eval import evaluate

def main():
    parser = argparse.ArgumentParser(
        description="Tic-Tac-Toe Q-Learning Agent Control Panel"
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Sub-command: train
    train_parser = subparsers.add_parser("train", help="Train a Q-learning agent")
    train_parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")
    train_parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    train_parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    train_parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate")
    train_parser.add_argument("--epsilon-decay", type=float, default=0.999, help="Epsilon decay rate")
    train_parser.add_argument("--min-epsilon", type=float, default=0.01, help="Minimum exploration rate")
    train_parser.add_argument("--q-init", type=float, default=0.5, help="Initial Q-value")
    train_parser.add_argument("--save-as", type=str, default="q_table.pkl", help="Filename to save the trained model")

    # Sub-command: eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument("--model", type=str, default="q_table.pkl", help="Model file to load")
    eval_parser.add_argument("--games", type=int, default=1000, help="Number of games to evaluate")
    eval_parser.add_argument("--verbose", action="store_true", help="Print sample game logs")
    eval_parser.add_argument("--samples", type=int, default=5, help="Number of sample games to show")
    eval_parser.add_argument("--visualize", action="store_true", help="Visualize Q-values for sample games")

    args = parser.parse_args()

    if args.command == "train":
        print("🚀 Starting training...")
        agent, stats = train(
            num_episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            min_epsilon=args.min_epsilon,
            init_q=args.q_init,
            stats_interval=1000  # You can make this configurable later
        )
        agent.save(args.save_as)
        print(f"✅ Training complete. Model saved as: {args.save_as}")

    elif args.command == "eval":
        try:
            print(f"🔍 Evaluating model from: {args.model}")
            evaluate(
                agent_path=args.model,
                num_games=args.games,
                verbose=args.verbose,
                sample_games=args.samples,
                visualize=args.visualize
            )
        except FileNotFoundError:
            print(f"Error: Model file '{args.model}' not found.")
            sys.exit(1)

    else:
        parser.print_help()

if __name__ == "__main__":
    main() 