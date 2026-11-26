"""python -m arousal {audit,train,replay}. No credentials or downloads required."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "train"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "train":
            command.add_argument("--folds", type=int, default=3)
            command.add_argument("--epochs", type=int, default=20)
            command.add_argument("--seed", type=int, default=7)
            command.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
            command.add_argument("--target-fah", type=float, default=1.)
    command = commands.add_parser("replay")
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--bdf", type=Path, required=True)
    command.add_argument("--rest", type=Path, required=True)
    command.add_argument("--onset", type=float, required=True, help="Video onset in BDF-relative seconds")
    command.add_argument("--duration", type=float, default=10.)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--lsl", action="store_true")
    command.add_argument("--speed", type=float, default=1.)
    args = parser.parse_args()
    if args.command == "audit":
        from .pipeline import audit
        print(audit(args.data, args.out))
    elif args.command == "train":
        from .pipeline import run_experiment
        if args.folds < 2 or args.epochs < 1 or args.target_fah < 0:
            parser.error("Need folds >= 2, epochs >= 1 and target-fah >= 0")
        run_experiment(args.data, args.out, args.folds, args.epochs, args.seed, args.device, args.target_fah)
    else:
        from .replay import replay
        replay(args.model, args.bdf, args.rest, args.onset, args.out, args.duration, args.lsl, args.speed)


if __name__ == "__main__":
    main()
