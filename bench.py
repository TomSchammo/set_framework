from argparse import ArgumentParser
from pathlib import Path
import copy
import numpy as np
import tensorflow as tf

from set_keras import SET_MLP_CIFAR10

from strategies.random_set import RandomSET
from strategies.neuron_centrality import NeuronCentralitySET
from strategies.fisher_diagonal_set import FisherDiagonalSET
from strategies.ema import NeuronEMASET


parser = ArgumentParser(add_help=False)
parser.add_argument('--target_accuracy', type=float, required=False)
parser.add_argument('--max_epochs', type=int, required=False)
parser.add_argument('--accuracy', action="store_true", required=False)
parser.add_argument('--time', action="store_true", required=False)
parser.add_argument('--help', action="store_true", required=False)
parser.add_argument('--seed', type=int, required=False)


def print_help():
    print("""
Usage: python bench.py --max_epochs <N> (--accuracy | --time) [--target_accuracy <X>] [--seed <N>] [--help]

Options:
  --max_epochs <N>         Required.
  --accuracy               Benchmark accuracy after fixed epochs.
  --time                   Benchmark epochs to reach target accuracy (requires --target_accuracy).
  --target_accuracy <X>    Required with --time.
  --seed <N>               Set numpy / tf seeds for reproducibility.
  --help                   Show help.
""")


def set_seeds(seed: int | None):
    if seed is None:
        return
    seed = int(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("Skipping registering GPU devices...")
        return
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=10240)]
    )


def make_baseline_state(max_epochs: int, use_skip: bool, seed: int | None):
    """
    Build a fresh model and return its full initial state (masks + weights).
    IMPORTANT: no direct access to skip_02; use model.get_state().
    """
    set_seeds(seed)
    dummy_strategy = RandomSET()  # only used to construct the model
    m = SET_MLP_CIFAR10(strategy=dummy_strategy, max_epochs=max_epochs, use_skip=use_skip)
    return m.get_state()


def main():
    args = parser.parse_args()

    if args.help:
        print_help()
        return

    assert args.max_epochs, "Max epochs has to be provided!"
    assert args.accuracy != args.time, "Expected either --accuracy or --time"

    if args.time:
        assert args.target_accuracy is not None, "You forgot --target_accuracy"
        run_type = "time"
        print(f"\n\n---- Training until convergence to {args.target_accuracy} accuracy (max {args.max_epochs} epochs) ----\n\n")
    else:
        run_type = "accuracy"
        if args.target_accuracy is not None:
            print("\033[33m[Warning]: target_accuracy is ignored when benchmarking fixed-epoch accuracy.\033[0m")
        print(f"\n\n---- Training for {args.max_epochs} epochs ----\n\n")

    target_accuracy = 1.0 if args.accuracy else float(args.target_accuracy)
    max_epochs = int(args.max_epochs)

    setup_gpu()
    set_seeds(args.seed)

    # ---- strategies you want ----
    strategies = [
        RandomSET(),                    # ONLY no-skip 
        NeuronCentralitySET(),           # run both
        NeuronEMASET(),                  # run both
        FisherDiagonalSET(zeta=0.3, beta=0.9, eps=1e-8, seed=args.seed),  # run both
    ]

    
    def is_skipable(strategy) -> bool:
        return strategy.__class__.__name__ != "RandomSET"

    baseline_noskip = make_baseline_state(max_epochs=max_epochs, use_skip=False, seed=args.seed)
    baseline_skip   = make_baseline_state(max_epochs=max_epochs, use_skip=True,  seed=args.seed)

    model_cls = SET_MLP_CIFAR10
    save_dir = Path(f"{model_cls.__name__.lower()}_results_{run_type}")
    save_dir.mkdir(exist_ok=True)

    results: list[tuple[str, str, int | float]] = []

    for strat in strategies:
        strat_name = strat.__class__.__name__

        runs = [("no_skip", False)]
        if is_skipable(strat):
            runs.append(("skip", True))

        for tag, use_skip in runs:
            
            init_state = copy.deepcopy(baseline_skip if use_skip else baseline_noskip)

            set_seeds(args.seed)
            model = model_cls(strategy=strat, max_epochs=max_epochs, use_skip=use_skip, init_state=init_state)

            epoch_count, best_accuracy = model.train(target_accuracy=target_accuracy)

            
            filename = f"{strat_name.lower()}__{tag}.csv"
            np.savetxt(save_dir / filename, np.asarray(model.accuracies_per_epoch), delimiter=';')

            if args.time:
                print(f"[{strat_name} | {tag}] took {epoch_count} epochs until convergence")
                results.append((strat_name, tag, epoch_count))
            else:
                print(f"[{strat_name} | {tag}] best {best_accuracy*100:.2f}% after {max_epochs} epochs")
                results.append((strat_name, tag, best_accuracy * 100.0))

    #pretty print
    print("\n" + "=" * 80)
    if args.time:
        print(f"{'Strategy':<30} | {'Arch':<8} | {'Epochs':>7}")
    else:
        print(f"{'Strategy':<30} | {'Arch':<8} | {'Accuracy (%)':>11}")
    print("-" * 80)

    for name, tag, metric in results:
        if isinstance(metric, float):
            print(f"{name:<30} | {tag:<8} | {metric:11.2f}")
        else:
            print(f"{name:<30} | {tag:<8} | {metric:>7}")

    print("=" * 80)


if __name__ == "__main__":
    main()
