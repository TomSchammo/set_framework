from strategies.random_set import RandomSET
from strategies.neuron_centrality import NeuronCentralitySET
from set_keras import SET_MLP_CIFAR10

from argparse import ArgumentParser
from strategies.ema import NeuronEMASet
from strategies.fisher_diagonal_skip_set import FisherDiagonalSkipSET

import tensorflow as tf

parser = ArgumentParser(add_help=False)
parser.add_argument('--target_accuracy', type=float, required=False)
parser.add_argument('--max_epochs', type=int, required=False)
parser.add_argument('--accuracy', action="store_true", required=False)
parser.add_argument('--time', action="store_true", required=False)
parser.add_argument('--help', action="store_true", required=False)


def print_help():
    print("""
Usage: python bench.py --max_epochs <N> (--accuracy | --time) [--target_accuracy <X>] [--help]

Options:
  --max_epochs <N>         Required. Number of epochs to train for (if --accuracy) or max allowed for convergence (if --time).
  --accuracy               Benchmark accuracy achieved after a fixed amount of epochs.
  --time                   Benchmark how many epochs are needed to achieve a target accuracy (must provide --target_accuracy).
  --target_accuracy <X>    Target accuracy to reach for convergence time benchmarking. Required with --time.
  --help                   Show this help message and exit.
""")


def main():

    args = parser.parse_args()

    if args.help:
        print_help()
        return

    assert args.max_epochs, "Max epochs has to be provided!"

    assert args.accuracy != args.time, "Expected either 'accuracy' or 'time'"

    if args.time:
        assert args.target_accuracy, "You forgot to set a target_accuracy!"
        print(
            f"\n\n---- Training until convergence to {args.target_accuracy} accuracy (max {args.max_epochs} epochs) ----\n\n"
        )

    if args.accuracy:
        if args.target_accuracy:
            print(
                f"\033[33m[Warning]: You set a target_accuracy ({args.target_accuracy}) but are not benchmarking convergance time! Target accuracy will have no effect!\033[0m"
            )
        print(f"\n\n---- Training for {args.max_epochs} epochs ----\n\n")

    target_accuracy = 1.0 if args.accuracy else args.target_accuracy
    max_epochs = args.max_epochs

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Limit GPU:0 to 10 GiB (10240 MiB)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=10240)]
        )

    #strategies = [NeuronCentralitySET()]
    strategies = [FisherDiagonalSkipSET()]

    models = [SET_MLP_CIFAR10]

    results: list[tuple[str, int | float]] = []

    for model_cls in models:
        for strategy in strategies:
            model = model_cls(strategy=strategy, max_epochs=max_epochs)

            epoch_count, best_accuracy = model.train(
                target_accuracy=target_accuracy)
            if args.time:
                print(f"took {epoch_count} epochs until convergance")
                results.append((f"{strategy.__class__.__name__}", epoch_count))
            else:
                print(
                    f"Reached {best_accuracy*100}% accuracy after {max_epochs} epochs"
                )
                results.append(
                    (f"{strategy.__class__.__name__}", best_accuracy * 100))


# Pretty print results in a table
    print("\n" + "=" * 60)
    if args.time:
        print(
            f"{'Strategy (' + str(target_accuracy * 100) + '%)':<40} | {'Epochs':>7}"
        )
    else:
        print(
            f"{'Strategy (' + str(max_epochs) + ' epochs)':<40} | {'Accuracy (%)':>7}"
        )

    print("-" * 60)
    for name, metric in results:
        if isinstance(metric, float):
            print(f"{name:<40} | {metric:7.2f}")
        else:
            print(f"{name:<40} | {metric:>7}")
    print("=" * 60)

if __name__ == "__main__":
    main()
