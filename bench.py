from utils import RandomSET
from set_keras import SET_MLP_CIFAR10

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--target_accuracy', type=float, required=False)
parser.add_argument('--max_epochs', type=int, required=True)
parser.add_argument('--accuracy',
                    type=bool,
                    action="store_true",
                    required=False)
parser.add_argument('--time', type=bool, action="store_true", required=False)

args = parser.parse_args()

print(
    f"---- Training until convergence to {args.target_accuracy} accuracy (max {args.max_epochs} epochs) ----"
)

assert args.accuracy != args.time, "Expected either 'accuracy' or 'time'"

if args.time:
    assert args.target_accuracy, "You forgot to set a target_accuracy!"

if args.accuracy and args.target_accuracy:
    print(
        f"\033[33m[Warning]: You set a target_accuracy ({args.target_accuracy}) but are not benchmarking convergance time! Target accuracy will have no effect!\033[0m"
    )

target_accuracy = 1.0 if args.accuracy else args.target_accuracy
max_epochs = args.max_epochs

strategies = [RandomSET()]

models = [SET_MLP_CIFAR10]

results: list[tuple[str, int]] = []

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
                (f"{strategy.__class__.__name__}", int(best_accuracy * 100)))

# Pretty print results in a table
print("\n" + "=" * 60)
if args.time:
    print(
        f"{'Strategy (' + str(target_accuracy * 100) + '%)':<40} | {'Epochs':>7}"
    )
else:
    print(f"{'Strategy (' + str(max_epochs) + 'epochs)':<40} | {'Epochs':>7}")

print("-" * 60)
for name, metric in results:
    print(f"{name:<40} | {metric:>7}")
print("=" * 60)
