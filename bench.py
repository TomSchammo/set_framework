from utils import RandomSET
from set_keras import SET_MLP_CIFAR10

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--target_accuracy', type=float, required=True)
parser.add_argument('--max_epochs', type=int, required=True)

args = parser.parse_args()

print(
    f"---- Training until convergence to {args.target_accuracy} accuracy (max {args.max_epochs} epochs) ----"
)

strategies = [RandomSET()]

models = [SET_MLP_CIFAR10]

results: list[tuple[str, int]] = []

for model_cls in models:
    for strategy in strategies:
        model = model_cls(strategy=strategy, max_epochs=args.max_epochs)

        epoch_count = model.train(target_accuracy=args.target_accuracy)
        print(f"took {epoch_count} epochs until convergance")
        results.append(
            (f"{model.__class__.__name__} + {strategy.__class__.__name__}",
             epoch_count))

# Pretty print results in a table
print("\n" + "=" * 60)
target_accuracy = args.target_accuracy
print(
    f"{'Model + Strategy (' + str(target_accuracy * 100) + '%)':<40} | {'Epochs':>7}"
)
print("-" * 60)
for name, epoch_count in results:
    print(f"{name:<40} | {epoch_count:>7}")
print("=" * 60)
