import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# Find all CSV files that start with "results_"
csv_files = glob.glob("results_*.csv")

if not csv_files:
    raise FileNotFoundError("No files found matching results_*.csv")

plt.figure()

for file in csv_files:
    # Extract strategy name from filename
    # e.g. results_MyStrategy.csv -> MyStrategy
    strategy_name = os.path.splitext(file)[0].replace("results_", "")

    # Read CSV
    df = pd.read_csv(file)

    # Expect columns: epoch, accuracy
    if not {"epoch", "accuracy"}.issubset(df.columns):
        raise ValueError(f"{file} must contain 'epoch' and 'accuracy' columns")

    plt.plot(df["epoch"], df["accuracy"], label=strategy_name)

# Plot formatting
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch for All Strategies")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_per_epoch.png", dpi=300)
plt.close()
