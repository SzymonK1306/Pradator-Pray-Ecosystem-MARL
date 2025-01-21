import pandas as pd
import matplotlib.pyplot as plt


def plot_predator_prey(filename):
    df = pd.read_csv(filename)

    df.columns = [
        "epoch", "num_predators", "num_preys", "avg_attack",
        "avg_speed_predators", "avg_resilience", "avg_speed_preys"
    ]

    df = df.astype({
        "epoch": int,
        "num_predators": int,
        "num_preys": int
    })

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["num_predators"], label="Predators", color="red")
    plt.plot(df["epoch"], df["num_preys"], label="Preys", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Population")
    plt.title("Population of Predators and Preys Over Time")
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 7))
    scatter = plt.scatter(df["num_preys"], df["num_predators"], c=df["epoch"], cmap="viridis", alpha=0.75)
    plt.colorbar(scatter, label="Epoch")
    plt.xlabel("Number of Preys")
    plt.ylabel("Number of Predators")
    plt.title("Predators vs. Preys (Colored by Epoch)")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["avg_attack"], label="Avg Attack", linestyle="--")
    plt.plot(df["epoch"], df["avg_speed_predators"], label="Avg Speed (Predators)", linestyle="-.")
    plt.plot(df["epoch"], df["avg_resilience"], label="Avg Resilience", linestyle=":")
    plt.plot(df["epoch"], df["avg_speed_preys"], label="Avg Speed (Preys)", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Genetic Traits")
    plt.title("Genetic Traits Over Time")
    plt.legend()
    plt.show()


plot_predator_prey("output_ENV_1_more_hunger_ceil.csv")
