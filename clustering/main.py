import click
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from adaboost import adaboost
from data import ADA_BOOST_25_SAMPLES_30_CLASSIFIERS, ADA_BOOST_25_SAMPLES_30_CLASSIFIERS_LABELS
from data import ADA_BOOST_100_SAMPLES_250_CLASSIFIERS, ADA_BOOST_100_SAMPLES_250_CLASSIFIERS_LABELS

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import pandas as pd

import seaborn as sns


@click.group()
def cli():
    pass


RED_LABEL = 0
BLUE_LABEL = 1
BLUE_RED_DATA = data = np.array([
    [0.5, 3.00, 0],
    [1.0, 4.25, 0],
    [1.5, 2.00, 0],
    [2.0, 2.75, 0],
    [2.5, 1.65, 0],
    [3.0, 2.70, 0],
    [3.5, 1.00, 0],
    [4.0, 2.50, 0],
    [4.5, 2.10, 0],
    [5.0, 2.75, 0],
    [0.5, 1.75, 1],
    [1.5, 1.50, 1],
    [2.5, 4.00, 1],
    [2.5, 2.10, 1],
    [3.0, 1.50, 1],
    [3.5, 1.85, 1],
    [4.0, 3.50, 1],
    [5.0, 1.45, 1]
])


@cli.command(name="knn-n3")
def nearest_neighbors():
    light_red = '#FFC0CB'
    light_blue = '#ADD8E6'
    X = BLUE_RED_DATA[:, :2]
    Y = BLUE_RED_DATA[:, 2]
    scatter_cmap = ListedColormap(["red", "blue"])
    boundary_cmap = ListedColormap([light_red, light_blue])
    knn = KNeighborsClassifier(n_neighbors=3)
    clusterer = knn.fit(BLUE_RED_DATA[:, :2], Y)
    x_values, y_values = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max()),
        np.linspace(X[:, 1].min(), X[:, 1].max())
    )
    grid = np.vstack([x_values.ravel(), y_values.ravel()]).T
    disp = DecisionBoundaryDisplay.from_estimator(
        clusterer, grid, plot_method='contourf', cmap=boundary_cmap
    )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=Y, cmap=scatter_cmap, edgecolors="k")
    plt.savefig("results/knn_n3.png")


@cli.command(name="ada-n25-c30")
def adaboost_25_samples_100_classifiers():
    weak_classifiers = ADA_BOOST_25_SAMPLES_30_CLASSIFIERS
    labels = ADA_BOOST_25_SAMPLES_30_CLASSIFIERS_LABELS
    n_samples, l_classifiers = ADA_BOOST_25_SAMPLES_30_CLASSIFIERS.shape
    X = np.arange(n_samples)
    strong_classifiers = adaboost(
        X, labels, weak_classifiers
    )
    df_strong = pd.DataFrame(strong_classifiers, columns=[
        f"C{i+1}" for i in range(l_classifiers)])
    df_weak = pd.DataFrame(weak_classifiers, columns=[
                           f"c{i+1}" for i in range(l_classifiers)])
    hits_strong = calculate_hits(strong_classifiers, labels)
    for df in [df_strong, df_weak]:
        df.insert(0, "X", [f"X{j+1}" for j in range(n_samples)])
        df.insert(1, "Z", labels)
    # add the hits to the strong classifiers as last row
    last_row = ["Hits", " ", *hits_strong]
    df_strong.loc[l_classifiers] = last_row
    df_strong.to_csv("results/ada-25-30-strong.csv", index=False)
    df_weak.to_csv("results/ada-25-30-weak.csv", index=False)


def calculate_hits(strong_classifiers: np.ndarray, labels: np.ndarray):
    _, l_classifiers = strong_classifiers.shape
    predictions_for_all_classifiers = np.sign(strong_classifiers)
    # compute accuracies by comparing with the labels
    hits = np.sum(predictions_for_all_classifiers ==
                  np.array([labels] * l_classifiers).T, axis=0)
    return hits


@cli.command(name="ada-n100-c250")
def adaboost_100_samples_250_classifiers():
    weak_classifiers = ADA_BOOST_100_SAMPLES_250_CLASSIFIERS
    labels = ADA_BOOST_100_SAMPLES_250_CLASSIFIERS_LABELS
    n_samples, l_classifiers = weak_classifiers.shape
    X = np.arange(n_samples)
    strong_classifiers = adaboost(
        X, labels, weak_classifiers
    )
    hits = calculate_hits(strong_classifiers, labels)
    df_strong_c250 = pd.DataFrame(strong_classifiers[:, -1], columns=["C250"])
    df_strong_c250.insert(0, "X", [f"X{j+1}" for j in range(n_samples)])
    df_strong_c250.insert(1, "Z", labels)
    last_row = ["Hits", " ", hits[-1]]
    df_strong_c250.loc[l_classifiers] = last_row
    df_strong_c250.to_csv("results/ada-100-250-strongc250.csv", index=False)

    accuracies = hits/n_samples

    accuracy_df = pd.DataFrame({
        "M": [i for i in range(l_classifiers)],
        "Accuracy": accuracies
    })
    sns.lineplot(data=accuracy_df, x="M", y="Accuracy")
    plt.savefig("results/ada-100-250-accuracy-vs-m.png")
    best_m_classifiers = []
    for m, value in enumerate(accuracies):
        if value > 0.9:
            best_m_classifiers.append(m+1)
    print(f"Accuracy of C250 = {accuracies[-1]*100}%")
    print(
        f"m values such that Cm accuracy > 90%: {best_m_classifiers}")


if __name__ == '__main__':
    cli()
