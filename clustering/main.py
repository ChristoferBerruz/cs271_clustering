import click
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


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


@cli.command()
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
    plt.show()


if __name__ == '__main__':
    cli()
