import click
import numpy as np
from typing import Dict, Tuple, List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from adaboost import adaboost
from kmeans import KMeans
from gaussian_mixture import GaussianMixture, EMCluster, Theta, GaussianMixtureEstimator
from dbscan import DBSCAN
from data import ADA_BOOST_25_SAMPLES_30_CLASSIFIERS, ADA_BOOST_25_SAMPLES_30_CLASSIFIERS_LABELS
from data import ADA_BOOST_100_SAMPLES_250_CLASSIFIERS, ADA_BOOST_100_SAMPLES_250_CLASSIFIERS_LABELS
from data import OLD_FAITHFUL
from data import DBSCAN_DATA


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
    last_row = [" ", "Hits", *hits_strong]
    df_strong.loc[l_classifiers] = last_row
    df_strong.to_csv("results/ada-25-30-strong.csv", index=False)
    df_weak.to_csv("results/ada-25-30-weak.csv", index=False)
    # finding the classifier where the accuracy is 100%
    accuracies = hits_strong/n_samples
    first_m_perfect_accuracy = np.argwhere(accuracies == 1).flatten()[0]
    print(
        f"First m such that Cm's accuracy is perfect: {first_m_perfect_accuracy+1}")


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
    last_row = [" ", "Hits", hits[-1]]
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
    

@cli.command(name="kmeans")
@click.option("--n-clusters", default=2, help="Number of clusters")
def kmeans(n_clusters: int):
    X = OLD_FAITHFUL
    n_kmeans = KMeans(n_clusters=n_clusters)
    estimator = n_kmeans.fit(X)
    centroids = np.array([cluster.centroid for cluster in estimator.clusters_])
    labels_of_X = estimator.predict(X)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', label="centroid")
    for label in range(n_clusters):
        cluster = X[labels_of_X == label]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f"cluster #{label}")
    # x-axis is duration, y-axis is waiting time
    plt.xlabel("Duration")
    plt.ylabel("Waiting time")
    plt.legend(loc="upper left")
    plt.savefig(f"results/kmeans_n{n_clusters}.png")


@cli.command(name="gm-2nditer")
def old_faithful_gm_second_iter():
    """Compute the second pji iteration of the gaussian mixture model
    on the Old Faithful dataset
    """
    X = OLD_FAITHFUL
    cluster_one = EMCluster(
        tau=0.6, theta=Theta(u=np.array([2.5, 65.0]), S=np.array([[1.0, 5.0], [5.0, 100.0]])))
    cluster_two = EMCluster(
        tau=0.4, theta=Theta(u=np.array([3.5, 70.0]), S=np.array([[2.0, 10.0], [10.0, 200.0]])))
    gm = GaussianMixture(clusters=[cluster_one, cluster_two])
    pji = gm.e_step(X)
    def prinpij(pji: np.ndarray):
        k_clusters, n_samples = pji.shape
        rows = 10
        cols = k_clusters*n_samples//rows
        new_matrix = np.zeros((rows, cols))
        collected = []
        flushes = 0
        for i in range(n_samples, ):
            for j in range(k_clusters):
                collected.append(pji[j, i])
                if len(collected) == 10:
                    new_matrix[:, flushes] = np.array(collected)
                    collected.clear()
                    flushes += 1
        for row in range(rows):
            tokens = []
            base = row // 2
            for col in range(cols):
                j = row % k_clusters + 1
                shift = 5*col + 1
                i = shift + base
                row_str = f"pji[{j}, {i:2}] = {new_matrix[row, col]:.4f}"
                tokens.append(row_str)
            print(" | ".join(tokens))
    print("PIJ after the first iteration")        
    prinpij(pji)
    gm.step(X)
    second_pji = gm.e_step(X)
    print("PIJ after the second iteration")
    prinpij(second_pji)
    gm.step(X)
    print("Recomputed parameters after the second iteration")
    for idx, cluster in enumerate(gm.clusters):
        cluster_idx = idx + 1
        print(f"Cluster # {cluster_idx}")
        print("Tau:", cluster.tau)
        print("Mean:", cluster.theta.u)
        print("Covariance matrix:")
        for row in cluster.theta.S:
            print(row)


@cli.command(name="em-old-faithful")
def em_clustering_old_faithful_data():
    X = OLD_FAITHFUL
    ##### PART A: Resuse the initial clusters from the other assignment #####
    reused_cluster_one = EMCluster(
        tau=0.6, theta=Theta(u=np.array([2.5, 65.0]), S=np.array([[1.0, 5.0], [5.0, 100.0]])))
    reused_cluster_two = EMCluster(
        tau=0.4, theta=Theta(u=np.array([3.5, 70.0]), S=np.array([[2.0, 10.0], [10.0, 200.0]])))
    reused_estimator = GaussianMixtureEstimator()
    fitted_reused_estimator = reused_estimator.fit(X, [reused_cluster_one, reused_cluster_two])

    ##### PART B: Random initialization of the clusters #####
    random_cluster_one = EMCluster(
        tau=0.55, theta=Theta(u=np.array([1.5, 60.0]), S=np.array([[1.0, 5.5], [5.0, 110.0]])))
    random_cluster_two = EMCluster(
        tau=0.45, theta=Theta(u=np.array([4.5, 80.0]), S=np.array([[2.0, 10.5], [10.0, 210.0]])))
    random_estimator = GaussianMixtureEstimator()
    fitted_random_estimator = random_estimator.fit(X, [random_cluster_one, random_cluster_two])

    ##### PART C: Use three clusters with some initial values #####
    cluster_one = EMCluster(
        tau=0.4, theta=Theta(u=np.array([2.5, 65.0]), S=np.array([[1.0, 5.0], [5.0, 100.0]])))
    cluster_two = EMCluster(
        tau=0.3, theta=Theta(u=np.array([3.5, 70.0]), S=np.array([[2.0, 10.0], [10.0, 200.0]])))
    cluster_three = EMCluster(
        tau=0.3, theta=Theta(u=np.array([4.5, 75.0]), S=np.array([[2.0, 10.0], [10.0, 200.0]])))
    three_clusters_estimator = GaussianMixtureEstimator()
    fitted_three_clusters_estimator = three_clusters_estimator.fit(X, [cluster_one, cluster_two, cluster_three])

    fitted_estimator_and_initial_params: Dict[str, Tuple[GaussianMixture, List[EMCluster]]] = {
        "PartA": (fitted_reused_estimator, [reused_cluster_one, reused_cluster_two]),
        "PartB": (fitted_random_estimator, [random_cluster_one, random_cluster_two]),
        "PartC": (fitted_three_clusters_estimator, [cluster_one, cluster_two, cluster_three])
    }

    available_mean_colors = ["red", "cyan", "magenta"]
    for part, (fitted_estimator, original_clusters) in fitted_estimator_and_initial_params.items():
        print(f"{part}")
        print("Original clusters")
        for idx, cluster in enumerate(original_clusters):
            print(f"Cluster # {idx + 1}")
            print("Tau:", cluster.tau)
            print("Mean:", cluster.theta.u)
            print("Covariance matrix:")
            for row in cluster.theta.S:
                print(row)
        print("---------------------------------")
        print("Fitted clusters")
        for idx, cluster in enumerate(fitted_estimator.gm.clusters):
            print(f"Cluster # {idx + 1}")
            print("Tau:", cluster.tau)
            print("Mean:", cluster.theta.u)
            print("Covariance matrix:")
            for row in cluster.theta.S:
                print(row)
        print("#################################\n")
        # plot the cluster
        fig, ax = plt.subplots()
        predicted_labels = fitted_estimator.predict(X)
        for idx, cluster in enumerate(fitted_estimator.gm.clusters):
            cluster_idx = idx
            cluster_points = X[predicted_labels == cluster_idx]
            mean = cluster.theta.u
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster #{cluster_idx+1}")
            ax.scatter(mean[0], mean[1], c=available_mean_colors[cluster_idx],
                       label=f"mean of cluster # {cluster_idx+1}", marker="x")
            title_by_part = {
                "PartA": "2 Clusters, initials values from equations",
                "PartB": "2 Clusters, user-defined initial values",
                "PartC": "3 Clusters, user-defined initial values"
            }
            ax.set_title(title_by_part[part])
        plt.xlabel("Duration")
        plt.ylabel("Waiting time")
        plt.legend(loc="upper left")
        fig.savefig(f"results/em_{part}.png")

@cli.command()
def dbscan():
    X = DBSCAN_DATA
    epsilon_ms = [
        (0.6, 3),
        (0.75, 4),
        (1.0, 5),
        (2.0, 10)
    ]
    for _, (epsilon, m) in enumerate(epsilon_ms):
        fig, ax = plt.subplots()
        dbscan = DBSCAN(m=m, epsilon=epsilon)
        dbscan.fit(X)
        predictions = dbscan.predict(X)
        outliers = X[predictions == -1]
        # find the number of outliers
        n_outliers = len(outliers)
        total_points = len(X)
        ax.scatter(outliers[:, 0], outliers[:, 1], c="black", marker="x", label=f"Outliers: {n_outliers}")
        ax.set_title(f"DBSCAN with epsilon = {epsilon}, m = {m}")
        print("Report for DBSCAN with epsilon = ", epsilon, "m = ", m)
        print("Total number of points:", total_points)
        print("Number of outliers:", n_outliers)
        # also let's find the number of clusters and how many points in each
        print("Number of clusters:", len(dbscan.clusters_))
        classses_to_colors = plt.get_cmap("tab20")
        for idx, cluster in enumerate(dbscan.clusters_):
            n_points = len(cluster.points_idx)
            print(f"Cluster # {idx} has {n_points} points")
            cluster_points = X[predictions == idx]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster # {idx}: {n_points}", color=classses_to_colors(idx))
        plt.legend(loc=(1.05, 0.5), borderaxespad=0)
        fig.savefig(
            f"results/dbscan_epsilon_{epsilon}_m_{m}.png",  bbox_inches='tight')

        print("##############################################")


if __name__ == '__main__':
    cli()
