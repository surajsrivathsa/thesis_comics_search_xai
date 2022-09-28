import time
from sklearn import datasets, metrics
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans, MPCKMeans
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def get_pca(data):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    return components


def get_tsne(data):
    tsne = TSNE(n_components=2, learning_rate='auto', init='random')
    components = tsne.fit_transform(data)
    return components


def cop_kmeans(dataset, k, ml=[], cl=[],
               initialization='kmpp',
               max_iter=300, tol=1e-4):

    ml, cl = transitive_closure(ml, cl, len(dataset))
    ml_info = get_ml_info(ml, dataset)
    tol = tolerance(tol, dataset)

    centers = initialize_centers(dataset, k, initialization)

    for _ in range(max_iter):
        clusters_ = [-1] * len(dataset)
        for i, d in enumerate(dataset):
            indices, _ = closest_clusters(centers, d)
            counter = 0
            if clusters_[i] == -1:
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    index = indices[counter]
                    if not violate_constraints(i, index, clusters_, ml, cl):
                        found_cluster = True
                        clusters_[i] = index
                        for j in ml[i]:
                            clusters_[j] = index
                    counter += 1

                if not found_cluster:
                    return None, None

        clusters_, centers_ = compute_centers(clusters_, dataset, k, ml_info)
        shift = sum(l2_distance(centers[i], centers_[i]) for i in range(k))
        if shift <= tol:
            break

        centers = centers_

    return clusters_, centers_


def l2_distance(point1, point2):
    return sum([(float(i)-float(j))**2 for (i, j) in zip(point1, point2)])

# taken from scikit-learn (https://goo.gl/1RYPP5)


def tolerance(tol, dataset):
    n = len(dataset)
    dim = len(dataset[0])
    averages = [sum(dataset[i][d] for i in range(n))/float(n)
                for d in range(dim)]
    variances = [sum((dataset[i][d]-averages[d]) **
                     2 for i in range(n))/float(n) for d in range(dim)]
    return tol * sum(variances) / dim


def closest_clusters(centers, datapoint):
    distances = [l2_distance(center, datapoint) for
                 center in centers]
    return sorted(range(len(distances)), key=lambda x: distances[x]), distances


def initialize_centers(dataset, k, method):
    if method == 'random':
        ids = list(range(len(dataset)))
        random.shuffle(ids)
        return [dataset[i] for i in ids[:k]]

    elif method == 'kmpp':
        chances = [1] * len(dataset)
        centers = []

        for _ in range(k):
            chances = [x/sum(chances) for x in chances]
            r = random.random()
            acc = 0.0
            for index, chance in enumerate(chances):
                if acc + chance >= r:
                    break
                acc += chance
            centers.append(dataset[index])

            for index, point in enumerate(dataset):
                cids, distances = closest_clusters(centers, point)
                chances[index] = distances[cids[0]]

        return centers


def violate_constraints(data_index, cluster_index, clusters, ml, cl):
    for i in ml[data_index]:
        if clusters[i] != -1 and clusters[i] != cluster_index:
            return True

    for i in cl[data_index]:
        if clusters[i] == cluster_index:
            return True

    return False


def compute_centers(clusters, dataset, k, ml_info):
    cluster_ids = set(clusters)
    k_new = len(cluster_ids)
    id_map = dict(zip(cluster_ids, range(k_new)))
    clusters = [id_map[x] for x in clusters]

    dim = len(dataset[0])
    centers = [[0.0] * dim for i in range(k)]

    counts = [0] * k_new
    for j, c in enumerate(clusters):
        for i in range(dim):
            centers[c][i] += dataset[j][i]
        counts[c] += 1

    for j in range(k_new):
        for i in range(dim):
            centers[j][i] = centers[j][i]/float(counts[j])

    if k_new < k:
        ml_groups, ml_scores, ml_centroids = ml_info
        current_scores = [sum(l2_distance(centers[clusters[i]], dataset[i])
                              for i in group)
                          for group in ml_groups]
        group_ids = sorted(range(len(ml_groups)),
                           key=lambda x: current_scores[x] - ml_scores[x],
                           reverse=True)

        for j in range(k-k_new):
            gid = group_ids[j]
            cid = k_new + j
            centers[cid] = ml_centroids[gid]
            for i in ml_groups[gid]:
                clusters[i] = cid

    return clusters, centers


def visualise_clusters(data, transformed_data, config=1):
    data['constrained_labels'] = data['constrained_labels'].astype(str)
    fig = px.scatter(data, x="tsne_original_x_axis", y="tsne_original_y_axis", color='labels', color_discrete_sequence=px.colors.qualitative.Vivid,
                     hover_name="headline", hover_data=[data.index, "keywords", "subject"], width=1200, height=900, title="Original labels with original data")
    fig.show()
    fig1 = px.scatter(data, x="tsne_original_x_axis", y="tsne_original_y_axis", color='constrained_labels', color_discrete_sequence=px.colors.qualitative.Vivid,
                      hover_name="headline", hover_data=[data.index, "keywords", "subject"], width=1200, height=900, title="Constrained labels with original data")
    fig1.show()
    fig2 = px.scatter(data, x="tsne_transformed_x_axis", y="tsne_transformed_y_axis", color='labels', color_discrete_sequence=px.colors.qualitative.Vivid,
                      hover_name="headline", hover_data=[data.index, "keywords", "subject"], width=1200, height=900, title="Original labels with transformed data")
    fig2.show()
    fig3 = px.scatter(data, x="tsne_transformed_x_axis", y="tsne_transformed_y_axis", color='constrained_labels', color_discrete_sequence=px.colors.qualitative.Vivid,
                      hover_name="headline", hover_data=[data.index, "keywords", "subject"], width=1200, height=900, title="Constrained labels with transformed data")
    fig3.show()
    fig4 = px.scatter(data, x="tsne_transformed_x_axis", y="tsne_transformed_y_axis", color='labels', color_discrete_sequence=px.colors.qualitative.Vivid,
                      hover_name="headline", hover_data=[data.index, "keywords", "subject"], facet_col="constrained_labels", width=1200, height=900, title="Label changes with transformed data")
    fig4.show()
    if(config == 3):
        tsne_components_transformed = get_tsne(transformed_data)
        data = pd.concat([data, pd.DataFrame(tsne_components_transformed, columns=[
                         'tsne_data_transformed_x_axis', 'tsne_data_transformed_y_axis'])], axis=1)
        fig5 = px.scatter(data, x="tsne_data_transformed_x_axis", y="tsne_data_transformed_y_axis", color='labels', color_discrete_sequence=px.colors.qualitative.Vivid,
                          hover_name="headline", hover_data=[data.index, "keywords", "subject"], width=1200, height=900, title="Original labels with scaled MPCKM data")
        fig5.show()
        fig6 = px.scatter(data, x="tsne_data_transformed_x_axis", y="tsne_data_transformed_y_axis", color='constrained_labels', color_discrete_sequence=px.colors.qualitative.Vivid,
                          hover_name="headline", hover_data=[data.index, "keywords", "subject"], width=1200, height=900, title="Constrained labels with scaled MPCKM data")
        fig6.show()


def get_ml_info(ml, dataset):
    flags = [True] * len(dataset)
    groups = []
    for i in range(len(dataset)):
        if not flags[i]:
            continue
        group = list(ml[i] | {i})
        groups.append(group)
        for j in group:
            flags[j] = False

    dim = len(dataset[0])
    scores = [0.0] * len(groups)
    centroids = [[0.0] * dim for i in range(len(groups))]

    for j, group in enumerate(groups):
        for d in range(dim):
            for i in group:
                centroids[j][d] += dataset[i][d]
            centroids[j][d] /= float(len(group))

    scores = [sum(l2_distance(centroids[j], dataset[i])
                  for i in groups[j])
              for j in range(len(groups))]

    return groups, scores, centroids


def transitive_closure(ml, cl, n):
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in cl:
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception(
                    'inconsistent constraints between %d and %d' % (i, j))

    return ml_graph, cl_graph


def perform_constrained_clustering(data, must_link, cannot_link, k, config=1):
    start_time = time.time()
    cluster_matrix = None
    if(config == 1):
        print('Results for COP-Means')
        clusters, centers = cop_kmeans(
            dataset=data, k=k, ml=must_link, cl=cannot_link)
    elif(config == 2):
        print('Results for PCK-Means')
        clusterer = PCKMeans(n_clusters=k)
        clusterer.fit(data, ml=must_link, cl=cannot_link)
        clusters = clusterer.labels_
        centers = clusterer.cluster_centers_
    else:
        print('Results for MPCK-Means')
        clusterer = MPCKMeans(n_clusters=k, max_iter=20)
        clusterer.fit(data, ml=must_link, cl=cannot_link)
        clusters = clusterer.labels_
        centers = clusterer.cluster_centers_
        cluster_matrix = clusterer.matrix_
        data = np.dot(data, clusterer.matrix_)
    print("--- %s seconds ---" % (time.time() - start_time))
    return clusters, centers, data, cluster_matrix
