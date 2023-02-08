import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

def hierarchy(folder,nvec, labs): #BASED on https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html

    model = AgglomerativeClustering(affinity="cosine", linkage="single", distance_threshold=0, n_clusters=None)
    model = model.fit(nvec)

    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    dpi=96
    plt.figure(figsize=(1024/dpi,768/dpi),dpi=dpi)
    plot_dendrogram(model, orientation='right', labels=np.array(labs)) # truncate_mode="level", p=3,
    plt.title("Topic cluster based on topic vector and cosine distance")
    plt.xlabel("Similarity")
    plt.tight_layout()
    plt.savefig(folder + "/topic_visual_hierarchy.png")


def tsne(topic_model,nvec,folder,topicShort):
    topVis = TSNE(perplexity=min(8, topic_model.ntopics - 1))
    tsneX = topVis.fit_transform(nvec)
    dpi = 96
    plt.figure(figsize=(1024 / dpi, 768 / dpi), dpi=dpi)
    co = lambda i: [r[i] for r in tsneX]
    ptsize = topic_model.pt / (np.max(topic_model.pt) - np.min(topic_model.pt) + 1e-4)
    ptsize = 20 * ptsize + 5
    colors = list(plt.cm.get_cmap('tab10').colors) + list(plt.cm.get_cmap('tab20b').colors) + list(plt.cm.get_cmap('tab20c').colors)
    colors = colors + colors + colors
    colors = colors[:topic_model.ntopics]
    x, y = co(0), co(1)
    for i in range(topic_model.ntopics):
        plt.scatter(x[i], y[i], s=np.array(ptsize[i] ** 2), c=np.array([colors[i]]), alpha=0.5, label=topicShort[i])
        plt.text(x=x[i] + 0.1, y=y[i] + 0.1, s=topicShort[i].split("_")[0], fontdict=dict(color=colors[i], size=15))
    plt.legend(loc=(1.04, 0))
    plt.tight_layout()
    plt.savefig(folder + "topic_visual_tsne.png", dpi=300)