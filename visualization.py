import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from data_preparation import process
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torchvision.models as models

from HGCN import HGCN
# data = process()
# node_features = data.node_features

# # Range of clusters to try
# cluster_range = range(31, 51)
# wcss = []

# for k in cluster_range:
#     kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(node_features)
#     wcss.append(kmeans.inertia_)

# plt.figure(figsize=(10, 6))
# plt.plot(cluster_range, wcss, marker='o', linestyle='-', color='b')
# plt.title('Elbow Method For Optimal k')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.xticks(cluster_range)
# plt.show()

model = HGCN()
state_dict = torch.load('HGCN.pth')
model.load_state_dict(state_dict)
model.eval()

# F3D visualization
weights = state_dict['hcn1.bias'].detach().numpy()

tsne = TSNE(n_components=3, perplexity=30.0, n_iter=300, random_state=42)
tsne_results = tsne.fit_transform(weights)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_results[:, 0], tsne_results[:, 1],
           tsne_results[:, 2], alpha=0.5)
ax.set_title('3D t-SNE Visualization of Hypergraph Model')
ax.set_xlabel('t-SNE Feature 1')
ax.set_ylabel('t-SNE Feature 2')
ax.set_zlabel('t-SNE Feature 3')
plt.show()
