import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from model.data_preparation import process
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torchvision.models as models

# from HGNN import HGN
from model.HGNNP import HGNP

def visualize_with_tsne(model, data, labels):
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass through the model to get the features
    with torch.no_grad():
        features = model(data.x, data.hg)
    
    # Convert features to numpy array if not already
    features_np = features.detach().numpy()
    
    # Initialize t-SNE with desired parameters
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    
    # Fit and transform the features to 2D
    tsne_results = tsne.fit_transform(features_np)
    
    # Plotting
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE visualization of model features")
    plt.show()

if __name__ == "__main__":
    model, data = HGNP()
    # Assume 'labels' is a 1D array or list of true class labels corresponding to 'data.x'
    labels = data.y.detach().numpy()  # Modify according to how your labels are stored
    
    # Visualize
    visualize_with_tsne(model, data, labels)
