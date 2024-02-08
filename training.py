
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch.nn.functional as F
from utils import seed_setting

# from data_preparation import process
from main import HGCN
from GCN import GCN
from GAT import GAT
from G_transformer import GCNCT
from DHGCN import DHGCN
import joblib
import pickle

# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=200, verbose=False, delta=0):
#         """
#         :param patience: How long to wait after last time validation loss improved.
#         :param verbose: If True, prints a message for each validation loss improvement.
#         :param delta: Minimum change in the monitored quantity to qualify as an improvement.
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta

#     def __call__(self, val_loss, model):
#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             if self.verbose:
#                 print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    #     """Saves model when validation loss decrease."""
    #     if self.verbose:
    #         print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
    #     torch.save(model.state_dict(), 'checkpoint.pt')
    #     self.val_loss_min = val_loss


# Random guess
def random_guess_baseline(y_true):
    seed_setting()

    # Generate random predictions with the same shape as y_true
    random_predictions = np.random.randint(0, 2, size=y_true.shape)

    # Flatten the arrays for multilabel classification
    y_true_flat = y_true.ravel()
    random_predictions_flat = random_predictions.ravel()

    # Calculate accuracy for flattened arrays
    acc = accuracy_score(y_true_flat, random_predictions_flat)

    # Calculate other metrics
    prec = precision_score(y_true_flat, random_predictions_flat, average='macro', zero_division=0)
    rec = recall_score(y_true_flat, random_predictions_flat, average='macro', zero_division=0)
    f1 = f1_score(y_true_flat, random_predictions_flat, average='macro', zero_division=0)

    return acc, prec, rec, f1

# Split dataset to train, test, and train
def main_training_loop(model, data):
    seed_setting()

    # print(f"criterion: {criterion}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) 
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    def train(model, data, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(data.node_features, data.hg)
        # print("Output shape:", out.shape)
               
        # train_outputs = out[data.train_mask]
        # print("Train outputs shape:", train_outputs.shape)
        # Get target labels for the training set
        target = data.y[data.train_mask].squeeze()
        target = target.long()
        # print("Target shape:", target.shape)
                
        loss = F.cross_entropy(out[data.train_mask], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss
    
    def validate(model, data):
        model.eval()
        with torch.no_grad():
            out = model(data.node_features, data.hg)
            target = data.y[data.val_mask].squeeze()
            target = target.long()
            val_loss = F.cross_entropy(out[data.val_mask], target)
        return val_loss.item()

    def test(model, data):
        model.eval()
        out = model(data.node_features, data.hg)
        pred = out.argmax(dim=1) 
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy() 
               
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        
        # cm = confusion_matrix(y_true_flat, y_pred_flat)
        # print("Confusion Matrix:\n", cm)
        
        # roc_auc = roc_auc_score(y_true_flat, y_pred_flat, multi_class="ovr")
        # print(f'ROC-AUC Score: {roc_auc:.4f}')
        
        # Micro metrics
        micro_precision = precision_score(y_true_flat, y_pred_flat, average='micro')
        micro_recall = recall_score(y_true_flat, y_pred_flat, average='micro')
        micro_f1 = f1_score(y_true_flat, y_pred_flat, average='micro')
        
        # Macro metrics
        test_precision = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
        test_recall = recall_score(y_true_flat, y_pred_flat, average='macro')
        test_f1 = f1_score(y_true_flat, y_pred_flat, average='macro')
        test_acc = accuracy_score(y_true_flat, y_pred_flat)
        
        return test_acc, test_precision, test_recall, test_f1, micro_precision, micro_recall, micro_f1
    
    best_val_loss = float('inf')
    # early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # Train and print the best model
    for epoch in range(1, 201):
        train_loss = train(model, data, optimizer)
        val_loss = validate(model, data)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}")
        if epoch%10 == 0:
            
            print(f"Epoch: {epoch}")
        # early_stopping(val_loss, model)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        
    test_acc, test_precision, test_recall, test_f1,  micro_precision, micro_recall, micro_f1 = test(model, data)
    print(f'Test Metrics - Accuracy: {test_acc:.4f}, Precision (Macro): {test_precision:.4f}, Recall (Macro): {test_recall:.4f}, F1 Score (Macro): {test_f1:.4f}')
    print(f'Micro Metrics - Precision: {micro_precision:.4f}, Recall: {micro_recall:.4f}, F1 Score: {micro_f1:.4f}')

    # Compare with random guess baseline
    y_true = data.y[data.test_mask].cpu().numpy()
    rand_acc, rand_prec, rand_rec, rand_f1 = random_guess_baseline(y_true)
    print(f'Random Guess - Accuracy: {rand_acc:.4f}, Precision: {rand_prec:.4f}, Recall: {rand_rec:.4f}, F1 Score: {rand_f1:.4f}')  


def final_train():
    # model, data = HGCN()
    model,data = DHGCN()
    # model,data = GAT()
    # model,data = GCN()
    # model,data = GCNCT()
    main_training_loop(model, data)
    # torch.save(model.state_dict(), 'HGCN.pth')
    
    
if __name__ == "__main__":  
    final_train()
    

class SimpleHypergraph:
    def __init__(self, num_v, hyperedges):
        self.num_v = num_v 
        self.hyperedges = hyperedges
        
    def to_edge_list(self):
        pass
def Hypergraph(data, df):
    num_v = data.x.shape[0]
    edge_index = data.edge_index
    user_to_index = {username: i for i,
                     username in enumerate(df['Username'])}

    group_hyperedges = []
    group_to_hyperedge = {}
    hyperedge_id = 0

    for _, row in df.iterrows():
        user = row['Username']
        try:
            # Convert string to list
            groups = ast.literal_eval(row['Groups'])
        except ValueError:
            groups = []
        for group in groups:
            if group not in group_to_hyperedge:
                group_to_hyperedge[group] = hyperedge_id
                hyperedge_id += 1
            group_hyperedges.append(
                (group_to_hyperedge[group], user_to_index[user]))

    # Convert group_hyperedges to a tensor
    group_hyperedges_tensor = torch.tensor(
        group_hyperedges, dtype=torch.long).t().contiguous()

    print("Shape of group-edges:", group_hyperedges_tensor.shape)
    print("Number of unique groups:", len(group_to_hyperedge))

    # Clustering Nodes with K-means
    k = 100
    node_features = data.node_features
    kmeans = KMeans(n_clusters=k, random_state=10).fit(
        node_features.detach().numpy())
    clusters = kmeans.labels_

    # Map each node to its new hyperedge (cluster)
    cluster_to_hyperedge = {i: hyperedge_id + i for i in range(k)}
    k_hyperedges = [(cluster_to_hyperedge[label], node)
                    for node, label in enumerate(clusters)]
    k_hyperedges_tensor = torch.tensor(
        k_hyperedges, dtype=torch.long).t().contiguous()

    print("Shape of k_hyperedges:", k_hyperedges_tensor.shape)

    # 2-hop hyperedge
    assert edge_index.shape[0] == 2
    group_hyperedges = []
    group_to_hyperedge = {}
    hyperedge_id = 0
    edge_index_2, edge_mask, ID_node_mask = dropout_node(
        edge_index, p=0.0, num_nodes=num_v)

    adj = SparseTensor.from_edge_index(
        edge_index_2, sparse_sizes=(num_v, num_v))
    adj = adj + adj @ adj
    row, col, _ = adj.coo()
    edge_index_2hop = torch.stack([row, col], dim=0)
    edge_index_2hop, _ = remove_self_loops(edge_index_2hop)

    print("Shape of edge_index_2hop:", edge_index_2hop.shape)

    group_hyperedges_tensor_shape = group_hyperedges_tensor.shape[1]
    edge_index_2hop_shape = edge_index_2hop.shape[1]
    
    all_hyperedges = group_hyperedges + k_hyperedges
    
    hyperedge_dict = {}
    for hyperedge_id, node_id in all_hyperedges:
        if hyperedge_id not in hyperedge_dict:
            hyperedge_dict[hyperedge_id] = []
        hyperedge_dict[hyperedge_id].append(node_id)

    # Now, create a list of lists for hyperedges
    hyperedge_list = list(hyperedge_dict.values())

    # Create the Hypergraph object
    hypergraph = SimpleHypergraph(hyperedges=hyperedge_list)
    data.hg = hypergraph
    num_v = data.x.shape[0]
    
    return data