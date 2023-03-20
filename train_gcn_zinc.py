import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from sklearn.metrics import r2_score
from torch.nn.functional import one_hot
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from gcn_model import GCN, GCNNet


def train(model, train_loader, optimizer, loss):
    model.train()
    loss_acc = 0
    total_graphs = 0
    total_preds = []
    labels = []

    for graph_batch in train_loader:
        graph_batch = graph_batch.to(device)
        optimizer.zero_grad()
        
        x_oh = one_hot(graph_batch.x.flatten(), num_classes=21).type(torch.cuda.FloatTensor)
        preds = model(x_oh, graph_batch.edge_index, graph_batch.batch).squeeze()
        loss_val = loss(preds, graph_batch.y)
        loss_acc += loss_val.item()
        total_graphs += graph_batch.num_graphs
        loss_val.backward()
        optimizer.step()

        total_preds.extend(preds.cpu().detach().numpy())
        labels.extend(graph_batch.y.cpu().detach().numpy())
        
    loss_acc /= total_graphs
    r2 = r2_score(labels, total_preds)
    return loss_acc, r2


def validate(model, valid_loader, loss):
    model.eval()
    loss_acc = 0
    total_graphs = 0
    total_preds = []
    labels = []
    with torch.no_grad():
        for graph_batch in valid_loader:
            graph_batch = graph_batch.to(device)
            x_oh = one_hot(graph_batch.x.flatten(), num_classes=21).type(torch.cuda.FloatTensor)
            preds = model(x_oh, graph_batch.edge_index, graph_batch.batch).squeeze()
            loss_val = loss(preds, graph_batch.y)
            loss_acc += loss_val.item()
            total_graphs += graph_batch.num_graphs
            total_preds.extend(preds.cpu().numpy())
            labels.extend(graph_batch.y.cpu().numpy())

    r2 = r2_score(labels, total_preds)            
    loss_acc /= total_graphs
    return loss_acc, r2


if __name__ == '__main__':
    import json

    DATA_FOLDER = './data/zinc/'
    NUMBER_OF_ATOMS = 21

    MODEL_PARAMS = {
        'GCN': {
            'channels': [
                NUMBER_OF_ATOMS,
                16,
                16
            ]
        },
        'MLP': {
            'channels': [
                16,
                10,
                1
            ]
        },
        'learning_rate': 0.0001,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_ds = ZINC(root=DATA_FOLDER, subset=True, split='train')
    val_ds = ZINC(root=DATA_FOLDER, subset=True, split='val')


    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)


    model = GCNNet(MODEL_PARAMS).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_PARAMS['learning_rate'])
    loss = nn.MSELoss()

    train_loss = []
    val_loss = []
    train_r2 = []
    val_r2 = []

    for epoch in range(3):
        print('EPOCH:', epoch+1)
        print('Training...')
        loss_value, r2_value = train(model, train_loader, optimizer, loss)
        train_loss.append(loss_value)
        train_r2.append(r2_value)

        print('Validating..')
        loss_value, r2_value = validate(model, val_loader, loss)
        val_loss.append(loss_value)
        val_r2.append(r2_value)


        print('Training Loss:', train_loss[-1])
        print('Training R2:', train_r2[-1])
        print('Validation Loss:', val_loss[-1])
        print('Validation R2:', val_r2[-1])


    RESULT_DICT = {
        'params': MODEL_PARAMS,
        'train_loss': train_loss,
        'test_loss': val_loss,
        'train_r2': train_r2,
        'test_r2': val_r2
    }

    with open('results.json', 'w') as f:
        json.dump(RESULT_DICT, f, indent=4)


