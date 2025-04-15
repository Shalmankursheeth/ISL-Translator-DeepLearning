import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from proto_model import ProtoNet
from utils import load_gesture_data
import random

class GestureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

def get_prototypes(support_x, support_y):
    prototypes = []
    for cls in torch.unique(support_y):
        class_feats = support_x[support_y == cls]
        proto = class_feats.mean(dim=0)
        prototypes.append(proto)
    return torch.stack(prototypes)

def train_proto():
    data, labels, class_map = load_gesture_data(r"D:\College\Sign\proto\data1")
    dataset = GestureDataset(data, labels)

    model = ProtoNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for episode in range(500):
        N_way = 3
        K_shot = 1
        Q_queries = 1

        # ✅ Only use classes that have enough samples
        valid_classes = [cls for cls in set(labels) if np.sum(labels == cls) >= (K_shot + Q_queries)]

        # ⚠️ If not enough valid classes, skip the episode
        if len(valid_classes) < N_way:
            print(f"Skipping episode {episode}: Not enough valid classes")
            continue

        classes = random.sample(valid_classes, N_way)

        support_x, support_y, query_x, query_y = [], [], [], []

        for i, cls in enumerate(classes):
            indices = np.where(labels == cls)[0]
            samples = np.random.choice(indices, K_shot + Q_queries, replace=False)
            support_x.extend(data[samples[:K_shot]])
            query_x.extend(data[samples[K_shot:]])
            support_y.extend([i] * K_shot)
            query_y.extend([i] * Q_queries)

        support_x = torch.tensor(support_x, dtype=torch.float32)
        query_x = torch.tensor(query_x, dtype=torch.float32)
        support_y = torch.tensor(support_y)
        query_y = torch.tensor(query_y)

        emb_support = model(support_x)
        emb_query = model(query_x)
        prototypes = get_prototypes(emb_support, support_y)

        # Compute distances
        dists = torch.cdist(emb_query, prototypes)
        pred = torch.argmin(dists, dim=1)
        acc = (pred == query_y).float().mean()

        loss = F.cross_entropy(-dists, query_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f"[Episode {episode}] Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")

    torch.save(model.state_dict(), "proto_gesture.pth")
    print("Training Done!")




if __name__ == "__main__":
    train_proto()
