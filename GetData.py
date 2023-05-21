import random
import numpy as np
import pandas as pd
import torch
EPSILON = 1e-10
def load_pretrain_data(batch_size = 64):
    batch_size = 64

    random.seed(17)
    test_ind = set()

    pre_train_size = 50000

    while len(test_ind) < 10000: 
        test_ind.add(random.randint(0, pre_train_size-1))

    features =[]
    labels = []

    with open("data/task4_hr35z9/pretrain_features.csv", 'r') as f:
        for row in f:
            features.append(row)

    with open("data/task4_hr35z9/pretrain_labels.csv", 'r') as f:
        for row in f:
            labels.append(row)

    # remove header
    features = features[1:]
    labels = labels[1:]

    # first try to note use representation of the molecules, only the extracted features
    features = [list(map(float,row.split(',')[2:])) for row in features]
    labels = [float(row.split(',')[1]) for row in labels]

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []


    for i in range(len(features)):
        if i in test_ind:
            test_features.append(features[i])
            test_labels.append(labels[i])
        else:
            train_features.append(features[i])
            train_labels.append(labels[i])

    # does not seem to make sense to normalize the data since it is very sparse
    # normalize train_features
    # train_features = (train_features - np.mean(train_features, axis=0)) / (np.std(train_features, axis=0)+EPSILON)

    # normalize test_features
    # test_features = (test_features - np.mean(test_features, axis=0)) / (np.std(test_features, axis=0)+EPSILON)

    # convert into tensor dataset
    train_features = torch.tensor(train_features, dtype=torch.float)
    train_labels = torch.tensor(train_labels, dtype=torch.float)
    test_features = torch.tensor(test_features, dtype=torch.float)
    test_labels = torch.tensor(test_labels, dtype=torch.float)

    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels) 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader