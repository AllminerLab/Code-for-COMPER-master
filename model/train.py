import torch
import torch.nn as nn
import torch.optim as optim
import linecache
import numpy as np
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InteractionData(Dataset):
    """
    Dataset that can either store all interaction data in memory or load it line
    by line when needed
    """

    def __init__(self, train_path_file, in_memory=True):
        self.in_memory = in_memory
        self.file = train_path_file
        self.num_interactions = 0
        self.interactions = []
        if in_memory:
            with open(self.file, "r") as f:
                for line in f:
                    self.interactions.append(eval(line.rstrip("\n")))
            self.num_interactions = len(self.interactions)
        else:
            with open(self.file, "r") as f:
                for line in f:
                    self.num_interactions += 1

    def __getitem__(self, idx):
        # load the specific interaction either from memory or from file line
        if self.in_memory:
            return self.interactions[idx]
        else:
            line = linecache.getline(self.file, idx+1)
            return eval(line.rstrip("\n"))

    def __len__(self):
        return self.num_interactions


def my_collate(batch):
    """
    Custom dataloader collate function since we have tuples of lists of paths
    """

    data = [line[0] for line in batch]
    user = [line[1] for line in batch]
    item = [line[2] for line in batch]
    val_len = [line[3] for line in batch]
    target = [line[4] for line in batch]
    user = torch.LongTensor(user)
    item = torch.LongTensor(item)
    val_len = torch.LongTensor(val_len)
    target = torch.Tensor(target)
    return [data, user, item, val_len, target]


def train(model, train_paths_file, valid_paths_file, batch_size, epochs, model_path, load_checkpoint, not_in_memory,
          lr, l2_reg):
    """
    -trains and outputs a model using the input data
    -formatted_data is a list of path lists, each of which consists of tuples of
    (path, label, path_length), where the path is padded to ensure same overall length
    """
    model = model.cuda()
    loss_function = nn.MSELoss(reduction='none')

    # l2 regularization is tuned from {10−5 , 10−4 , 10−3 , 10−2 }
    # Learning rate is found from {0.001, 0.002, 0.01, 0.02} with grid search
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

    if load_checkpoint:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # DataLoader used for batches
    interaction_data_train = InteractionData(train_paths_file, in_memory=not not_in_memory)
    train_loader = DataLoader(dataset=interaction_data_train, collate_fn=my_collate, batch_size=batch_size,
                              shuffle=True)
    interaction_data_valid = InteractionData(valid_paths_file, in_memory=not not_in_memory)
    valid_loader = DataLoader(dataset=interaction_data_valid, collate_fn=my_collate, batch_size=batch_size,
                              shuffle=False)

    for epoch in range(epochs):
        model, training_rmse, training_mae = training(model, optimizer, loss_function, train_loader, model_path)
        validating_rmse, validating_mae = predict(model, loss_function, valid_loader, model_path)
        print("Epoch: %d, Training_RMSE: %f, Training_MAE: %f, Validating_RMSE: %f, Validating_MAE: %f"
              % (epoch + 1, training_rmse, training_mae, validating_rmse, validating_mae))


def training(model, optimizer, loss_function, data_loader, model_path):
    rmse_metric = np.zeros(2)
    mae_metric = np.zeros(2)
    model.train()
    for interaction_batch, users, items, val_lens, targets in data_loader:
        # construct tensor of all paths in batch, tensor of all lengths, and tensor of interaction id
        paths = []
        lengths = []
        for inter_id, interaction_paths in enumerate(interaction_batch):
            for path, length in interaction_paths:
                paths.append(path)
                lengths.append(length)
        paths = torch.tensor(paths, dtype=torch.long, device='cuda')
        lengths = torch.tensor(lengths, dtype=torch.long, device='cuda')

        model.zero_grad()
        prediction_scores = model(paths, lengths, users.cuda(), items.cuda(),
                                  val_lens.cuda(), is_training=True).cuda()

        # Compute the loss, gradients, and update the parameters by calling .step()
        loss = loss_function(prediction_scores, targets.cuda())
        loss.sum().backward()
        optimizer.step()
        rmse_metric += (float(loss.sum()), len(targets))
        mae_metric += (float(torch.sum(abs(prediction_scores - targets.cuda()))), len(targets))

    mse = rmse_metric[0] / rmse_metric[1]
    rmse = math.sqrt(mse)
    mae = mae_metric[0] / mae_metric[1]

    # Save model to disk
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)

    return model, rmse, mae


def predict(model, loss_function, valid_loader, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    rmse_metric = np.zeros(2)
    mae_metric = np.zeros(2)
    for interaction_batch, users, items, val_lens, targets in valid_loader:
        # construct tensor of all paths in batch, tensor of all lengths, and tensor of interaction id
        paths = []
        lengths = []
        for inter_id, interaction_paths in enumerate(interaction_batch):
            for path, length in interaction_paths:
                paths.append(path)
                lengths.append(length)
        paths = torch.tensor(paths, dtype=torch.long, device='cuda')
        lengths = torch.tensor(lengths, dtype=torch.long, device='cuda')

        with torch.no_grad():
            # Run the forward pass
            prediction_scores = model(paths, lengths, users.cuda(), items.cuda(),
                                      val_lens.cuda(), is_training=False).cuda()
            # Compute the loss
            loss = loss_function(prediction_scores, targets.cuda())
        rmse_metric += (float(loss.sum()), len(targets))
        mae_metric += (float(torch.sum(abs(prediction_scores - targets.cuda()))), len(targets))

    mse = rmse_metric[0] / rmse_metric[1]
    rmse = math.sqrt(mse)
    mae = mae_metric[0] / mae_metric[1]
    return rmse, mae