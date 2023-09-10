import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from model import InteractionData, my_collate
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, test_paths_file, batch_size, model_path, not_in_memory):
    model = model.to(device)
    loss_function = nn.MSELoss(reduction='none')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # DataLoader used for batches
    interaction_data = InteractionData(test_paths_file, in_memory=not not_in_memory)
    data_loader = DataLoader(dataset=interaction_data, collate_fn=my_collate, batch_size=batch_size, shuffle=True)

    model.eval()
    rmse_metric = np.zeros(2)
    mae_metric = np.zeros(2)
    for interaction_batch, users, items, val_lens, targets in data_loader:
        # construct tensor of all paths in batch, tensor of all lengths, and tensor of interaction id
        paths = []
        lengths = []
        for inter_id, interaction_paths in enumerate(interaction_batch):
            for path, length in interaction_paths:
                paths.append(path)
                lengths.append(length)

        paths = torch.tensor(paths, dtype=torch.long, device=device)
        lengths = torch.tensor(lengths, dtype=torch.long, device=device)

        # sort based on path lengths, largest first, so that we can pack paths
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
