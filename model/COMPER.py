import torch
import torch.nn as nn
import torch.nn.functional as F
import constants.consts as consts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class COMPER(nn.Module):

    def __init__(self, e_emb_dim, t_emb_dim, r_emb_dim, hidden_dim, attention_dim, e_vocab_size, t_vocab_size,
                 r_vocab_size, miu):
        super(COMPER, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.miu = miu

        self.entity_embeddings = nn.Embedding(e_vocab_size, e_emb_dim)
        self.type_embeddings = nn.Embedding(t_vocab_size, t_emb_dim)
        self.rel_embeddings = nn.Embedding(r_vocab_size, r_emb_dim)

        self.bias = nn.Embedding(e_vocab_size, 1)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(e_emb_dim + t_emb_dim + r_emb_dim, hidden_dim)

        # The attention parameters
        self.attention_W_q = nn.Linear(2 * e_emb_dim, attention_dim, bias=False)
        self.attention_W_k = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.attention_W_b = nn.Linear(attention_dim, 1, bias=False)

        # The linear layer that maps from hidden state space to label
        self.linear1 = nn.Linear(hidden_dim, 64)
        self.linear2 = nn.Linear(64, 1)
        """self.linear1 = nn.Linear(hidden_dim, 8)
        self.linear2 = nn.Linear(8, 4)
        self.linear3 = nn.Linear(4, 1)"""
        # self.Dropout = nn.Dropout(0.5)

    # def forward(self, paths, inter_ids, path_lengths, users, items, val_lens, batch_size, is_training):
        # transpose, so entities 1st row, types 2nd row, and relations 3nd (these are dim 1 and 2 since batch is 0)
        # this could just be the input if we want
    def forward(self, paths, lengths, users, items, val_lens, is_training):
        """paths = paths.reshape(-1, consts.MAX_PATH_LEN, 3)
        lengths = lengths.reshape(-1)"""

        inter_ids, _ = torch.arange(0, paths.shape[0] / consts.SAMPLES, device='cuda').repeat(consts.SAMPLES).sort()

        paths, inter_id, lengths, perm_idx = self.sort_batch(paths, inter_ids, lengths)
        sample_indexs = (lengths > 1).nonzero().squeeze(-1)
        drop_num = len(lengths) - len(sample_indexs)
        paths = paths[sample_indexs]
        lengths = lengths[sample_indexs]

        t_paths = torch.transpose(paths, 1, 2)

        # then concatenate embeddings, batch is index 0, so selecting along index 1
        # right now we do fetch embedding for padding tokens, but that these aren't used
        entity_embed = self.entity_embeddings(t_paths[:, 0, :])
        type_embed = self.type_embeddings(t_paths[:, 1, :])
        rel_embed = self.rel_embeddings(t_paths[:, 2, :])
        triplet_embed = torch.cat((entity_embed, type_embed, rel_embed), 2)  # concatenates lengthwise

        # we need dimensions to be input size x batch_size x embedding dim, so transpose first 2 dim
        batch_sec_embed = torch.transpose(triplet_embed, 0, 1)

        # pack padded sequences, so we don't do extra computation
        packed_embed = nn.utils.rnn.pack_padded_sequence(batch_sec_embed, lengths)

        # last_out is the output state before padding for each path, since we only want final output
        # self.lstm.flatten_parameters()
        packed_out, (last_out, _) = self.lstm(packed_embed)
        path_embedding = last_out[-1]
        path_embedding = torch.cat((path_embedding, torch.rand(drop_num, self.hidden_dim, device='cuda')), 0)
        # Get attention pooling of path_embedding over interaction id groups
        path_embedding = self.sort_path_embedding(path_embedding, perm_idx)
        path_embedding = path_embedding.reshape(-1, consts.SAMPLES, self.hidden_dim)

        users_embedding = self.entity_embeddings(users)
        items_embedding = self.entity_embeddings(items)
        queries = torch.cat((users_embedding, items_embedding), dim=1)
        sub_graph_embeddings = self.Attention(queries, path_embedding, path_embedding, val_lens, is_training)

        """start = True
        sub_graph_embeddings = torch.Tensor()
        for i in range(batch_size):
            # get ixs for this interaction
            inter_ixs = (inter_ids == i).nonzero().squeeze(1)

            # weighted pooled scores for this interaction
            query = self.entity_embeddings(torch.tensor((users[i], items[i])).to(device)).reshape(-1)

            sub_graph_embedding = self.Attention(path_embedding[inter_ixs], query, is_training)
        
            if start:
                # unsqueeze turns it into 2d tensor, so that we can concatenate along existing dim
                sub_graph_embeddings = sub_graph_embedding
                start = not start
            else:
                sub_graph_embeddings = torch.cat((sub_graph_embeddings, sub_graph_embedding), dim=0)"""

        # pass through linear layers
        """layer_1 = self.Dropout(F.relu(self.linear1(sub_graph_embeddings)))
        layer_2 = self.Dropout(F.relu(self.linear2(layer_1)))
        predict_scores = self.linear3(layer_2).squeeze(1)"""

        # predict_scores = self.linear2(self.Dropout(F.relu(self.linear1(sub_graph_embeddings))))
        predict_scores = self.linear2(F.relu(self.linear1(sub_graph_embeddings)))
        # predict_scores = self.linear1(sub_graph_embeddings).squeeze(1)

        b_u = self.bias(users)
        b_i = self.bias(items)
        output = predict_scores + b_u + b_i + self.miu

        return output.squeeze(-1)


    """def Attention(self, paths_embedding, query, is_training):
        features = self.attention_W_q(query) + self.attention_W_k(paths_embedding.unsqueeze(0))
        features = torch.tanh(features)
        weights = self.attention_W_b(features).squeeze(-1)
        attention_weights = F.softmax(weights, dim=1)
        sub_graph_embedding = torch.mm(attention_weights, paths_embedding).squeeze(1)
        # sub_graph_embedding = torch.mean(paths_embedding, 0).unsqueeze(0)

        return sub_graph_embedding"""


    def Attention(self, queries, keys, values, val_lens, is_training):
        queries, keys = self.attention_W_q(queries), self.attention_W_k(keys)
        # queries = (batch_size,num_hidden) keys = (batch_size,num_keys,num_hidden)
        features = queries.unsqueeze(1) + keys
        features = torch.tanh(features)
        scores = self.attention_W_b(features).squeeze(-1)  # (batch_size,num_keys)

        # mask
        mask = torch.arange((scores.shape[1]), dtype=torch.float32, device='cuda')[None, :] < val_lens[:, None]
        scores[~mask] = -1e6
        attention_weights = nn.functional.softmax(scores, dim=-1).unsqueeze(1)
        fusion_result = torch.bmm(attention_weights, values).squeeze(1)
        return fusion_result

    def sort_path_embedding(self, path_embedding, indexes):
        _, perm_idx = indexes.sort(0, descending=False)
        seq_tensor = path_embedding[perm_idx]
        return seq_tensor

    def sort_batch(self, batch, indexes, lengths):
        """
        sorts a batch of paths by path length, in decreasing order
        """
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        indexes_tensor = indexes[perm_idx]
        return seq_tensor.cuda(), indexes_tensor.cuda(), seq_lengths.cpu(), perm_idx.cuda()

    """def paths_split(self, interaction_batch):
        # construct tensor of all paths in batch, tensor of all lengths, and tensor of interaction id
        paths = []
        lengths = []
        inter_ids = []
        for inter_id, interaction_paths in enumerate(interaction_batch):
            for path, length in interaction_paths:
                paths.append(path)
                lengths.append(length)
            inter_ids.extend([inter_id for i in range(len(interaction_paths))])

        inter_ids = torch.tensor(inter_ids, dtype=torch.long)
        paths = torch.tensor(paths, dtype=torch.long)
        lengths = torch.tensor(lengths, dtype=torch.long)

        # sort based on path lengths, largest first, so that we can pack paths
        s_path_batch, s_inter_ids, s_lengths = self.sort_batch(paths, inter_ids, lengths)
        return s_path_batch.cuda(), s_inter_ids.cuda(), s_lengths.cpu()"""