import pandas as pd
import numpy as np
import gzip
import json
import pickle
import argparse
from tqdm import tqdm
import torch
from scipy.sparse import csr_matrix
from path_extraction import find_paths_user_to_items
from data.format import format_paths
import random
import sys
from os import path, mkdir
sys.path.append(path.dirname(path.abspath('../constants')))
import constants.consts as consts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',
                        default=consts.DATA_FILE,
                        help='Path to the json.gz file containing rating information')
    parser.add_argument('--rating_data_file',
                        default='rating_data.csv',
                        help='Path to the csv file containing rating data')
    parser.add_argument('--rating_train_data_file',
                        default='rating_train.csv',
                        help='Path to the csv file containing training data')
    parser.add_argument('--rating_valid_data_file',
                        default='rating_valid.csv',
                        help='Path to the csv file containing validating data')
    parser.add_argument('--rating_test_data_file',
                        default='rating_test.csv',
                        help='Path to the csv file containing testing data')
    parser.add_argument('--split_data',
                        default=False,
                        help='whether to split the data')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.3,
                        help='alpha for constructing similarity')

    return parser.parse_args(args=[])


def read_rating_data(data_file):
    rating_data = []
    with gzip.open(data_file) as f:
        for l in f:
            rating_data.append(json.loads(l.strip()))
    rating_data_df = pd.DataFrame(rating_data)
    rating_data_df = rating_data_df[['reviewerID', 'asin', 'overall']]
    rating_data_df.columns = ['user_id', 'item_id', 'ratings']

    # save data
    rating_data_df.to_csv(consts.DATASET_DIR + 'rating_data.csv', index=False)


def train_valid_test_split(rating_data_file, dir, training_data_file, validating_data_file, testing_data_file):
    with open(consts.DATASET_DIR + rating_data_file, 'r', encoding='utf8') as fp:
        data = pd.read_csv(fp)

    # data split
    valid_test = np.random.choice(len(data), size=int(0.2 * len(data)), replace=False)
    valid_test_idx = np.zeros(len(data), dtype=bool)
    valid_test_idx[valid_test] = True
    rating_valid_test = data[valid_test_idx]
    rating_train = data[~valid_test_idx]

    num_ratings_valid_test = rating_valid_test.shape[0]
    test = np.random.choice(num_ratings_valid_test, size=int(0.50 * num_ratings_valid_test), replace=False)
    test_idx = np.zeros(num_ratings_valid_test, dtype=bool)
    test_idx[test] = True
    rating_test = rating_valid_test[test_idx]
    rating_valid = rating_valid_test[~test_idx]

    print("The number of training ratings is %d" % (len(rating_train)))
    print("The number of validating ratings is %d" % (len(rating_valid)))
    print("The number of testing ratings is %d" % (len(rating_test)))

    # save data
    rating_train.to_csv(dir + training_data_file, index=False)
    rating_valid.to_csv(dir + validating_data_file, index=False)
    rating_test.to_csv(dir + testing_data_file, index=False)


def create_directory(dir):
    print("Creating directory %s" % dir)
    try:
        mkdir(dir)
    except FileExistsError:
        print("Directory already exists")


def relation_cons(data_file, training_data_file, validating_data_file, testing_data_file, alpha, export_dir):
    """
    return: Write out python dictionaries for the edge of graph
    """

    with open(data_file, 'r', encoding='utf8') as fp:
        data = pd.read_csv(fp)
    with open(training_data_file, 'r', encoding='utf8') as fp:
        rating_train = pd.read_csv(fp)
    with open(validating_data_file, 'r', encoding='utf8') as fp:
        rating_valid = pd.read_csv(fp)
    with open(testing_data_file, 'r', encoding='utf8') as fp:
        rating_test = pd.read_csv(fp)

    user_item_dict = data.set_index('user_id').groupby('user_id')['item_id'].apply(list).to_dict()
    item_user_dict = data.set_index('item_id').groupby('item_id')['user_id'].apply(list).to_dict()
    train_user_item_dict = rating_train.set_index('user_id').groupby('user_id')['item_id'].apply(list).to_dict()
    train_item_user_dict = rating_train.set_index('item_id').groupby('item_id')['user_id'].apply(list).to_dict()
    valid_user_item_dict = rating_valid.set_index('user_id').groupby('user_id')['item_id'].apply(list).to_dict()
    valid_item_user_dict = rating_valid.set_index('item_id').groupby('item_id')['user_id'].apply(list).to_dict()
    test_user_item_dict = rating_test.set_index('user_id').groupby('user_id')['item_id'].apply(list).to_dict()
    test_item_user_dict = rating_test.set_index('item_id').groupby('item_id')['user_id'].apply(list).to_dict()
    with open(export_dir + consts.USER_ITEM_DICT, 'wb') as handle:
        pickle.dump(user_item_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + consts.ITEM_USER_DICT, 'wb') as handle:
        pickle.dump(item_user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + consts.TRAIN_USER_ITEM_DICT, 'wb') as handle:
        pickle.dump(train_user_item_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + consts.TRAIN_ITEM_USER_DICT, 'wb') as handle:
        pickle.dump(train_item_user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + consts.VALID_USER_ITEM_DICT, 'wb') as handle:
        pickle.dump(valid_user_item_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + consts.VALID_ITEM_USER_DICT, 'wb') as handle:
        pickle.dump(valid_item_user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + consts.TEST_USER_ITEM_DICT, 'wb') as handle:
        pickle.dump(test_user_item_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + consts.TEST_ITEM_USER_DICT, 'wb') as handle:
        pickle.dump(test_item_user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # id_count
    users_id_num = data["user_id"].unique().shape[0]
    items_id_num = data["item_id"].unique().shape[0]
    ratings_num = data["ratings"].shape[0]
    print("the number of users: ", users_id_num)
    print("the number of items: ", items_id_num)
    print("the number of ratings: ", ratings_num)

    # from id to num
    # this id just used to construct the rating matrix
    user2id = dict((uid, i) for (i, uid) in enumerate(data["user_id"].unique()))
    item2id = dict((sid, i) for (i, sid) in enumerate(data["item_id"].unique()))
    id2user = dict((i, uid) for (i, uid) in enumerate(data["user_id"].unique()))
    id2item = dict((i, sid) for (i, sid) in enumerate(data["item_id"].unique()))
    user_id = list(map(lambda x: user2id[x], rating_train['user_id']))
    item_id = list(map(lambda x: item2id[x], rating_train['item_id']))
    rating_train['user_id'] = user_id
    rating_train['item_id'] = item_id

    # construct rating matrix
    # this matrix is used to construct the similarity of user pairs and item pairs
    rating_matrix_arr = np.zeros((users_id_num, items_id_num), dtype=float)
    for i in range(len(rating_train)):
        rating_matrix_arr[int(rating_train.iloc[i][0]), int(rating_train.iloc[i][1])] = rating_train.iloc[i][2]
    rating_matrix = torch.tensor(rating_matrix_arr, dtype=torch.float)
    # construct similarity
    user_sim_dict = {}
    user_sim_nums = 0
    user_sim_matrix = torch.cov(rating_matrix) / torch.sqrt(torch.mm(torch.var(rating_matrix, 1).unsqueeze(1),
                                                                     torch.var(rating_matrix, 1).unsqueeze(0))) >= alpha
    user_sparse = csr_matrix(user_sim_matrix)
    user_sim_pairs = user_sparse.todok().keys()
    for pair in user_sim_pairs:
        if pair[0] != pair[1]:
            user_sim_nums += 1
            user_i, user_j = id2user[pair[0]], id2user[pair[1]]
            if user_i not in user_sim_dict:
                user_sim_dict[user_i] = []
            user_sim_dict[user_i].append(user_j)
    print("user similar pair numbers: ", int(user_sim_nums/2))
    with open(export_dir + consts.USER_SIM_DICT, 'wb') as handle:
        pickle.dump(user_sim_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    item_sim_dict = {}
    item_sim_nums = 0
    item_sim_matrix = torch.cov(rating_matrix.transpose(1, 0))/torch.sqrt(torch.mm(torch.var(rating_matrix, 0).unsqueeze(1),
                                                                torch.var(rating_matrix, 0).unsqueeze(0))) >= alpha
    item_sparse = csr_matrix(item_sim_matrix)
    item_sim_pairs = item_sparse.todok().keys()
    for pair in item_sim_pairs:
        if pair[0] != pair[1]:
            item_sim_nums += 1
            item_i, item_j = id2item[pair[0]], id2item[pair[1]]
            if item_i not in item_sim_dict:
                item_sim_dict[item_i] = []
            item_sim_dict[item_i].append(item_j)
    print("item similar pair numbers: ", int(item_sim_nums/2))
    with open(export_dir + consts.ITEM_SIM_DICT, 'wb') as handle:
        pickle.dump(item_sim_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def data_prep(data_file, export_dir):
    """
    return: Write out python dictionaries for the edges of graph
    """
    with open(data_file, 'r', encoding='utf8') as fp:
        data = pd.read_csv(fp)

    # train_user_item_k.dict
    # dict where key = a user, value = list of item be ranked k by this user
    user_item_1_dict = data[data["ratings"] == 1].set_index('user_id').groupby('user_id')['item_id'].apply(
        list).to_dict()
    user_item_2_dict = data[data["ratings"] == 2].set_index('user_id').groupby('user_id')['item_id'].apply(
        list).to_dict()
    user_item_3_dict = data[data["ratings"] == 3].set_index('user_id').groupby('user_id')['item_id'].apply(
        list).to_dict()
    user_item_4_dict = data[data["ratings"] == 4].set_index('user_id').groupby('user_id')['item_id'].apply(
        list).to_dict()
    user_item_5_dict = data[data["ratings"] == 5].set_index('user_id').groupby('user_id')['item_id'].apply(
        list).to_dict()
    with open(export_dir + 'train_' + consts.USER_ITEM_1_DICT, 'wb') as handle:
        pickle.dump(user_item_1_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'train_' + consts.USER_ITEM_2_DICT, 'wb') as handle:
        pickle.dump(user_item_2_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'train_' + consts.USER_ITEM_3_DICT, 'wb') as handle:
        pickle.dump(user_item_3_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'train_' + consts.USER_ITEM_4_DICT, 'wb') as handle:
        pickle.dump(user_item_4_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'train_' + consts.USER_ITEM_5_DICT, 'wb') as handle:
        pickle.dump(user_item_5_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # train_item_user_k.dict
    # dict where key = a item, value = list of user rank k to this item
    item_user_1_dict = data[data["ratings"] == 1].set_index('item_id').groupby('item_id')['user_id'].apply(
        list).to_dict()
    item_user_2_dict = data[data["ratings"] == 2].set_index('item_id').groupby('item_id')['user_id'].apply(
        list).to_dict()
    item_user_3_dict = data[data["ratings"] == 3].set_index('item_id').groupby('item_id')['user_id'].apply(
        list).to_dict()
    item_user_4_dict = data[data["ratings"] == 4].set_index('item_id').groupby('item_id')['user_id'].apply(
        list).to_dict()
    item_user_5_dict = data[data["ratings"] == 5].set_index('item_id').groupby('item_id')['user_id'].apply(
        list).to_dict()
    with open(export_dir + 'train_' + consts.ITEM_USER_1_DICT, 'wb') as handle:
        pickle.dump(item_user_1_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'train_' + consts.ITEM_USER_2_DICT, 'wb') as handle:
        pickle.dump(item_user_2_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'train_' + consts.ITEM_USER_3_DICT, 'wb') as handle:
        pickle.dump(item_user_3_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'train_' + consts.ITEM_USER_4_DICT, 'wb') as handle:
        pickle.dump(item_user_4_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'train_' + consts.ITEM_USER_5_DICT, 'wb') as handle:
        pickle.dump(item_user_5_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def ix_mapping(data_file, mapping_export_dir):
    pad_token = consts.PAD_TOKEN
    type_to_ix = {'user': consts.USER_TYPE, 'item': consts.ITEM_TYPE, pad_token: consts.PAD_TYPE}
    relation_to_ix = {'user_sim': consts.USER_SIM_REL, 'item_sim': consts.ITEM_SIM_REL,
                      'user_item_1': consts.USER_ITEM_1_REL,
                      'user_item_2': consts.USER_ITEM_2_REL, 'user_item_3': consts.USER_ITEM_3_REL,
                      'user_item_4': consts.USER_ITEM_4_REL,
                      'user_item_5': consts.USER_ITEM_5_REL, 'item_user_1': consts.ITEM_USER_1_REL,
                      'item_user_2': consts.ITEM_USER_2_REL,
                      'item_user_3': consts.ITEM_USER_3_REL, 'item_user_4': consts.ITEM_USER_4_REL,
                      'item_user_5': consts.ITEM_USER_5_REL,
                      '#END_RELATION': consts.END_REL, pad_token: consts.PAD_REL}

    # entity vocab set is combination of users and items
    with open(data_file, 'r', encoding='utf8') as fp:
        data = pd.read_csv(fp)

    users = set(data["user_id"].unique())
    items = set(data["item_id"].unique())

    # Id-ix mappings
    entity_to_ix = {(user, consts.USER_TYPE): ix for ix, user in enumerate(users)}
    entity_to_ix.update({(item, consts.ITEM_TYPE): ix + len(users) for ix, item in enumerate(items)})
    entity_to_ix[pad_token] = len(entity_to_ix)

    # Ix-id mappings
    ix_to_type = {v: k for k, v in type_to_ix.items()}
    ix_to_relation = {v: k for k, v in relation_to_ix.items()}
    ix_to_entity = {v: k for k, v in entity_to_ix.items()}

    # Export mappings
    # eg. Musical_Instruments_ix_mapping/type_to_ix.dict
    with open(mapping_export_dir + consts.TYPE_TO_IX, 'wb') as handle:
        pickle.dump(type_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + consts.RELATION_TO_IX, 'wb') as handle:
        pickle.dump(relation_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + consts.ENTITY_TO_IX, 'wb') as handle:
        pickle.dump(entity_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + consts.IX_TO_TYPE, 'wb') as handle:
        pickle.dump(ix_to_type, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + consts.IX_TO_RELATION, 'wb') as handle:
        pickle.dump(ix_to_relation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(mapping_export_dir + consts.IX_TO_ENTITY, 'wb') as handle:
        pickle.dump(ix_to_entity, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_to_ids(entity_to_ix, rel_dict, start_type, end_type):
    new_rel = {}
    for key, values in rel_dict.items():
        key_id = entity_to_ix[(key, start_type)]
        value_ids = []
        for val in values:
            value_ids.append(entity_to_ix[(val, end_type)])
        new_rel[key_id] = value_ids
    return new_rel


def ix_update(import_dir, mapping_dir, export_dir):
    with open(mapping_dir + consts.ENTITY_TO_IX, 'rb') as handle:
        entity_to_ix = pickle.load(handle)
    with open(import_dir + consts.USER_SIM_DICT, 'rb') as handle:
        user_sim_dict = pickle.load(handle)
    with open(import_dir + consts.ITEM_SIM_DICT, 'rb') as handle:
        item_sim_dict = pickle.load(handle)
    with open(import_dir + consts.USER_ITEM_DICT, 'rb') as handle:
        user_item_dict = pickle.load(handle)
    with open(import_dir + consts.ITEM_USER_DICT, 'rb') as handle:
        item_user_dict = pickle.load(handle)
    with open(import_dir + consts.TRAIN_USER_ITEM_DICT, 'rb') as handle:
        train_user_item_dict = pickle.load(handle)
    with open(import_dir + consts.TRAIN_ITEM_USER_DICT, 'rb') as handle:
        train_item_user_dict = pickle.load(handle)
    with open(import_dir + consts.VALID_USER_ITEM_DICT, 'rb') as handle:
        valid_user_item_dict = pickle.load(handle)
    with open(import_dir + consts.VALID_ITEM_USER_DICT, 'rb') as handle:
        valid_item_user_dict = pickle.load(handle)
    with open(import_dir + consts.TEST_USER_ITEM_DICT, 'rb') as handle:
        test_user_item_dict = pickle.load(handle)
    with open(import_dir + consts.TEST_ITEM_USER_DICT, 'rb') as handle:
        test_item_user_dict = pickle.load(handle)

    # mapping id to ix
    user_sim_ix = convert_to_ids(entity_to_ix, user_sim_dict, consts.USER_TYPE, consts.USER_TYPE)
    item_sim_ix = convert_to_ids(entity_to_ix, item_sim_dict, consts.ITEM_TYPE, consts.ITEM_TYPE)
    user_item_ix = convert_to_ids(entity_to_ix, user_item_dict, consts.USER_TYPE, consts.ITEM_TYPE)
    item_user_ix = convert_to_ids(entity_to_ix, item_user_dict, consts.ITEM_TYPE, consts.USER_TYPE)
    train_user_item_ix = convert_to_ids(entity_to_ix, train_user_item_dict, consts.USER_TYPE, consts.ITEM_TYPE)
    train_item_user_ix = convert_to_ids(entity_to_ix, train_item_user_dict, consts.ITEM_TYPE, consts.USER_TYPE)
    valid_user_item_ix = convert_to_ids(entity_to_ix, valid_user_item_dict, consts.USER_TYPE, consts.ITEM_TYPE)
    valid_item_user_ix = convert_to_ids(entity_to_ix, valid_item_user_dict, consts.ITEM_TYPE, consts.USER_TYPE)
    test_user_item_ix = convert_to_ids(entity_to_ix, test_user_item_dict, consts.USER_TYPE, consts.ITEM_TYPE)
    test_item_user_ix = convert_to_ids(entity_to_ix, test_item_user_dict, consts.ITEM_TYPE, consts.USER_TYPE)
    with open(export_dir + 'ix_' + consts.USER_SIM_DICT, 'wb') as handle:
        pickle.dump(user_sim_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'ix_' + consts.ITEM_SIM_DICT, 'wb') as handle:
        pickle.dump(item_sim_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'ix_' + consts.USER_ITEM_DICT, 'wb') as handle:
        pickle.dump(user_item_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'ix_' + consts.ITEM_USER_DICT, 'wb') as handle:
        pickle.dump(item_user_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'ix_' + consts.TRAIN_USER_ITEM_DICT, 'wb') as handle:
        pickle.dump(train_user_item_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'ix_' + consts.TRAIN_ITEM_USER_DICT, 'wb') as handle:
        pickle.dump(train_item_user_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'ix_' + consts.VALID_USER_ITEM_DICT, 'wb') as handle:
        pickle.dump(valid_user_item_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'ix_' + consts.VALID_ITEM_USER_DICT, 'wb') as handle:
        pickle.dump(valid_item_user_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'ix_' + consts.TEST_USER_ITEM_DICT, 'wb') as handle:
        pickle.dump(test_user_item_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(export_dir + 'ix_' + consts.TEST_ITEM_USER_DICT, 'wb') as handle:
        pickle.dump(test_item_user_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for k in range(5):
        data_file = 'train_user_item_' + str(k + 1) + '.dict'
        with open(import_dir + data_file, 'rb') as handle:
            user_item_dict = pickle.load(handle)
        user_item_ix = convert_to_ids(entity_to_ix, user_item_dict, consts.USER_TYPE, consts.ITEM_TYPE)
        with open(export_dir + 'ix_' + data_file, 'wb') as handle:
            pickle.dump(user_item_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for k in range(5):
        data_file = 'train_item_user_' + str(k + 1) + '.dict'
        with open(import_dir + data_file, 'rb') as handle:
            item_user_dict = pickle.load(handle)
        item_user_ix = convert_to_ids(entity_to_ix, item_user_dict, consts.ITEM_TYPE, consts.USER_TYPE)
        with open(export_dir + 'ix_' + data_file, 'wb') as handle:
            pickle.dump(item_user_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)


def sample_paths(paths, samples):
    index_list = list(range(len(paths)))
    random.shuffle(index_list)
    indices = index_list[:samples]
    return [paths[i] for i in indices]


def construct_paths(data_file, training_data_file, validating_data_file, testing_data_file, import_dir, path_dir,
                    mapping_dir, samples):
    """
    Constructs paths from the target user to the target item
    """
    create_directory(path_dir)
    train_path_file = open(path_dir + consts.TRAIN_PATH_FILE, 'w')
    valid_path_file = open(path_dir + consts.VALID_PATH_FILE, 'w')
    test_path_file = open(path_dir + consts.TEST_PATH_FILE, 'w')

    # load data
    with open(data_file, 'r', encoding='utf8') as handle:
        rating_data = pd.read_csv(handle)
    with open(training_data_file, 'r', encoding='utf8') as handle:
        rating_train = pd.read_csv(handle)
    with open(validating_data_file, 'r', encoding='utf8') as handle:
        rating_valid = pd.read_csv(handle)
    with open(testing_data_file, 'r', encoding='utf8') as handle:
        rating_test = pd.read_csv(handle)
    with open(mapping_dir + consts.ENTITY_TO_IX, 'rb') as handle:
        entity_to_ix = pickle.load(handle)
    with open(mapping_dir + consts.TYPE_TO_IX, 'rb') as handle:
        type_to_ix = pickle.load(handle)
    with open(mapping_dir + consts.RELATION_TO_IX, 'rb') as handle:
        relation_to_ix = pickle.load(handle)
    with open(mapping_dir + consts.IX_TO_ENTITY, 'rb') as handle:
        ix_to_entity = pickle.load(handle)
    with open(import_dir + 'ix_' + consts.USER_SIM_DICT, 'rb') as handle:
        user_sim = pickle.load(handle)
    with open(import_dir + 'ix_' + consts.ITEM_SIM_DICT, 'rb') as handle:
        item_sim = pickle.load(handle)
    with open(import_dir + 'ix_' + consts.USER_ITEM_DICT, 'rb') as handle:
        user_item = pickle.load(handle)
    with open(import_dir + 'ix_train_' + consts.USER_ITEM_1_DICT, 'rb') as handle:
        user_item_1 = pickle.load(handle)
    with open(import_dir + 'ix_train_' + consts.USER_ITEM_2_DICT, 'rb') as handle:
        user_item_2 = pickle.load(handle)
    with open(import_dir + 'ix_train_' + consts.USER_ITEM_3_DICT, 'rb') as handle:
        user_item_3 = pickle.load(handle)
    with open(import_dir + 'ix_train_' + consts.USER_ITEM_4_DICT, 'rb') as handle:
        user_item_4 = pickle.load(handle)
    with open(import_dir + 'ix_train_' + consts.USER_ITEM_5_DICT, 'rb') as handle:
        user_item_5 = pickle.load(handle)
    with open(import_dir + 'ix_train_' + consts.ITEM_USER_1_DICT, 'rb') as handle:
        item_user_1 = pickle.load(handle)
    with open(import_dir + 'ix_train_' + consts.ITEM_USER_2_DICT, 'rb') as handle:
        item_user_2 = pickle.load(handle)
    with open(import_dir + 'ix_train_' + consts.ITEM_USER_3_DICT, 'rb') as handle:
        item_user_3 = pickle.load(handle)
    with open(import_dir + 'ix_train_' + consts.ITEM_USER_4_DICT, 'rb') as handle:
        item_user_4 = pickle.load(handle)
    with open(import_dir + 'ix_train_' + consts.ITEM_USER_5_DICT, 'rb') as handle:
        item_user_5 = pickle.load(handle)

    # trackers for statistics
    train_paths_not_found = 0
    valid_paths_not_found = 0
    test_paths_not_found = 0
    total_interactions = 0
    avg_num_paths = 0

    for user, items in tqdm(list(user_item.items())):
        total_interactions += len(items)
        item_to_paths = None

        for item in items:
            if item_to_paths is None:
                item_to_paths = find_paths_user_to_items(user, user_sim, item_sim, user_item_1, user_item_2,
                                                         user_item_3, user_item_4, user_item_5, item_user_1,
                                                         item_user_2, item_user_3,item_user_4, item_user_5, 2, 20)
                item_to_paths_len3 = find_paths_user_to_items(user, user_sim, item_sim, user_item_1, user_item_2,
                                                              user_item_3, user_item_4, user_item_5, item_user_1,
                                                              item_user_2, item_user_3, item_user_4, item_user_5, 3, 10)
                item_to_paths_len4 = find_paths_user_to_items(user, user_sim, item_sim, user_item_1, user_item_2,
                                                              user_item_3, user_item_4, user_item_5, item_user_1,
                                                              item_user_2, item_user_3, item_user_4, item_user_5, 4, 5)
                """item_to_paths_len5 = find_paths_user_to_items(user, user_sim, item_sim, user_item_1, user_item_2,
                                                              user_item_3, user_item_4, user_item_5, item_user_1,
                                                              item_user_2, item_user_3, item_user_4, item_user_5, 5, 3)"""
                for i in item_to_paths_len3.keys():
                    item_to_paths[i].extend(item_to_paths_len3[i])
                for i in item_to_paths_len4.keys():
                    item_to_paths[i].extend(item_to_paths_len4[i])
                """for i in item_to_paths_len5.keys():
                    item_to_paths[i].extend(item_to_paths_len5[i])"""

            # add paths for interaction
            item_paths = item_to_paths[item]
            item_paths = sample_paths(item_paths, samples)
            rating = float(rating_data.loc[rating_data.user_id == ix_to_entity[user][0]].loc[
                               rating_data.item_id == ix_to_entity[item][0]].ratings.values[0])
            if len(item_paths) > 0:
                interaction = (format_paths(item_paths, entity_to_ix, type_to_ix, relation_to_ix, samples), user, item, len(item_paths), rating)
                if ix_to_entity[item][0] in rating_train.loc[rating_train.user_id == ix_to_entity[user][0]]["item_id"].unique():
                    train_path_file.write(repr(interaction) + "\n")
                elif ix_to_entity[item][0] in rating_valid.loc[rating_valid.user_id == ix_to_entity[user][0]]["item_id"].unique():
                    valid_path_file.write(repr(interaction) + "\n")
                elif ix_to_entity[item][0] in rating_test.loc[rating_test.user_id == ix_to_entity[user][0]]["item_id"].unique():
                    test_path_file.write(repr(interaction) + "\n")
                avg_num_paths += len(item_paths)
            else:
                padding_path = [[[user, consts.USER_TYPE, consts.PAD_REL], [item, consts.ITEM_TYPE, consts.END_REL]]]
                interaction = (format_paths(padding_path, entity_to_ix, type_to_ix, relation_to_ix, samples), user, item, len(padding_path), rating)
                if ix_to_entity[item][0] in rating_train.loc[rating_train.user_id == ix_to_entity[user][0]]["item_id"].unique():
                    train_paths_not_found += 1
                    train_path_file.write(repr(interaction) + "\n")
                elif ix_to_entity[item][0] in rating_valid.loc[rating_valid.user_id == ix_to_entity[user][0]]["item_id"].unique():
                    valid_paths_not_found += 1
                    valid_path_file.write(repr(interaction) + "\n")
                elif ix_to_entity[item][0] in rating_test.loc[rating_test.user_id == ix_to_entity[user][0]]["item_id"].unique():
                    test_paths_not_found += 1
                    test_path_file.write(repr(interaction) + "\n")
                continue

    avg_num_paths = avg_num_paths / (
                total_interactions - train_paths_not_found - valid_paths_not_found - test_paths_not_found)

    print("number of paths attempted to find:", total_interactions)
    print("number of train paths not found:", train_paths_not_found)
    print("number of valid paths not found:", valid_paths_not_found)
    print("number of test paths not found:", test_paths_not_found)
    print("avg num paths per interaction:", avg_num_paths)

    train_path_file.close()
    valid_path_file.close()
    test_path_file.close()


def main():
    print("Data preparation:")
    args = parse_args()
    print("Forming knowledge graph...")
    create_directory(consts.DATA_DIR)

    # train_valid_test_split
    if args.split_data:
        # read data
        # read_rating_data(consts.DATASET_DIR + args.data_file)
        train_valid_test_split(args.rating_data_file, consts.DATASET_DIR, args.rating_train_data_file,
                               args.rating_valid_data_file, args.rating_test_data_file)

    relation_cons(consts.DATASET_DIR + args.rating_data_file, consts.DATASET_DIR + args.rating_train_data_file,
                  consts.DATASET_DIR + args.rating_valid_data_file, consts.DATASET_DIR + args.rating_test_data_file,
                  args.alpha, consts.DATA_DIR)

    data_prep(consts.DATASET_DIR + args.rating_train_data_file, consts.DATA_DIR)

    print("Mapping ids to indices...")
    create_directory(consts.DATA_IX_DIR)
    create_directory(consts.DATA_IX_MAPPING_DIR)
    ix_mapping(consts.DATASET_DIR + args.rating_data_file, consts.DATA_IX_MAPPING_DIR)
    ix_update(consts.DATA_DIR, consts.DATA_IX_MAPPING_DIR, consts.DATA_IX_DIR)

    print("Constructing paths from user to item...")
    construct_paths(consts.DATASET_DIR + args.rating_data_file, consts.DATASET_DIR + args.rating_train_data_file,
                    consts.DATASET_DIR + args.rating_valid_data_file, consts.DATASET_DIR + args.rating_test_data_file,
                    consts.DATA_IX_DIR, consts.PATH_DATA_DIR, consts.DATA_IX_MAPPING_DIR, consts.SAMPLES)


if __name__ == '__main__':
    main()


