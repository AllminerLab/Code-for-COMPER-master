import pickle
import argparse
import random
import numpy as np
import torch.nn as nn
import pandas as pd

import constants.consts as consts
from model import COMPER, train, test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        default=True,
                        action='store_true',
                        help='whether to train the model')
    parser.add_argument('--eval',
                        default=True,
                        action='store_true',
                        help='whether to evaluate the model')
    parser.add_argument('--model',
                        type=str,
                        default='model.pt',
                        help='name to save or load model from')
    parser.add_argument('--load_checkpoint',
                        default=False,
                        action='store_true',
                        help='whether to load the current model state before training ')
    parser.add_argument('-e', '--epochs',
                        type=int,
                        default=5,
                        help='number of epochs for training model')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=256,
                        help='batch_size')
    parser.add_argument('--not_in_memory',
                        default=False,
                        action='store_true',
                        help='denotes that the path data does not fit in memory')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='learning rate')
    parser.add_argument('--l2_reg',
                        type=float,
                        default=0.01,
                        help='l2 regularization coefficent')
    parser.add_argument('--np_baseline',
                        default=False,
                        action='store_true',
                        help='Run the model with the number of path baseline if True')

    return parser.parse_args()


def load_string_to_ix_dicts():
    """
    Loads the dictionaries mapping entity, relation, and type to id
    """
    data_path = 'data/' + consts.DATA_IX_MAPPING_DIR

    with open(data_path + consts.TYPE_TO_IX, 'rb') as handle:
        type_to_ix = pickle.load(handle)
    with open(data_path + consts.RELATION_TO_IX, 'rb') as handle:
        relation_to_ix = pickle.load(handle)
    with open(data_path + consts.ENTITY_TO_IX, 'rb') as handle:
        entity_to_ix = pickle.load(handle)

    return type_to_ix, relation_to_ix, entity_to_ix


def get_miu(data_file):
    with open(data_file, 'r', encoding='utf8') as fp:
        training_data = pd.read_csv(fp)
    miu = np.mean(training_data.iloc[:, 2])
    return miu

def main():
    """
    Main function for model testing and training
    """
    print("Main Loaded")
    random.seed(1000)
    args = parse_args()
    model_path = "model/" + args.model

    t_to_ix, r_to_ix, e_to_ix = load_string_to_ix_dicts()

    miu = get_miu('data/' + consts.DATASET_DIR + 'rating_train.csv')
    model = COMPER(consts.ENTITY_EMB_DIM, consts.TYPE_EMB_DIM, consts.REL_EMB_DIM, consts.HIDDEN_DIM,
                   consts.ATTENTION_DIM, len(e_to_ix), len(t_to_ix), len(r_to_ix), miu)

    if args.train:
        print("Training Starting")

        # 初始化
        for m in model.children():
            if isinstance(m, (nn.Embedding, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        # load paths from disk
        train_paths_file = 'data/' + consts.PATH_DATA_DIR + consts.TRAIN_PATH_FILE
        valid_paths_file = 'data/' + consts.PATH_DATA_DIR + consts.VALID_PATH_FILE
        train(model, train_paths_file, valid_paths_file, args.batch_size, args.epochs, model_path,
              args.load_checkpoint, args.not_in_memory, args.lr, args.l2_reg)

    if args.eval:
        print("Evaluation Starting")

        # load paths from disk
        test_paths_file = 'data/' + consts.PATH_DATA_DIR + consts.TEST_PATH_FILE
        rmse, mae = test(model, test_paths_file, args.batch_size, model_path, args.not_in_memory)
        print("Testing_RMSE: %f, Testing_MAE: %f" % (rmse, mae))
        with open('result/main_result.txt', 'a') as fp:
            fp.write('RMSE = %s, MAE = %s\n' % (rmse, mae))


if __name__ == "__main__":
    main()
