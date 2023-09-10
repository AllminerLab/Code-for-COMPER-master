"""DATASET_DIR = 'Digital_Music_dataset/'
DATA_DIR = 'Digital_Music_data/'
DATA_IX_DIR = 'Digital_Music_data_ix/'
DATA_IX_MAPPING_DIR = 'Digital_Music_ix_mapping/'
PATH_DATA_DIR = 'Digital_Music_path_data/'
DATA_FILE = 'reviews_Digital_Music_5.json.gz'"""

"""DATASET_DIR = 'MovieLens_100k_dataset/'
DATA_DIR = 'MovieLens_100k_data/'
DATA_IX_DIR = 'MovieLens_100k_data_ix/'
DATA_IX_MAPPING_DIR = 'MovieLens_100k_ix_mapping/'
PATH_DATA_DIR = 'MovieLens_100k_path_data/'
DATA_FILE = 'u.data'"""

DATASET_DIR = 'Douban_dataset/'
DATA_DIR = 'Douban_data/'
DATA_IX_DIR = 'Douban_data_ix/'
DATA_IX_MAPPING_DIR = 'Douban_ix_mapping/'
PATH_DATA_DIR = 'Douban_path_data/'
DATA_FILE = 'training_test_dataset.mat'

"""DATASET_DIR = 'Jester_dataset/'
DATA_DIR = 'Jester_data/'
DATA_IX_DIR = 'Jester_data_ix/'
DATA_IX_MAPPING_DIR = 'Jester_ix_mapping/'
PATH_DATA_DIR = 'Jester_path_data/'
DATA_FILE = 'Jester.csv'"""

USER_SIM_DICT = 'user_sim.dict'
ITEM_SIM_DICT = 'item_sim.dict'
USER_ITEM_DICT = 'user_item.dict'
ITEM_USER_DICT = 'item_user.dict'
ITEM_DIRECTOR_DICT = 'item_director.dict'
DIRECTOR_ITEM_DICT = 'director_item.dict'
ITEM_ACTOR_DICT = 'item_actor.dict'
ACTOR_ITEM_DICT = 'actor_item.dict'
TRAIN_USER_ITEM_DICT = 'train_user_item.dict'
TRAIN_ITEM_USER_DICT = 'train_item_user.dict'
VALID_USER_ITEM_DICT = 'valid_user_item.dict'
VALID_ITEM_USER_DICT = 'valid_item_user.dict'
TEST_USER_ITEM_DICT = 'test_user_item.dict'
TEST_ITEM_USER_DICT = 'test_item_user.dict'
USER_ITEM_1_DICT = 'user_item_1.dict'
USER_ITEM_2_DICT = 'user_item_2.dict'
USER_ITEM_3_DICT = 'user_item_3.dict'
USER_ITEM_4_DICT = 'user_item_4.dict'
USER_ITEM_5_DICT = 'user_item_5.dict'
ITEM_USER_1_DICT = 'item_user_1.dict'
ITEM_USER_2_DICT = 'item_user_2.dict'
ITEM_USER_3_DICT = 'item_user_3.dict'
ITEM_USER_4_DICT = 'item_user_4.dict'
ITEM_USER_5_DICT = 'item_user_5.dict'
TRAIN_PATH_FILE = 'train_path_file.txt'
VALID_PATH_FILE = 'valid_path_file.txt'
TEST_PATH_FILE = 'test_path_file.txt'

TYPE_TO_IX = 'type_to_ix.dict'
RELATION_TO_IX = 'relation_to_ix.dict'
ENTITY_TO_IX = 'entity_to_ix.dict'
IX_TO_TYPE = 'ix_to_type.dict'
IX_TO_RELATION = 'ix_to_relation.dict'
IX_TO_ENTITY = 'ix_to_entity.dict'

PAD_TOKEN = '#PAD_TOKEN'
USER_TYPE = 0
ITEM_TYPE = 1
PAD_TYPE = 2
DIRECTOR_TYPE = 3
ACTOR_TYPE = 4

USER_ITEM_1_REL = 0
USER_ITEM_2_REL = 1
USER_ITEM_3_REL = 2
USER_ITEM_4_REL = 3
USER_ITEM_5_REL = 4
ITEM_USER_1_REL = 5
ITEM_USER_2_REL = 6
ITEM_USER_3_REL = 7
ITEM_USER_4_REL = 8
ITEM_USER_5_REL = 9
USER_SIM_REL = 10
ITEM_SIM_REL = 11
END_REL = 12
PAD_REL = 13
ITEM_DIRECTOR_REL = 14
DIRECTOR_ITEM_REL = 15
ITEM_ACTOR_REL = 16
ACTOR_ITEM_REL = 17

ENTITY_EMB_DIM = 128
TYPE_EMB_DIM = 32
REL_EMB_DIM = 32
HIDDEN_DIM = 256
ATTENTION_DIM = 128
MAX_PATH_LEN = 5
SAMPLES = 30
# SAMPLES = 100