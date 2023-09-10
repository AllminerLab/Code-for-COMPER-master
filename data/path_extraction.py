import sys
from os import path
from collections import defaultdict
import copy
import random
sys.path.append(path.dirname(path.dirname(path.abspath('./constants'))))
import constants.consts as consts


class PathState:
    def __init__(self, path, length, entities):
        self.path = path    # array of [entity, entity type, relation to next] triplets
        self.length = length
        self.entities = entities    # set to keep track of the entities alr in the path to avoid cycles


def get_random_index(nums, max_length):
    index_list = list(range(max_length))
    random.shuffle(index_list)
    return index_list[:nums]


def find_paths_user_to_items(start_user, user_sim, item_sim, user_item_1, user_item_2, user_item_3, user_item_4,
                             user_item_5, item_user_1,  item_user_2, item_user_3, item_user_4, item_user_5, max_length,
                             sample_nums):
    '''
    Finds paths of max depth from a user to items
    '''
    item_to_paths = defaultdict(list)
    stack = []
    start = PathState([[start_user, consts.USER_TYPE, consts.END_REL]], 0, {start_user})
    stack.append(start)
    while len(stack) > 0:
        front = stack.pop()
        entity, type = front.path[-1][0], front.path[-1][1]
        # add path to item_to_paths dict, just want paths of max_length rn since length in [2,3,4,5]
        if type == consts.ITEM_TYPE and front.length == max_length:
            item_to_paths[entity].append(front.path)

        if front.length == max_length:
            continue

        if type == consts.USER_TYPE:
            if entity in user_sim:
                user_list = user_sim[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    if user not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.USER_SIM_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities | {user})
                        stack.append(new_state)

            if entity in user_item_1:
                item_list = user_item_1[entity]
                index_list = get_random_index(sample_nums, len(item_list))
                for index in index_list:
                    item = item_list[index]
                    if item not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.USER_ITEM_1_REL
                        new_path.append([item, consts.ITEM_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities | {item})
                        stack.append(new_state)

            if entity in user_item_2:
                item_list = user_item_2[entity]
                index_list = get_random_index(sample_nums, len(item_list))
                for index in index_list:
                    item = item_list[index]
                    if item not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.USER_ITEM_2_REL
                        new_path.append([item, consts.ITEM_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities | {item})
                        stack.append(new_state)

            if entity in user_item_3:
                item_list = user_item_3[entity]
                index_list = get_random_index(sample_nums, len(item_list))
                for index in index_list:
                    item = item_list[index]
                    if item not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.USER_ITEM_3_REL
                        new_path.append([item, consts.ITEM_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities | {item})
                        stack.append(new_state)

            if entity in user_item_4:
                item_list = user_item_4[entity]
                index_list = get_random_index(sample_nums, len(item_list))
                for index in index_list:
                    item = item_list[index]
                    if item not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.USER_ITEM_4_REL
                        new_path.append([item, consts.ITEM_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities | {item})
                        stack.append(new_state)

            if entity in user_item_5:
                item_list = user_item_5[entity]
                index_list = get_random_index(sample_nums, len(item_list))
                for index in index_list:
                    item = item_list[index]
                    if item not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.USER_ITEM_5_REL
                        new_path.append([item, consts.ITEM_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities | {item})
                        stack.append(new_state)

        elif type == consts.ITEM_TYPE:
            if entity in item_sim:
                item_list = item_sim[entity]
                index_list = get_random_index(sample_nums, len(item_list))
                for index in index_list:
                    item = item_list[index]
                    if item not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.ITEM_SIM_REL
                        new_path.append([item, consts.ITEM_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities | {item})
                        stack.append(new_state)

            if entity in item_user_1:
                user_list = item_user_1[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    if user not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.ITEM_USER_1_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities | {user})
                        stack.append(new_state)

            if entity in item_user_2:
                user_list = item_user_2[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    if user not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.ITEM_USER_2_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities | {user})
                        stack.append(new_state)

            if entity in item_user_3:
                user_list = item_user_3[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    if user not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.ITEM_USER_3_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities | {user})
                        stack.append(new_state)

            if entity in item_user_4:
                user_list = item_user_4[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    if user not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.ITEM_USER_4_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities | {user})
                        stack.append(new_state)

            if entity in item_user_5:
                user_list = item_user_5[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    if user not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.ITEM_USER_5_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities | {user})
                        stack.append(new_state)

    return item_to_paths