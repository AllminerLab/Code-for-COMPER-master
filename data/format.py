import constants.consts as consts

"""
functions used for converting path data into format for the model
"""

def format_paths(paths, e_to_ix, t_to_ix, r_to_ix, sampels):
    """
    Pads paths up to max path length, converting each path into tuple
    of (padded_path, path length).
    """

    new_paths = []
    padding_path = pad_path([], e_to_ix, t_to_ix, r_to_ix, consts.MAX_PATH_LEN, consts.PAD_TOKEN)
    for path in paths:
        path_len = len(path)
        pad_path(path, e_to_ix, t_to_ix, r_to_ix, consts.MAX_PATH_LEN, consts.PAD_TOKEN)
        new_paths.append((path, path_len))
    for i in range(sampels - len(paths)):
        new_paths.append((padding_path, 1))
    return new_paths


def pad_path(seq, e_to_ix, t_to_ix, r_to_ix, max_len, padding_token):
    """
    Pads paths up to max path length
    """
    relation_padding = r_to_ix[padding_token]
    type_padding = t_to_ix[padding_token]
    entity_padding = e_to_ix[padding_token]

    while len(seq) < max_len:
        seq.append([entity_padding, type_padding, relation_padding])

    return seq