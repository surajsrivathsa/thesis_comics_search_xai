import os, sys
import pandas as pd, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import common_functions.backend_utils as utils


story_pace_feature_dict = utils.load_local_explanation_story_pace()
print("keys local explanation: {}".format(len(story_pace_feature_dict.keys())))
print("example local explanation: {}".format(story_pace_feature_dict[10]))
print()
print(" ================ ================ ================== ")
print()


def fetch_story_pace(book_id: int):
    if book_id in story_pace_feature_dict:
        print(story_pace_feature_dict[book_id], type(story_pace_feature_dict[book_id]))
        return [int(i) for i in story_pace_feature_dict[book_id]]
    else:
        return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def fetch_5w_1h_facets(book_id: int):

    return ""


def fetch_lrp_keywords(book_id: int):

    return ""

