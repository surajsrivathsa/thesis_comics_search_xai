import os, sys
import pandas as pd, numpy as np
import ast

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import common_functions.backend_utils as utils


story_pace_feature_dict = utils.load_local_explanation_story_pace()
book_cover_prompt_dict = utils.load_local_explanation_book_cover()
w5_h1_facets_dict = utils.load_local_explanation_w5_h1_facets()
print("keys local explanation: {}".format(len(story_pace_feature_dict.keys())))
print("example local explanation: {}".format(story_pace_feature_dict[10]))
print()
print(" ================ ================ ================== ")
print()


def fetch_story_pace(book_id: int):
    if book_id in story_pace_feature_dict:
        # print(story_pace_feature_dict[book_id], type(story_pace_feature_dict[book_id]))
        try:
            return [int(i) for i in story_pace_feature_dict[book_id]]
        except Exception as e:
            return [int(i) for i in story_pace_feature_dict[book_id]]
    else:
        return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def fetch_5w_1h_facets(book_id: int):
    """
    {
        "Who": ["george", "all of you", "man of tomorrow"],
        "What": ["comic strip game", "romance novel", "tiny que romney prmmeneet comic"],
        "When": ["hours ice my min", "this trip", "twomin shinamaroosha saga"],
        "Why": ["why man this trip is so good", "conceive such an innate ly human pictorial document"],
        "Where": ["backcountry mountains", "high up in the mountain", "on him and see if he is making any progress"],
        "How": ["never been accomplished", "exercise his ingenuity to the utmost", "comic and glaphappy setup"],
    }
    """
    default_facets = {
        "Who": [""],
        "What": [""],
        "When": [""],
        "Why": [""],
        "Where": [""],
        "How": [""],
    }
    if book_id in w5_h1_facets_dict:
        # print(w5_h1_facets_dict[book_id], type(w5_h1_facets_dict[book_id]))
        book_facet_obj = w5_h1_facets_dict[book_id]
        if isinstance(book_facet_obj, dict):
            return book_facet_obj
        elif (
            "{" in book_facet_obj
            and "}" in book_facet_obj
            and isinstance(book_facet_obj, str)
        ):
            book_facet_obj = book_facet_obj.replace("JSON Response:", "").strip()
            brackets_start_idx = book_facet_obj.find("{")
            brackets_end_idx = book_facet_obj.find("}")
            book_facet_obj = book_facet_obj[brackets_start_idx : brackets_end_idx + 1]
            book_facet_obj_dict = ast.literal_eval(book_facet_obj)
            if isinstance(book_facet_obj_dict, dict):
                return book_facet_obj_dict
            else:
                return default_facets
        else:
            return default_facets

    else:
        return default_facets
    return ""


def fetch_book_cover_keywords(book_id: int):
    if book_id in book_cover_prompt_dict:
        # print(book_cover_prompt_dict[book_id], type(book_cover_prompt_dict[book_id]))
        return book_cover_prompt_dict[book_id]
    else:
        return ["comic book cover"]

