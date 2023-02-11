import os, sys
import pandas as pd, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import common_functions.backend_utils as utils
from search.coarse import coarse_search

book_metadata_dict, comic_book_metadata_df = utils.load_book_metadata()
comic_book_metadata_df.rename(
    columns={"Book Title": "book_title", "Year": "year"}, inplace=True
)
comic_book_metadata_df.fillna("", inplace=True)


def perform_coarse_search(
    b_id: int,
    feature_weight_dict={
        "cld": 0.1,
        "edh": 0.1,
        "hog": 0.1,
        "text": 1.0,
        "comic_img": 1.0,
        "comic_txt": 1.0,
    },
    top_n=200,
):

    # query_book_comic_id = b_id # 1262 # 1647(tin-tin), 520(aquaman), 558(asterix), 587(Avengers), 650(Batman), 1270(Justice Society)

    top_n_results_df = coarse_search.comics_coarse_search(
        query_comic_book_id=b_id, feature_weight_dict=feature_weight_dict, top_n=top_n
    )
    coarse_filtered_book_df = top_n_results_df[
        ["comic_no", "book_title", "genre", "year"]
    ]

    coarse_filtered_book_df.fillna("", inplace=True)

    coarse_filtered_book_lst = coarse_filtered_book_df.to_dict("records")
    coarse_filtered_book_new_lst = []

    for idx, d in enumerate(coarse_filtered_book_lst):
        d["id"] = idx
        coarse_filtered_book_new_lst.append(d)

    print("Query Book : {} ".format(b_id))
    # return (coarse_filtered_book_new_lst, coarse_filtered_book_df)
    return (coarse_filtered_book_new_lst, coarse_filtered_book_df)

