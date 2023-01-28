import os, sys
import pandas as pd, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import common_functions.backend_utils as utils
import search.interpretable.interpretable_search as interpretable_search

book_metadata_dict, comic_book_metadata_df = utils.load_book_metadata()
comic_book_metadata_df.rename(
    columns={"Book Title": "book_title", "Year": "year"}, inplace=True
)
comic_book_metadata_df.fillna("", inplace=True)




def adaptive_rerank_coarse_search_results(
    normalized_feature_importance_dict: dict,
    coarse_search_results_lst: list,
    query_comic_book_id: int,
    top_k=20,
):

    interpretable_search_top_k_df = interpretable_search.adaptive_rerank_coarse_search_results(normalized_feature_importance_dict=normalized_feature_importance_dict, 
                                        coarse_search_results_lst=coarse_search_results_lst, query_comic_book_id=query_comic_book_id, top_k=top_k)


    interpretable_filtered_book_df = interpretable_search_top_k_df[
        ["comic_no", "book_title", "genre", "year"]
    ]

    interpretable_filtered_book_df.fillna("", inplace=True)

    interpretable_filtered_book_lst = interpretable_filtered_book_df.to_dict("records")
    interpretable_filtered_book_new_lst = []

    for idx, d in enumerate(interpretable_filtered_book_lst):
        d["id"] = idx
        interpretable_filtered_book_new_lst.append(d)

    return (interpretable_filtered_book_new_lst, interpretable_filtered_book_df)

