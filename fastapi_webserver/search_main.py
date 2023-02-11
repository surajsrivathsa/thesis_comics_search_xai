from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json, os, sys, random
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import common_functions.backend_utils as utils
from search.coarse import coarse_search
import coarse_search_utils as cs_utils
import rerank_results as rrr
import interpretable_search_utils as is_utils


book_metadata_dict, comic_book_metadata_df = utils.load_book_metadata()
comic_book_metadata_df.rename(
    columns={"Book Title": "book_title", "Year": "year"}, inplace=True
)
comic_book_metadata_df.fillna("", inplace=True)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Book(BaseModel):
    id: int
    comic_no: int
    book_title: str
    genre: str
    year: str
    clicked: Optional[float] = 0.0


class BookList(BaseModel):
    clicked_book_lst: List[Book]


@app.get("/fake_book/{b_id}", status_code=200)
def get_fake_coarse_results(b_id: int):
    coarse_filtered_book_df = comic_book_metadata_df[
        ["comic_no", "book_title", "genre", "year"]
    ].sample(n=20, random_state=b_id)
    coarse_filtered_book_lst = coarse_filtered_book_df.to_dict("records")
    coarse_filtered_book_new_lst = []

    for idx, d in enumerate(coarse_filtered_book_lst):
        d["id"] = idx
        coarse_filtered_book_new_lst.append(d)

    print("Query Book : {}".format(b_id))
    return coarse_filtered_book_new_lst


# @app.get("/book/{b_id}", status_code=200)
def get_coarse_results(b_id: int):

    # query_book_comic_id = b_id # 1262 # 1647(tin-tin), 520(aquaman), 558(asterix), 587(Avengers), 650(Batman), 1270(Justice Society)
    top_n = 200
    feature_weight_dict = {
        "cld": 0.1,
        "edh": 0.1,
        "hog": 0.1,
        "text": 1.7,
    }

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
    return coarse_filtered_book_new_lst


def create_fake_clicks_for_previous_timestep_data(
    coarse_filtered_book_df: pd.DataFrame,
):

    index_lst = [i for i in range(20)]
    click_lst_size = random.randint(1, 10)
    clicked_books_idx_lst = np.random.choice(
        index_lst, size=click_lst_size, replace=False
    )
    coarse_filtered_book_df["clicked"] = 0.0
    coarse_filtered_book_df.loc[clicked_books_idx_lst, "clicked"] = 1.0

    clicked_book_lst = (
        coarse_filtered_book_df[["comic_no", "clicked"]]
        .iloc[:20, :]
        .copy()
        .fillna("")
        .to_dict("records")
    )
    return clicked_book_lst


def create_real_clicks_for_previous_timestamp_data(clicked_book_lst: List):
    clicked_book_dict_lst = [
        {"comic_no": obj.comic_no, "clicked": obj.clicked} for obj in clicked_book_lst
    ]
    return clicked_book_dict_lst


@app.post("/book_search", status_code=200)
async def search_with_real_clicks(
    cbl: BookList,
    b_id: int = Query(...),
    generate_fake_clicks: bool = Query(default=True),
):
    # print(cbl.clicked_book_lst)
    # print(b_id, generate_fake_clicks)
    # b_id: int, clicksinfo_dict: dict, generate_fake_clicks=True
    (
        coarse_filtered_book_new_lst,
        coarse_filtered_book_df,
    ) = cs_utils.perform_coarse_search(b_id=b_id)

    if generate_fake_clicks:
        clicksinfo_dict = create_fake_clicks_for_previous_timestep_data(
            coarse_filtered_book_df=coarse_filtered_book_df
        )
        # print(clicksinfo_dict)
    else:
        clicksinfo_dict = create_real_clicks_for_previous_timestamp_data(
            cbl.clicked_book_lst
        )  # [{"comic_no": "1", "clicked": 1.0}, {"comic_no": "3", "clicked": 0.0}]

    (
        feature_importance_dict,
        normalized_feature_importance_dict,
        clf_coef,
    ) = rrr.adapt_facet_weights_from_previous_timestep_click_info(
        previous_click_info_lst=clicksinfo_dict, query_book_id=b_id
    )

    (
        interpretable_filtered_book_lst,
        interpretable_filtered_book_df,
    ) = is_utils.adaptive_rerank_coarse_search_results(
        normalized_feature_importance_dict=normalized_feature_importance_dict,
        query_comic_book_id=b_id,
        coarse_search_results_lst=coarse_filtered_book_new_lst,
        top_k=20,
    )

    # add facet weights to UI
    interpretable_filtered_book_new_lst = [
        d | normalized_feature_importance_dict for d in interpretable_filtered_book_lst
    ]
    print(interpretable_filtered_book_lst[0] | normalized_feature_importance_dict)
    print(
        feature_importance_dict, clf_coef,
    )
    interpretable_filtered_book_new_lst = [
        interpretable_filtered_book_lst.copy(),
        normalized_feature_importance_dict,
    ]
    return interpretable_filtered_book_new_lst


@app.get("/book", status_code=200)
def search_all(
    b_id: int = Query(...), generate_fake_clicks: bool = Query(default=True),
):

    # b_id: int, clicksinfo_dict: dict, generate_fake_clicks=True
    (
        coarse_filtered_book_new_lst,
        coarse_filtered_book_df,
    ) = cs_utils.perform_coarse_search(b_id=b_id)

    if generate_fake_clicks:
        clicksinfo_dict = create_fake_clicks_for_previous_timestep_data(
            coarse_filtered_book_df=coarse_filtered_book_df
        )
    else:
        clicksinfo_dict = [{"0": "1"}, {"2": "0"}]

    (
        feature_importance_dict,
        normalized_feature_importance_dict,
        clf_coef,
    ) = rrr.adapt_facet_weights_from_previous_timestep_click_info(
        previous_click_info_lst=clicksinfo_dict, query_book_id=b_id
    )

    (
        interpretable_filtered_book_lst,
        interpretable_filtered_book_df,
    ) = is_utils.adaptive_rerank_coarse_search_results(
        normalized_feature_importance_dict=normalized_feature_importance_dict,
        query_comic_book_id=b_id,
        coarse_search_results_lst=coarse_filtered_book_new_lst,
        top_k=20,
    )

    # add facet weights to UI
    interpretable_filtered_book_new_lst = [
        d | normalized_feature_importance_dict for d in interpretable_filtered_book_lst
    ]
    print(interpretable_filtered_book_lst[0] | normalized_feature_importance_dict)
    print(
        feature_importance_dict, clf_coef,
    )
    interpretable_filtered_book_new_lst = [
        interpretable_filtered_book_lst.copy(),
        normalized_feature_importance_dict,
    ]
    return interpretable_filtered_book_new_lst


if __name__ == "__main__":
    # get entry page results
    coarse_filtered_book_new_lst, coarse_filtered_book_df = get_coarse_results(542)
    clicked_book_lst = create_fake_clicks_for_previous_timestep_data(
        coarse_filtered_book_df
    )
    print(clicked_book_lst[:20])
