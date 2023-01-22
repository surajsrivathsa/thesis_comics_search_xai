from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional
import json, os, sys
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import common_functions.backend_utils as utils

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
    id: Optional[int] = None
    comic_no: int
    book_title: str
    genre: str
    year: int


@app.get("/book/{b_id}", status_code=200)
def get_coarse_results(b_id: int):
    coarse_filtered_book_df = comic_book_metadata_df[
        ["comic_no", "book_title", "genre", "year"]
    ].sample(n=20, random_state=b_id)
    coarse_filtered_book_lst = coarse_filtered_book_df.to_dict("records")
    print("Query Book : {}".format(b_id))
    return coarse_filtered_book_lst

