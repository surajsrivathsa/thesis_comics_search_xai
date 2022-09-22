import numpy as np
import pandas as pd

## import custom functions
import common_constants.backend_constants as cst


def cosine_similarity(u: np.array, v: np.array):
  u = np.expand_dims(u, 1)
  n = np.sum(u * v, axis=2)
  d = np.linalg.norm(u, axis=2) * np.linalg.norm(v, axis=1)
  return n / d

def l2_similarity(u: np.array, v: np.array):
  d = np.linalg.norm(u - v, axis=1)
  return d


def load_book_metadata():
    comic_book_filepath = cst.BOOK_METADATA_FILEPATH
    comic_book_metadata_df = pd.read_csv(comic_book_filepath)
    comic_book_metadata_df['our_idx'] = comic_book_metadata_df.index.copy()

    book_metadata_dict = {}
    counter = 0
    for idx, row in comic_book_metadata_df.iterrows():
        if str(row['Book Title']) != 'nan':
            book_metadata_dict[counter] = [row['comic_no'], row['Book Title'], row['genre']]
            counter += 1

    print(book_metadata_dict[0])
    return (book_metadata_dict, comic_book_metadata_df)
    

def load_all_coarse_features():
    cld_tfidf_df = pd.read_csv(cst.CLD_TF_IDF_FILEPATH)
    cld_tfidf_np = cld_tfidf_df.to_numpy()
    cld_tfidf_np = cld_tfidf_np[:165, 1:]

    edh_tfidf_df = pd.read_csv(cst.EDH_TF_IDF_FILEPATH)
    edh_tfidf_np = edh_tfidf_df.to_numpy()
    edh_tfidf_np = edh_tfidf_np[:165, 1:]

    hog_tfidf_df = pd.read_csv(cst.HOG_TF_IDF_FILEPATH)
    hog_tfidf_np = hog_tfidf_df.to_numpy()
    hog_tfidf_np = hog_tfidf_np[:165, 1:]
    
    text_tfidf_df = pd.read_csv(cst.TEXT_TF_IDF_FILEPATH)
    text_tfidf_np = text_tfidf_df.to_numpy()
    text_tfidf_np = text_tfidf_np[:165, 1:]
    
    return (cld_tfidf_df, cld_tfidf_np, edh_tfidf_df, edh_tfidf_np, hog_tfidf_df, hog_tfidf_np, text_tfidf_df, text_tfidf_np)






