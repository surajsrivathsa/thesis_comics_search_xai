import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
    comic_book_metadata_df["our_idx"] = comic_book_metadata_df.index.copy()

    book_metadata_dict = {}
    counter = 0
    for idx, row in comic_book_metadata_df.iterrows():
        if str(row["Book Title"]) != "nan":
            book_metadata_dict[counter] = [
                row["comic_no"],
                row["Book Title"],
                row["genre"],
            ]
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

    return (
        cld_tfidf_df,
        cld_tfidf_np,
        edh_tfidf_df,
        edh_tfidf_np,
        hog_tfidf_df,
        hog_tfidf_np,
        text_tfidf_df,
        text_tfidf_np,
    )


def load_all_interpretable_features():
  
    with open(cst.STORY_PACE_FEATURE_FILEPATH, "rb") as handle:
        pace_of_story_info_feature_dict = pickle.load(handle)

    panel_ratio_np = np.array(pace_of_story_info_feature_dict['panel_ratio_per_book'][:165])
    panel_ratio_np = (panel_ratio_np - np.min(panel_ratio_np) + 1e-6)/(np.max(panel_ratio_np) + 1e-6 - np.min(panel_ratio_np))
    print(panel_ratio_np.shape)
    panel_ratio_lst = panel_ratio_np.tolist()

    interpretable_features_df = pd.read_csv(cst.INTERPRETABLE_FEATURES_FILEPATH)
    interpretable_features_df['panel_ratio'] = panel_ratio_lst
    interpretable_features_df = interpretable_features_df.iloc[:165, 2:]
    interpretable_feature_lst = list(interpretable_features_df.columns)
    interpretable_features_np = interpretable_features_df.to_numpy()

    interpretable_scaled_features_np = MinMaxScaler().fit_transform(interpretable_features_df.values)

    
    return (
        interpretable_scaled_features_np,
        interpretable_feature_lst,
    )

