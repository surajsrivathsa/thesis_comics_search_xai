from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
import pandas as pd, numpy as np, os, sys
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


## import custom functions
import common_functions.backend_utils as utils
import common_constants.backend_constants as cst

# clf = PassiveAggressiveClassifier(max_iter=100, random_state=7, tol=1e-3)
clf_pipe = Pipeline(
    [("scl", StandardScaler()), ("clf", SGDClassifier(max_iter=1000, tol=1e-3))]
)

## Loading features and metadata beforehand
(
    interpretable_scaled_features_np,
    interpretable_feature_lst,
    gender_feat_np,
    supersense_feat_np,
    genre_feat_np,
    panel_ratio_feat_np,
    comic_cover_img_np,
    comic_cover_txt_np
) = utils.load_all_interpretable_features()
book_metadata_dict, comic_book_metadata_df = utils.load_book_metadata()
print(interpretable_scaled_features_np.shape)


def get_top_n_matching_book_info(
    idx_top_n_np,
    sim_score_top_n_np,
    comic_info_dict=book_metadata_dict,
    print_n=20,
    query_book_id=1,
    feature_similarity_type="cld",
):
    sim_score_top_n_squeezed_np = np.squeeze(sim_score_top_n_np)
    list_of_records = []
    # print(comic_info_dict[query_book_id])
    query_comic_no, query_book_title, query_genre, year = comic_info_dict[query_book_id]
    # print(idx_top_n_np, sim_score_top_n_squeezed_np.shape)
    for i in range(1, print_n):

        book_idx = idx_top_n_np[i]
        sim_score_book = sim_score_top_n_squeezed_np[i]

        try:
            comic_no, book_title, genre, year = comic_info_dict[book_idx]
        except Exception as e:
            comic_no, book_title, genre, year = (-1, "not exist", "not exist", "n.a")

        list_of_records.append(
            {
                "rank": i,
                "sim_score": sim_score_book,
                "comic_no": comic_no,
                "book_title": book_title,
                "genre": genre,
                "year": year,
                "query_comic_no": query_comic_no,
                "query_book_title": query_book_title,
                "query_genre": query_genre,
                "feature_similarity_type": feature_similarity_type,
            }
        )

    df = pd.DataFrame.from_dict(list_of_records)
    return df


def permutation_based_feature_importance(
    model, X, y, stddev_weight=2, feature_col_labels_lst=[]
):
    r = permutation_importance(model, X, y, n_repeats=30, random_state=5)
    feature_importance_dict = {}
    mean_lst = []
    feature_nm_lst = []

    # for rank, i in enumerate(r.importances_mean.argsort()[::-1]):
    # print(r.importances_mean)
    for rank, i in enumerate(r.importances_mean.argsort()[::-1]):
        # if r.importances_mean[i] - stddev_weight * r.importances_std[i] > 0:
        # print("Feat Name: {}  : {} +/- {}".format(feature_col_labels_lst[i], r.importances_mean[i], r.importances_std[i],))
        feature_importance_dict[feature_col_labels_lst[i]] = [
            r.importances_mean[i],
            r.importances_std[i],
            rank,
        ]
        mean_lst.append(r.importances_mean[i])
        feature_nm_lst.append(feature_col_labels_lst[i])

    normalized_feature_importance_lst = [
        (float(mn) - min(mean_lst) + 1e-3) / (max(mean_lst) - min(mean_lst) + 1e-2)
        for mn in mean_lst
    ]

    normalized_feature_importance_dict = {
        feat_name: norm_mn
        for feat_name, norm_mn in zip(feature_nm_lst, normalized_feature_importance_lst)
    }

    print(feature_importance_dict)

    return (feature_importance_dict, normalized_feature_importance_dict)


def adapt_facet_weights_from_previous_timestep_click_info(
    previous_click_info_lst: list, query_book_id: int
):

    selected_idx = [d["comic_no"] for d in previous_click_info_lst]
    # print(selected_idx)
    previous_labels_lst = [
        d["clicked"] for d in previous_click_info_lst
    ]  # [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    query_book_id = query_book_id  # [d["comic_no"] for d in previous_click_info_lst if d["is_query"] == 1][0]
    # print(previous_labels_lst)
    features_np = interpretable_scaled_features_np[selected_idx, :]
    labels_np = np.array([previous_labels_lst]).T

    # featurize the facets with query combo
    gender_l1_feat_np = utils.l1_similarity(
        gender_feat_np[selected_idx, :],
        gender_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    supersense_l1_feat_np = utils.l1_similarity(
        supersense_feat_np[selected_idx, :],
        supersense_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    genre_l1_feat_np = utils.l1_similarity(
        genre_feat_np[selected_idx, :],
        genre_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )
    panel_l1_feat_np = utils.l1_similarity(
        panel_ratio_feat_np[selected_idx, :],
        panel_ratio_feat_np[max(query_book_id, 0) : query_book_id + 1, :],
    )

    comic_cover_img_l1_feat_np = utils.l1_similarity(
        comic_cover_img_np[selected_idx, :],
        comic_cover_img_np[max(query_book_id, 0) : query_book_id + 1, :],
    )

    comic_cover_txt_l1_feat_np = utils.l1_similarity(
        comic_cover_txt_np[selected_idx, :],
        comic_cover_txt_np[max(query_book_id, 0) : query_book_id + 1, :],
    )

    features_np = np.zeros((len(selected_idx), 6))
    features_np[:, 0] = gender_l1_feat_np
    features_np[:, 1] = supersense_l1_feat_np
    features_np[:, 2] = genre_l1_feat_np
    features_np[:, 3] = panel_l1_feat_np
    features_np[:, 4] = comic_cover_img_l1_feat_np
    features_np[:, 5] = comic_cover_txt_l1_feat_np

    clf_pipe.fit(features_np, labels_np.ravel())
    # print(clf_pipe['clf'].get_params())
    # clf = clf_pipe.best_estimator_.named_steps["clf"]
    (
        feature_importance_dict,
        normalized_feature_importance_dict,
    ) = permutation_based_feature_importance(
        model=clf_pipe,
        X=features_np,
        y=labels_np,
        stddev_weight=0.5,
        feature_col_labels_lst=["gender", "supersense", "genre_comb", "panel_ratio", "comic_cover_img", "comic_cover_txt"],
    )

    return (
        feature_importance_dict,
        normalized_feature_importance_dict,
        clf_pipe["clf"].coef_,
    )


if __name__ == "__main__":

    comic_no_lst = [
        336,
        149,
        41,
        79,
        21,
        385,
        111,
        305,
        265,
        78,
        387,
        415,
        154,
        195,
        320,
        318,
        251,
        447,
        462,
        46,
        179,
        450,
        82,
        774,
        4,
        289,
        1518,
        463,
        142,
        1493,
        161,
        418,
        80,
        457,
        466,
        250,
        1452,
        1233,
        54,
        426,
        356,
        301,
        378,
        37,
        634,
        134,
        1341,
        471,
        467,
        75,
        1504,
        390,
        465,
        256,
        20,
        1508,
        1562,
        504,
        641,
        330,
        177,
        123,
        1117,
        636,
        182,
        339,
        225,
        113,
        279,
        643,
        249,
        340,
        1132,
        83,
        496,
        136,
        324,
        1519,
        1065,
        487,
        129,
        335,
        449,
        1489,
        288,
        285,
        56,
        1503,
        515,
        446,
        421,
        282,
        581,
        261,
        1337,
        1066,
        211,
        456,
        802,
        688,
        263,
        72,
        323,
        1555,
        185,
        1116,
        226,
        1560,
        24,
        291,
        483,
        281,
        10,
        939,
        247,
        485,
        1514,
        139,
        223,
        1063,
        412,
        362,
        183,
        303,
        1565,
        513,
        30,
        1535,
        1137,
        431,
        255,
        153,
        484,
        1018,
        68,
        42,
        518,
        36,
        205,
        1386,
        198,
        906,
        1463,
        904,
        232,
        107,
        203,
        14,
        1506,
        1062,
        1045,
        23,
        128,
        81,
        109,
        1494,
        187,
        687,
        384,
        377,
        99,
        901,
        1520,
        480,
        1134,
        1136,
        114,
        224,
        199,
        193,
        16,
        521,
        357,
        1135,
        1650,
        1026,
        1479,
        420,
        1509,
        1563,
        380,
        284,
        1390,
        619,
        509,
        150,
        647,
        287,
        936,
        220,
        63,
        244,
        981,
        455,
        1068,
        652,
        334,
        1564,
        974,
    ]
    clicked_lst = [
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    previous_click_info_lst = [
        {"comic_no": comic_no, "clicked": clicked}
        for comic_no, clicked in zip(comic_no_lst, clicked_lst)
    ]

    """
    previous_click_info_lst = [
        {"comic_no": 537, "clicked": 1.0},
        {"comic_no": 539, "clicked": 1.0},
        {"comic_no": 553, "clicked": 0.0},
        {"comic_no": 770, "clicked": 0.0},
        {"comic_no": 1144, "clicked": 0.0},
        {"comic_no": 1026, "clicked": 0.0},
        {"comic_no": 1, "clicked": 0.0},
        {"comic_no": 128, "clicked": 0.0},
        {"comic_no": 148, "clicked": 0.0},
        {"comic_no": 533, "clicked": 1.0},
        {"comic_no": 334, "clicked": 1.0},
        {"comic_no": 150, "clicked": 0.0},
        {"comic_no": 700, "clicked": 0.0},
        {"comic_no": 750, "clicked": 0.0},
        {"comic_no": 800, "clicked": 0.0},
        {"comic_no": 1200, "clicked": 0.0},
        {"comic_no": 652, "clicked": 0.0},
        {"comic_no": 1558, "clicked": 0.0},
        {"comic_no": 1500, "clicked": 0.0},
        {"comic_no": 1005, "clicked": 0.0},
        {"comic_no": 1234, "clicked": 0.0},
    ]
    """
    (
        feature_importance_dict,
        normalized_feature_importance_dict,
        clf_coeff,
    ) = adapt_facet_weights_from_previous_timestep_click_info(
        previous_click_info_lst, query_book_id=5
    )

    print()
    print(" =================== =================== =================== ")
    print()
    print(normalized_feature_importance_dict)
    print()
    print(" =================== =================== =================== ")
    print()
    print(feature_importance_dict)
    print()
    print(" =================== =================== =================== ")
    print()
    print(clf_coeff)

