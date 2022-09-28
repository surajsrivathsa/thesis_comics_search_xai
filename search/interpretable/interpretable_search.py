import os, sys
from pathlib import Path
import pandas as pd, numpy as np
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from sklearn.preprocessing import StandardScaler, MinMaxScaler

## import custom functions
import common_functions.backend_utils as utils
import common_constants.backend_constants as cst

from triplet_loss import group_triplet_loss, individual_triplet_loss
from supervised_feature_importance import (
    supervised_feature_importance_using_different_models,
)
from unsupervised_feature_importance import (
    unsupervised_feature_importance_from_laplacian,
)

## Loading features and metadata beforehand
(
    interpretable_scaled_features_np,
    interpretable_feature_lst,
) = utils.load_all_interpretable_features()
book_metadata_dict, comic_book_metadata_df = utils.load_book_metadata()
print(interpretable_scaled_features_np.shape)


def find_best_features_using_triplet_loss(
    anchors_book_idx,
    pos_constraint_book_idx,
    neg_constraint_book_idx,
    selected_idx,
    triplet_loss_type="group",
):

    anchor_feature_np = interpretable_scaled_features_np[anchors_book_idx, :]
    positive_feature_np = interpretable_scaled_features_np[pos_constraint_book_idx, :]
    negative_feature_np = interpretable_scaled_features_np[neg_constraint_book_idx, :]

    if triplet_loss_type == "group":
        feature_importance = group_triplet_loss(
            anchor_feature_np=anchor_feature_np,
            positive_feature_np=positive_feature_np,
            negative_feature_np=negative_feature_np,
            global_weights=np.array([1.0, 1.0, 1.0]),
        )
    else:
        feature_importance = individual_triplet_loss(
            anchor_feature_np=anchor_feature_np,
            positive_feature_np=positive_feature_np,
            negative_feature_np=negative_feature_np,
            individual_global_weights=np.ones(shape=positive_feature_np.shape[1]),
            cols_lst=interpretable_feature_lst,
        )

    return feature_importance


def find_best_features_using_supervised_feature_importance(selected_idx):

    features_np = interpretable_scaled_features_np[selected_idx, :]
    labels_np = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]).T

    (
        ridge_regression_feature_importance,
        logistic_regression_feature_importance,
        sgd_feature_importance,
    ) = supervised_feature_importance_using_different_models(
        features_np, labels_np, interpretable_feature_lst
    )
    return (
        ridge_regression_feature_importance,
        logistic_regression_feature_importance,
        sgd_feature_importance,
    )


def find_best_features_using_unsupervised_feature_importance(selected_idx):
    features_np = interpretable_scaled_features_np[selected_idx, :]
    labels_np = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]).T

    feature_importance = unsupervised_feature_importance_from_laplacian(
        features_np=features_np,
        labels_np=labels_np,
        col_name_lst=interpretable_feature_lst,
    )
    return feature_importance


if __name__ == "__main__":
    anchors_book_idx = [34]
    pos_constraint_book_idx = [14, 40, 99]
    neg_constraint_book_idx = [0, 20, 56]
    users_idx = anchors_book_idx + pos_constraint_book_idx + neg_constraint_book_idx
    selected_idx = users_idx  # np.random.choice(idx_arr, 7, replace=False)
    print(
        "Anchor: {}  | Positive: {} | Negative: {}".format(
            selected_idx[0], selected_idx[1:4], selected_idx[4:]
        )
    )

    feature_importance = find_best_features_using_triplet_loss(
        anchors_book_idx=anchors_book_idx,
        pos_constraint_book_idx=pos_constraint_book_idx,
        neg_constraint_book_idx=neg_constraint_book_idx,
        selected_idx=selected_idx,
    )
    print()
    print("Group Based triplet loss: {}".format(feature_importance))
    print()
    print(" === +++++++++++++++++++++++++++ ==")
    print()

    feature_importance = find_best_features_using_triplet_loss(
        anchors_book_idx=anchors_book_idx,
        pos_constraint_book_idx=pos_constraint_book_idx,
        neg_constraint_book_idx=neg_constraint_book_idx,
        selected_idx=selected_idx,
        triplet_loss_type="individual",
    )

    print()
    print("Individual Based triplet loss: {}".format(feature_importance))
    print()
    print(" === +++++++++++++++++++++++++++ ==")
    print()

    (
        ridge_regression_feature_importance,
        logistic_regression_feature_importance,
        sgd_feature_importance,
    ) = find_best_features_using_supervised_feature_importance(
        selected_idx=selected_idx
    )
    print()
    print(" === +++++++++++++++++++++++++++ === ")
    print()
    print(
        "ridge regression feature importance: {}".format(
            ridge_regression_feature_importance
        )
    )
    print()
    print(
        "logistic regression feature importance: {}".format(
            logistic_regression_feature_importance
        )
    )
    print()
    print("sgd feature importance: {}".format(sgd_feature_importance))
    print()
    print(" === +++++++++++++++++++++++++++ === ")
    print()

    laplacian_feature_importance = find_best_features_using_unsupervised_feature_importance(
        selected_idx=selected_idx
    )
    print("laplacian feature importance: {}".format(laplacian_feature_importance))
    print()
    print(" === +++++++++++++++++++++++++++ === ")
    print()

