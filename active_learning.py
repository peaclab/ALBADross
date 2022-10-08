#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.insert(0, "../../../src")

import argparse
import pandas as pd
import os

from pathlib import Path
import json
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stderr, level=logging.INFO
)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.INFO)

from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# General ML
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    silhouette_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from modules.clustering_helpers import select_labeled_samples
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Active Learning
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling

# In-house Module Imports
from config import Configuration
from datasets import EclipseSampledDataset, VoltaSampledDataset
from utils import *


def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]


def equal_app_sampling(classifier, X_pool, app_labels):
    """
    Selects one random sample from each application exist in the app_labels.Assumes the given
    app_labels has all the applications

    """

    query_idx = []
    for app in np.unique(app_labels):
        query_idx.append(np.random.choice(np.where(app_labels == app)[0], 1)[0])

    query_idx = np.array(query_idx)
    return query_idx, X_pool[query_idx]


def call_FAR_function(false_alarm_rates, anomaly_miss_rates, test_label, y_pred, conf):
    false_alarm_rate, anom_miss_rate = FAR_AMR_Calculate(
        true_label=test_label["anom"].to_numpy(),
        pred_label=y_pred,
        result_dir=str(conf["results_dir"]),
        save_name="",
        save=False,
        verbose=False,
    )
    false_alarm_rates.append(false_alarm_rate)
    anomaly_miss_rates.append(anom_miss_rate)


query_strategy_dict = {
    "uncertainty": uncertainty_sampling,
    "margin": margin_sampling,
    "entropy": entropy_sampling,
    "random": random_sampling,
    "equal_app": equal_app_sampling,
}


def main():

    logging.info("Argument List: %s", str(sys.argv))
    OUTER_DIR = sys.argv[1]  # active_learning_experiments_final_hdfs or active_learning_experiments
    SYSTEM = sys.argv[2]  # volta or eclipse
    FE_NAME = sys.argv[3]  # it can be either mvts of tsfresh
    MODEL_CONFIG = sys.argv[4]  # "exp_1_active_learning_random"
    NUM_FEATURE = int(sys.argv[5])  # example: 250 ,2000, 4000
    query_strategy = str(sys.argv[6])  # "uncertainty", "margin", "entropy", "random", "equal_app"
    CV_INDEX = int(sys.argv[7])  # it can be integer value within the range 0 1 2 3 4
    repeat_num = int(sys.argv[8])
    query_size = int(sys.argv[9])
    classifier_name = sys.argv[10]

    # Constants
    FS_NAME = "CHI"
    user = "aksar"
    logging.warning(f"Are you sure that you are: {user}?")

    method = (
        "baseline"
        if (query_strategy == "random" or query_strategy == "equal_app")
        else "active_learning"
    )
    num_samples_per_pair = 1
    OUTPUT_DIR = f"/projectnb/peaclab-mon/{user}/{OUTER_DIR}"
    EXP_NAME = f"{FE_NAME}_experiments"
    FEATURE_SELECTION = False
    SCALER = (  # For now, do the scaling inside the notebook, then you can move that to the class function
        "None"
    )
    logging.warning("Results will be generated in {}, double check please!".format(MODEL_CONFIG))

    conf = Configuration(
        ipython=True,
        overrides={
            "output_dir": Path(OUTPUT_DIR),  # change
            "system": SYSTEM,
            "exp_name": EXP_NAME,
            "cv_fold": CV_INDEX,
            "model_config": MODEL_CONFIG,
        },
    )

    with open(str(conf["experiment_dir"]) + "/anom_dict.json") as f:
        ANOM_DICT = json.load(f)
    with open(str(conf["experiment_dir"]) + "/app_dict.json") as f:
        APP_DICT = json.load(f)

    APP_REVERSE_DICT = {}
    for app_name, app_encoding in APP_DICT.items():
        APP_REVERSE_DICT[app_encoding] = app_name

    ANOM_REVERSE_DICT = {}
    for anom_name, anom_encoding in ANOM_DICT.items():
        ANOM_REVERSE_DICT[anom_encoding] = anom_name

    if SYSTEM == "eclipse":
        eclipseDataset = EclipseSampledDataset(conf)
        train_data, train_label, test_data, test_label = eclipseDataset.load_dataset(
            cv_fold=CV_INDEX,
            scaler=SCALER,
            borghesi=False,
            mvts=True if FE_NAME == "mvts" else False,
            tsfresh=True if FE_NAME == "tsfresh" else False,
        )

    elif SYSTEM == "volta":
        voltaDataset = VoltaSampledDataset(conf)
        train_data, train_label, test_data, test_label = voltaDataset.load_dataset(
            cv_fold=CV_INDEX,
            scaler=SCALER,
            borghesi=False,
            mvts=True if FE_NAME == "mvts" else False,
            tsfresh=True if FE_NAME == "tsfresh" else False,
        )

    assert list(train_data.index) == list(train_label.index)  # check the order of the labels
    assert list(test_data.index) == list(test_label.index)  # check the order of the labels

    if FEATURE_SELECTION:
        selected_features = pd.read_csv(conf["experiment_dir"] / "selected_features.csv")
        train_data = train_data[list(selected_features["0"].values)]
        test_data = test_data[list(selected_features["0"].values)]

    train_label["anom_names"] = train_label.apply(lambda x: ANOM_REVERSE_DICT[x["anom"]], axis=1)
    train_label["app_names"] = train_label["app"].apply(lambda x: APP_REVERSE_DICT[x])
    test_label["anom_names"] = test_label.apply(lambda x: ANOM_REVERSE_DICT[x["anom"]], axis=1)
    test_label["app_names"] = test_label["app"].apply(lambda x: APP_REVERSE_DICT[x])

    all_data = pd.concat([train_data, test_data])
    all_data = all_data.dropna(axis=1, how="any")
    all_label = pd.concat([train_label, test_label])

    train_data = all_data.loc[train_label.index]
    test_data = all_data.loc[test_label.index]

    logging.info("Train data shape %s", train_data.shape)
    logging.info("Train label shape %s", train_label.shape)
    logging.info("Test data shape %s", test_data.shape)
    logging.info("Test label shape %s", test_label.shape)

    logging.info("Train data label dist: \n%s", train_label["anom"].value_counts())
    logging.info("Test data label dist: \n%s", test_label["anom"].value_counts())

    SCALER = "MinMax"

    if SCALER == "MinMax":

        minmax_scaler = MinMaxScaler().fit(train_data)
        train_data = pd.DataFrame(
            minmax_scaler.transform(train_data), columns=train_data.columns, index=train_data.index
        )
        test_data = pd.DataFrame(
            minmax_scaler.transform(test_data), columns=test_data.columns, index=test_data.index
        )

    elif SCALER == "Standard":

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(train_data)
        train_data = pd.DataFrame(
            scaler.transform(train_data), columns=train_data.columns, index=train_data.index
        )
        test_data = pd.DataFrame(
            scaler.transform(test_data), columns=test_data.columns, index=test_data.index
        )

    # Implement new feature selection strategies below
    if FS_NAME == "CHI":

        selector = SelectKBest(chi2, k=NUM_FEATURE)
        selector.fit(train_data, train_label["anom"])
        train_data = train_data[train_data.columns[selector.get_support(indices=True)]]
        selected_columns = train_data.columns
        test_data = test_data[test_data.columns & selected_columns]

    elif FS_NAME == "TSFRESH":
        logging.warning(
            "NUM_FEATURE parameter will be overwritten by the automatic selection process"
        )

        y_train = train_label["anom"]
        X_train = train_data

        relevant_features = set()

        for label in y_train.unique():
            y_train_binary = y_train == label
            X_train_filtered = tsfresh.select_features(X_train, y_train_binary)
            print(
                "Number of relevant features for class {}: {}/{}".format(
                    label, X_train_filtered.shape[1], X_train.shape[1]
                )
            )
            relevant_features = relevant_features.union(set(X_train_filtered.columns))
        train_data = train_data[relevant_features]
        test_data = test_data[relevant_features]
        NUM_FEATURE = len(relevant_features)

    elif FS_NAME == "NONE":
        logging.info("No feature selection strategy is specified, will be using all features")
        NUM_FEATURE = len(train_data.columns)

    logging.info(train_data.shape)
    logging.info(test_data.shape)

    # Read the node_ids considered labeled
    labeled_train_label = pd.read_csv(
        conf["experiment_dir"]
        / f"CV_{CV_INDEX}"
        / f"labeled_train_label_{num_samples_per_pair}.csv",
        index_col=["node_id"],
    )
    labeled_test_label = pd.read_csv(
        conf["experiment_dir"]
        / f"CV_{CV_INDEX}"
        / f"labeled_test_label_{num_samples_per_pair}.csv",
        index_col=["node_id"],
    )
    node_indices_labeled = list(labeled_train_label["anom"].index.values)

    logging.info("Labeled data label dist: \n%s", labeled_train_label["anom"].value_counts())
    logging.info("Unlabeled data label dist: \n%s", labeled_test_label["anom"].value_counts())

    # Set a new column for label status
    node_indices_unlabeled = []
    for node in train_label.index:
        if node not in node_indices_labeled:
            node_indices_unlabeled.append(node)
    train_label["label_status"] = train_label["anom"]  # for the full data case
    train_label["label_status"] = np.where(
        train_label.index.get_level_values("node_id").isin(node_indices_unlabeled),
        -1,
        train_label["label_status"],
    )

    # initial_labeled_pool contains one sample from each application anomaly pair
    initial_labeled_pool = train_label[(train_label["label_status"] != -1)]
    # Active learning or random sampling will be querying from the same pool
    initial_unlabeled_pool = train_label[(train_label["label_status"] == -1)]

    if classifier_name == "rf":
        best_params = {
            "criterion": "entropy" if SYSTEM == "eclipse" else "entropy",
            "max_depth": 8 if SYSTEM == "eclipse" else 8,
            "n_estimators": 200 if SYSTEM == "eclipse" else 20,
        }

        selected_classifier = RandomForestClassifier(
            criterion=best_params["criterion"],
            max_depth=best_params["max_depth"],
            n_estimators=best_params["n_estimators"],
        )

    elif classifier_name == "lr":
        best_params = {
            "C": 1.0 if SYSTEM == "eclipse" else 10,
            "penalty": "l1" if SYSTEM == "eclipse" else "l1",
            "solver": "liblinear" if SYSTEM == "eclipse" else "liblinear",
        }

        selected_classifier = LogisticRegression(
            C=best_params["C"],
            penalty=best_params["penalty"],
            solver=best_params["solver"],
        )

    elif classifier_name == "mlp":

        best_params = {
            "alpha": 0.0001 if SYSTEM == "eclipse" else 0.01,
            "hidden_layer_sizes": [50, 100, 50] if SYSTEM == "eclipse" else [100],
            "max_iter": 100 if SYSTEM == "eclipse" else 100,
        }

        selected_classifier = MLPClassifier(
            random_state=1,
            alpha=best_params["alpha"],
            hidden_layer_sizes=best_params["hidden_layer_sizes"],
            max_iter=best_params["max_iter"],
        )
    elif classifier_name == "lgbm":

        best_params = {
            "num_leaves": 31 if SYSTEM == "eclipse" else 128,
            "learning_rate": 0.1 if SYSTEM == "eclipse" else 0.1,
            "max_depth": -1 if SYSTEM == "eclipse" else 8,
            "colsample_bytree": 1 if SYSTEM == "eclipse" else 1,
        }
        selected_classifier = LGBMClassifier(
            objective="multiclass",
            num_leaves=best_params["num_leaves"],
            learning_rate=best_params["learning_rate"],
            max_depth=best_params["max_depth"],
            colsample_bytree=best_params["colsample_bytree"],
            random_state=5,
        )

    else:
        selected_classifier = RandomForestClassifier()

    logging.info(selected_classifier)

    scores = pd.DataFrame()
    all_app_names = list(APP_DICT.keys())
    selected_apps = dict.fromkeys(all_app_names, 0)
    selected_anoms = dict.fromkeys(list(ANOM_REVERSE_DICT.keys()), 0)

    # Create the label and data for the starting condition composed of selected apps
    y_initial = initial_labeled_pool
    x_initial = train_data[train_data.index.get_level_values("node_id").isin(y_initial.index)]
    y_initial = y_initial["anom"].to_numpy()
    x_initial = x_initial.to_numpy()

    x_unlabeled = train_data[
        train_data.index.get_level_values("node_id").isin(initial_unlabeled_pool.index)
    ]
    y_unlabeled = initial_unlabeled_pool
    x_unlabeled = x_unlabeled.to_numpy()

    # Initializations
    macro_f1_scores = []
    anomaly_miss_rates = []
    false_alarm_rates = []

    if query_strategy != "random":
        selected_indices_apps = []
        selected_indices_anoms = []

    if query_strategy == "equal_app":
        step = len(all_app_names)
    else:
        step = 1

    x_pool = x_unlabeled.copy()
    y_pool = y_unlabeled.copy()
    y_pool_anom = y_pool["anom"].to_numpy()
    y_pool_app = y_pool["app_names"].to_numpy()

    learner = ActiveLearner(
        estimator=selected_classifier,
        query_strategy=query_strategy_dict[query_strategy],
        X_training=x_initial,
        y_training=y_initial,
    )

    y_pred = learner.predict(test_data.to_numpy())
    report_dict = classification_report(test_label["anom"].to_numpy(), y_pred, output_dict=True)
    macro_f1_scores.append(report_dict["macro avg"]["f1-score"])
    call_FAR_function(false_alarm_rates, anomaly_miss_rates, test_label, y_pred, conf)

    for i in range(0, query_size, step):
        if not i % 50:
            logging.info(f"Current query index: {i}")

        if query_strategy == "equal_app":
            query_idx, query_sample = learner.query(x_pool, app_labels=y_pool_app)
            learner.teach(
                X=x_pool[query_idx],
                y=y_pool_anom[query_idx],
            )
        else:
            query_idx, query_sample = learner.query(x_pool)
            learner.teach(
                X=x_pool[query_idx].reshape(1, -1),
                y=y_pool_anom[query_idx].reshape(
                    1,
                ),
            )

        if query_strategy != "random":
            selected_indices_apps.append(y_pool_app[query_idx][0])
            selected_indices_anoms.append(y_pool_anom[query_idx][0])

        x_pool, y_pool_anom, y_pool_app = (
            np.delete(x_pool, query_idx, axis=0),
            np.delete(y_pool_anom, query_idx, axis=0),
            np.delete(y_pool_app, query_idx, axis=0),
        )
        y_pred = learner.predict(test_data.to_numpy())

        report_dict = classification_report(test_label["anom"].to_numpy(), y_pred, output_dict=True)
        macro_f1_scores.append(report_dict["macro avg"]["f1-score"])
        call_FAR_function(false_alarm_rates, anomaly_miss_rates, test_label, y_pred, conf)

    for j in range(0, len(macro_f1_scores)):
        scores = scores.append(
            {
                "query_iter": j * step,
                "macro_avg_f1_score": macro_f1_scores[j],
                "false_alarm_rate": false_alarm_rates[j],
                "anomaly_miss_rate": anomaly_miss_rates[j],
                "repeat_num": repeat_num,
            },
            ignore_index=True,
        )

    scores["fold"] = CV_INDEX
    scores["method"] = method
    scores["query_strategy"] = query_strategy
    scores["model"] = selected_classifier.__class__.__name__
    scores["dataset"] = SYSTEM
    scores["fe"] = FE_NAME
    scores["feature_count"] = NUM_FEATURE
    scores["query_size"] = query_size

    scores = scores.sort_values(by=["query_iter"]).reset_index(drop=True)

    train_app_names = "all"
    test_app_names = "all"

    filename = f"train:{train_app_names}#test:{test_app_names}#{FE_NAME}#{NUM_FEATURE}#{method}#{query_strategy}#{query_size}#{selected_classifier.__class__.__name__}#{repeat_num}.csv"
    scores.to_csv(Path(conf["results_dir"]) / filename)

    logging.info("Saving scores: %s", filename)

    if query_strategy != "random":
        selected_app_anom_df = pd.DataFrame()
        selected_app_anom_df["apps"] = selected_indices_apps
        selected_app_anom_df["anoms"] = selected_indices_anoms
        selected_app_anom_df.to_csv(
            Path(conf["results_dir"])
            / f"train:{train_app_names}#test:{test_app_names}#{FE_NAME}#{NUM_FEATURE}#{method}#{query_strategy}#{query_size}#{selected_classifier.__class__.__name__}#{repeat_num}#app-anom-selection.csv",
            index=False,
        )

        logging.info("Saved selected apps and anoms")


if __name__ == "__main__":
    main()
