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
from modules.clustering_helpers import select_labeled_samples_unseen_inputs


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
}


def main():

    user = "aksar"
    logging.warning(f"Are you sure that you are: {user}?")

    ### Update manually ###
    MODEL_CONFIG = "exp_3_active_learning"
    logging.info(f"Model config name: {MODEL_CONFIG}?")

    logging.info("Argument List: %s", str(sys.argv))
    SYSTEM = sys.argv[1]  # volta or eclipse
    FE_NAME = sys.argv[2]  # it can be either mvts of tsfresh
    NUM_FEATURE = int(sys.argv[3])  # example: 250 ,2000, 4000
    query_strategy = str(sys.argv[4])  # "uncertainty", "margin", "entropy", "random"
    CV_INDEX = int(sys.argv[5])  # it can be integer value within the range 0 1 2 3 4
    repeat_num = int(sys.argv[6])
    query_size = int(sys.argv[7])
    classifier_name = sys.argv[8]

    # Constants
    FS_NAME = "CHI"
    method = "random" if query_strategy == "random" else "active_learning"
    num_samples_per_pair = 1
    OUTPUT_DIR = f"/projectnb/peaclab-mon/{user}/active_learning_experiments"
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
            scaler=SCALER,
            cv_fold=CV_INDEX,
            borghesi=False,
            mvts=True if FE_NAME == "mvts" else False,
            tsfresh=True if FE_NAME == "tsfresh" else False,
        )

    elif SYSTEM == "volta":
        voltaDataset = VoltaSampledDataset(conf)
        train_data, train_label, test_data, test_label = voltaDataset.load_dataset(
            scaler=SCALER,
            cv_fold=CV_INDEX,
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

    metadata = pd.read_csv(conf["hdf_data_path"] / f"{SYSTEM}_metadata.csv", index_col=["node_id"])
    metadata = metadata.loc[all_label.index]

    train_label["input"] = metadata.loc[train_label.index]["input"].values
    test_label["input"] = metadata.loc[test_label.index]["input"].values
    labeled_train_label["input"] = metadata.loc[labeled_train_label.index]["input"].values
    labeled_test_label["input"] = metadata.loc[labeled_test_label.index]["input"].values

    if classifier_name == "rf":
        selected_classifier = RandomForestClassifier()
    elif classifier_name == "lr":
        selected_classifier = LogisticRegression()
    else:
        selected_classifier = RandomForestClassifier()

    if SYSTEM == "volta":
        unique_app_inputs = ["X", "Y", "Z"]
    elif SYSTEM == "eclipse":
        unique_app_inputs = None

    all_test_input_groups = []
    all_train_input_groups = []

    for temp_input in list(combinations(unique_app_inputs, 1)):

        all_train_input_groups.append([temp_input[0]])
        all_test_input_groups.append(list(set(unique_app_inputs) - set(temp_input)))

    for temp_input in list(combinations(unique_app_inputs, 2)):

        all_train_input_groups.append([temp_input[0], temp_input[1]])
        all_test_input_groups.append(list(set(unique_app_inputs) - set(temp_input)))

    scores = pd.DataFrame()

    for test_inputs, train_inputs in zip(all_test_input_groups, all_train_input_groups):

        logging.info("Test inputs: %s", test_inputs)
        logging.info("Train inputs: %s", train_inputs)

        test_inputs_label = test_label[test_label["input"].isin(test_inputs)]
        assert set(test_inputs_label["input"].unique()) == set(test_inputs)

        test_inputs_data = test_data.loc[test_inputs_label.index]
        assert list(test_inputs_data.index) == list(test_inputs_label.index)

        (
            initial_labeled_pool_labels,
            unlabeled_pool_labels,
            node_indices_labeled,
        ) = select_labeled_samples_unseen_inputs(train_label, train_inputs)
        logging.info("Labeled data label dist: \n%s", labeled_train_label["anom"].value_counts())
        logging.info("Labeled data label dist: \n%s", labeled_train_label["app"].value_counts())
        logging.info("Labeled data label dist: \n%s", labeled_train_label["input"].value_counts())
        logging.info("########################")
        logging.info(
            "Unlabeled data label dist: \n%s", unlabeled_pool_labels["anom"].value_counts()
        )
        logging.info("Unlabeled data label dist: \n%s", unlabeled_pool_labels["app"].value_counts())
        logging.info(
            "Unlabeled data label dist: \n%s", unlabeled_pool_labels["input"].value_counts()
        )
        logging.info("########################")

        # Create the label and data for the starting condition composed of selected apps
        y_initial = initial_labeled_pool_labels
        x_initial = train_data[train_data.index.get_level_values("node_id").isin(y_initial.index)]

        y_initial = initial_labeled_pool_labels["anom"].to_numpy()
        x_initial = x_initial.to_numpy()

        x_unlabeled = train_data[
            train_data.index.get_level_values("node_id").isin(unlabeled_pool_labels.index)
        ]
        y_unlabeled = unlabeled_pool_labels  # ['anom'].to_numpy()
        x_unlabeled = x_unlabeled.to_numpy()

        # Initializations
        macro_f1_scores = []
        anomaly_miss_rates = []
        false_alarm_rates = []

        if query_strategy != "random":
            selected_indices_apps = []
            selected_indices_anoms = []
            selected_indices_inputs = []

        X_pool = x_unlabeled.copy()
        y_pool = y_unlabeled.copy()
        y_pool_anom = y_pool["anom"].to_numpy()
        y_pool_app = y_pool["app_names"].to_numpy()
        y_pool_input = y_pool["input"].to_numpy()

        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            query_strategy=query_strategy_dict[query_strategy],
            X_training=x_initial,
            y_training=y_initial,
        )

        # logging.info("Test inputs label check: %s", test_inputs_label['input'].unique())
        y_pred = learner.predict(test_inputs_data.to_numpy())
        report_dict = classification_report(
            test_inputs_label["anom"].to_numpy(), y_pred, output_dict=True
        )
        macro_f1_scores.append(report_dict["macro avg"]["f1-score"])
        call_FAR_function(false_alarm_rates, anomaly_miss_rates, test_inputs_label, y_pred, conf)
        logging.info("Active learning query starts")

        for i in range(query_size):
            query_idx, query_sample = learner.query(X_pool)

            if query_strategy != "random":
                selected_indices_apps.append(y_pool_app[query_idx][0])
                selected_indices_anoms.append(y_pool_anom[query_idx][0])
                selected_indices_inputs.append(y_pool_input[query_idx][0])

            learner.teach(
                X=X_pool[query_idx].reshape(1, -1),
                y=y_pool_anom[query_idx].reshape(
                    1,
                ),
            )

            X_pool, y_pool_anom, y_pool_app = (
                np.delete(X_pool, query_idx, axis=0),
                np.delete(y_pool_anom, query_idx, axis=0),
                np.delete(y_pool_app, query_idx, axis=0),
            )
            y_pred = learner.predict(test_inputs_data.to_numpy())

            report_dict = classification_report(
                test_inputs_label["anom"].to_numpy(), y_pred, output_dict=True
            )
            macro_f1_scores.append(report_dict["macro avg"]["f1-score"])
            call_FAR_function(
                false_alarm_rates, anomaly_miss_rates, test_inputs_label, y_pred, conf
            )

        for j in range(len(macro_f1_scores)):
            scores = scores.append(
                {
                    "query_iter": j,
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

        train_input_names = "-".join(train_inputs)
        test_input_names = "-".join(test_inputs)

        filename = f"train:{train_input_names}#test:{test_input_names}#{FE_NAME}#{NUM_FEATURE}#{method}#{query_strategy}#{query_size}#{selected_classifier.__class__.__name__}#{repeat_num}.csv"
        scores.to_csv(Path(conf["results_dir"]) / filename)

        logging.info("Saving: %s", filename)

        if query_strategy != "random":
            selected_app_anom_df = pd.DataFrame()
            selected_app_anom_df["apps"] = selected_indices_apps
            selected_app_anom_df["anoms"] = selected_indices_anoms
            selected_app_anom_df["inputs"] = selected_indices_inputs
            selected_app_anom_df.to_csv(
                Path(conf["results_dir"])
                / f"train:{train_input_names}#test:{test_input_names}#{FE_NAME}#{NUM_FEATURE}#{method}#{query_strategy}#{query_size}#{selected_classifier.__class__.__name__}#{repeat_num}#app-anom-selection.csv",
                index=False,
            )
            logging.info("Saved selected apps and anoms")


if __name__ == "__main__":
    main()
