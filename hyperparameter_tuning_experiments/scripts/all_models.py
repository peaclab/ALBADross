#!/usr/bin/env python
# coding: utf-8

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier

import sys

sys.path.insert(0, "../../../src/")

import pandas as pd
import os, sys
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

# In-house Module Imports
from config import Configuration
from datasets import EclipseSampledDataset, VoltaSampledDataset
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from utils import *
import re

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)
import json

### new ML models
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


def main():
    ### Settings
    SYSTEM = sys.argv[1]  # volta or eclipse
    OUTER_DIR = sys.argv[
        2
    ]  # ex: active_learning_experiments, active_learning_experiments_final_hdfs
    MODEL = sys.argv[3]  # logistic_regression, random_forest, lgbm, mlp
    OUTPUT_DIR = f"/projectnb/peaclab-mon/aksar/{OUTER_DIR}"
    num_samples_per_pair = 1
    NUM_FEATURE = 2000
    CV_INDEX = 0
    FS_NAME = "CHI"
    SCALER = (  # For now, do the scaling inside the notebook, then you can move that to the class function
        "None"
    )
    FEATURE_SELECTION = False
    if SYSTEM == "volta":
        FE_NAME = "tsfresh"
        EXP_NAME = "tsfresh_experiments"
    elif SYSTEM == "eclipse":
        FE_NAME = "mvts"
        EXP_NAME = "mvts_experiments"

    MODEL_CONFIG = "tuning_results"  # rf_tuncer or rf_tuncer_worst_case
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
    logging.info("Test data label dist: \n%s", test_label["anom"].value_counts()),

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

    ### define parameter search space
    if MODEL == "random_forest":
        param_grid = {
            "n_estimators": [8, 10, 20, 100, 200],
            "max_depth": [None, 4, 8, 10, 20],
            "criterion": ["gini", "entropy"],
        }

        clf = RandomForestClassifier(random_state=42)

    elif MODEL == "logistic_regression":

        param_grid = {
            "penalty": ["l1", "l2"],
            "C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "solver": ["liblinear"],
        }

        clf = LogisticRegression(random_state=0, dual=False, max_iter=12000)

    elif MODEL == "mlp":

        param_grid = {
            "max_iter": [100, 200, 500, 1000],
            "hidden_layer_sizes": [(10, 10, 10), (30, 20, 10), (50, 100, 50), (100)],
            "alpha": [0.0001, 0.001, 0.01],
        }

        clf = MLPClassifier(random_state=1)

    elif MODEL == "lgbm":

        param_grid = {
            "num_leaves": [2, 8, 31, 128],
            "learning_rate": [0.01, 0.1, 0.3],
            "max_depth": [-1, 2, 8],
            "colsample_bytree": [0.5, 1.0],
        }
        clf = LGBMClassifier(objective="multiclass", random_state=5)
        train_data = train_data.values

    else:
        raise ("Invalid classifier")

    logging.info(f"Tunning {MODEL}...")
    clf.fit(train_data, train_label["anom"])  # previously we were giving x_initial, y_initial
    pred = clf.predict(test_data)
    initial_report_dict = classification_report(test_label["anom"], pred, output_dict=True)
    print(
        f"Inıtial Macro-Avg  F-1 for {MODEL} on Test data: ",
        initial_report_dict["macro avg"]["f1-score"],
    )
    CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring="f1_macro")
    CV_clf.fit(train_data, train_label["anom"])
    logging.info(CV_clf.best_params_)
    pred = CV_clf.predict(test_data)
    final_report_dict = classification_report(test_label["anom"], pred, output_dict=True)
    print(
        f"Tuned Macro-Avg  F-1 for {MODEL} on Test data: ",
        final_report_dict["macro avg"]["f1-score"],
    )
    CV_clf.best_params_["initial_f1_score"] = initial_report_dict["macro avg"]["f1-score"]
    CV_clf.best_params_["tuned_f1_score"] = final_report_dict["macro avg"]["f1-score"]

    jsonpath = conf["results_dir"] / f"{MODEL}_Best_Params.json"
    jsonpath.write_text(json.dumps(CV_clf.best_params_))


if __name__ == "__main__":
    main()
