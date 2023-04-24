## a simple usecase on the uwcse

import os
import requests
from re3py.data.task_settings import *
from re3py.data.data_and_statistics import *
from re3py.learners.core.heuristic import *
from re3py.eval.evaluation import *
from re3py.utilities.cross_validation import *
from re3py.learners.random_forest import *

import pytest


def download_data(dataset_name, experiment_dir, folds_dir):
    print(f"Downloading {dataset_name} data ...")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(folds_dir, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/re3py/re3py/master_with_resources/data"
    base_url_datasets = f"{base_url}/datasets/{dataset_name}/"
    url_folds = f"{base_url}/folds/{dataset_name}/folds1.txt"
    url_and_path = []
    for extension in ["_descriptive.txt", "_target.txt", ".s"]:
        file_name = f"{dataset_name}{extension}"
        url = f"{base_url_datasets}{file_name}"
        path = os.path.join(experiment_dir, file_name)
        url_and_path.append((url, path))
    path_folds = os.path.join(folds_dir, "folds1.txt")
    url_and_path.append((url_folds, path_folds))
    for url, path in url_and_path:
        r = requests.get(url)
        if r.ok:
            with open(path, "w", encoding="utf-8") as f:
                print(r.text, file=f)
        else:
            raise ValueError(f"Could not download {url} to {path}")


all_relevant = [1, 2]
depths = [2, 3]
treenums = [3, 6]


@pytest.mark.parametrize("tnum", all_relevant)
@pytest.mark.parametrize("depth", depths)
@pytest.mark.parametrize("treenum", treenums)
def test_rf_classification(tnum, depth, treenum):
    dataset_name = "uwcse"
    print(os.listdir())
    experiment_dir = f"./data/datasets/{dataset_name}/"
    folds_dir = f"./data/folds/{dataset_name}"
    if not os.path.exists(experiment_dir):
        download_data(dataset_name, experiment_dir, folds_dir)
    descriptive = experiment_dir + '{}_descriptive.txt'.format(dataset_name)
    target = experiment_dir + '{}_target.txt'.format(dataset_name)
    s_file = experiment_dir + '{}.s'.format(dataset_name)
    s = Settings(s_file)
    d = Dataset(s_file, descriptive, target)
    tree_params = {
        'heuristic': HeuristicGini(),
        'max_number_atom_tests': tnum,
        'allowed_atom_tests': s.get_atom_tests_structured(),
        'allowed_aggregators': s.get_aggregates(),
        'minimal_examples_in_leaf': 1,
        'java_port': None,  # 22222,
        'max_depth': depth,
        "per_class_bootstrap": True,
        "only_existential": True
    }
    target_values = list(
        {element.get_target()
         for element in d.get_target_data()})
    a_train_test = [Accuracy(target_values), Accuracy(target_values)]
    dataset_folds = create_folds(
        d, folds_file="./data/folds/{}/folds1.txt".format(dataset_name))
    experiment_name = "test"
    for fold, (train_set, test_set) in enumerate(dataset_folds):
        rf = RandomForest(treenum,
                          **tree_params)  # DecisionTree(**tree_params)  #
        rf.fit(train_set)
        rf.dump_to_text('{}{}.txt'.format(experiment_name, fold))
        true_values_train = [
            e.get_target() for e in train_set.get_target_data()
        ]
        true_values_test = [e.get_target() for e in test_set.get_target_data()]
        true = [true_values_train, true_values_test]
        predicted_values_train = [
            rf.predict(e) for e in train_set.get_target_data()
        ]
        predicted_values_test = [
            rf.predict(e) for e in test_set.get_target_data()
        ]
        predicted = [predicted_values_train, predicted_values_test]
        for a, t, p in zip(a_train_test, true, predicted):
            a.add_many(t, p)
            a.evaluate()
            print(a.measure_value)
