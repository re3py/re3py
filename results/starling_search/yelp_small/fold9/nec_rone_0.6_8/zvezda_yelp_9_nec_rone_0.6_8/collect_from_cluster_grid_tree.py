import subprocess
import os
from evaluation import Accuracy, Precision, Recall
import pandas as pd
import re
import itertools


IMPURITIES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
LEAF_SIZES = [1, 5, 10, 15, 20]

combinations = [(str(impurity), str(leaf)) for impurity, leaf in itertools.product(IMPURITIES, LEAF_SIZES)]


def try_unzip(directory, files_to_find, tar_gz_archive, tar_archive=None):
    if tar_archive is None:
        tar_archive = tar_gz_archive[:-3]
    if not os.path.exists(directory):
        return False, []
    files_found = []
    files_full = [os.path.join(directory, file) for file in files_to_find]
    if all(os.path.exists(file_full) for file_full in files_full):
        return True, files_full
    already_present = [s for s in os.listdir(directory)]
    seven_zip = r'"C:\Program Files\7-Zip\7z.exe"'
    subprocess.call("{0} e {1}/{2} -o{3}".format(seven_zip, directory, tar_gz_archive, directory))
    if os.path.exists("{}/{}".format(directory, tar_archive)):
        subprocess.call("{0} e {1}/{2} -o{3}".format(seven_zip, directory, tar_archive, directory))
    else:
        print("{}/{} missing".format(directory, tar_archive))
        raise ZeroDivisionError
    for s in os.listdir(directory):
        if s in files_to_find:
            files_found.append(os.path.join(directory, s))
        elif s in already_present:
            pass
        else:
            os.remove(directory + "/" + s)
    if len(files_to_find) == len(files_found):
        return True, files_full  # keep same ordering
    else:
        return False, files_found


def load_accuracy(log_file):
    """
    This is the format of log files:
    train
    1.0
    {'N': 0, 'A': 1}
    [[59, 0], [0, 26]]
    test
    1.0
    {'N': 0, 'A': 1}
    [[7, 0], [0, 3]]

    We load both confusion matrices to Accuracy objects.
    :param log_file: path to the file
    :return:
    """

    def create_one(i0):
        classes = eval(lines[i0])
        matrix = eval(lines[i0 + 1])
        classes = sorted(classes, key=lambda c: classes[c])
        return Accuracy(classes, confusion_matrix=matrix)

    with open(log_file) as f:
        lines = [l.strip() for l in f.readlines()]

    a_train = create_one(2)
    a_test = create_one(6)
    return a_train, a_test


def collect_results_for_dataset(data_dir):
    point_pattern = "impurity(.+)_leaf([0-9]+)"
    best_configurations = []
    for fold in range(10):
        log_files = {}
        results_dir = os.path.join(data_dir, "fold{}".format(fold), "results")
        files_to_collect = [f"experiment_impurity{impurity}_leaf{leaf}.log" for impurity, leaf in combinations]
        is_ok, file_paths = try_unzip(results_dir, files_to_collect, "experiment.tar.gz")
        if not is_ok:
            print("Something missing for", results_dir)
            continue
        else:
            # previous_a_test = None
            for file_path in file_paths:
                _, a_test = load_accuracy(file_path)
                # c_matrix = a_test.confusion_matrix
                # if previous_a_test is not None:
                #     n_classes = len(a_test.confusion_matrix)
                #     a_test.confusion_matrix = [[a_test.confusion_matrix[i][j] - previous_a_test.confusion_matrix[i][j]
                #                                 for j in range(n_classes)] for i in range(n_classes)]
                #     a_test.examples -= sum(sum(line) for line in previous_a_test.confusion_matrix)
                a_test.evaluate()
                grid_point_object = re.search(point_pattern, file_path)
                grid_point = (grid_point_object.group(1), grid_point_object.group(2))
                log_files[grid_point] = a_test.measure_value
                # previous_a_test = a_test
                # previous_a_test.confusion_matrix = c_matrix
        best = max(log_files.items(), key=lambda pair: pair[1])
        best_configurations.append(best)
    return best_configurations


def collect_results(model_dir, datasets=None):
    best_configurations = {}
    for dataset in os.listdir(model_dir):
        if dataset is not None and dataset not in datasets:
            print("Skipping", dataset)
            continue
        full_path = os.path.join(model_dir, dataset)
        if os.path.isdir(full_path):
            best_configurations[dataset] = collect_results_for_dataset(full_path)
    with open(os.path.join(model_dir, "best_configurations.txt"), "w") as f:
        print(best_configurations, file=f)


def collect_all_results_for_dataset(data_dir):
    all_configurations = {}
    for fold in range(10):
        log_files = {}
        fold_dir = os.path.join(data_dir, "fold{}".format(fold))
        configuration_dirs = [(os.path.join(fold_dir, d, "results"), d)
                              for d in os.listdir(fold_dir) if os.path.isdir(os.path.join(fold_dir, d))]
        for configuration_dir, configuration in configuration_dirs:
            c = tuple(configuration[len("tocka"):].split('_'))
            log_files[c] = -1.0
            files_to_collect = ["experiment.log"]
            is_ok, file_path = try_unzip(configuration_dir, files_to_collect, "experiment.tar.gz")
            if not is_ok:
                print("Something missing for", configuration_dir)
                continue
            else:
                _, a_test = load_accuracy(os.path.join(configuration_dir, "experiment.log"))
                a_test.evaluate()
                log_files[c] = a_test.measure_value
        # best = max(log_files.items(), key=lambda pair: pair[1])
        # all_configurations.append(best)
        all_configurations[fold] = log_files
    return all_configurations


def collect_all_results(model_dir):
    # create date frame
    columns = ["data set", "fold", "shrinkage", "row proportion", "evaluated splits", "depth", "accuracy"]
    matrix = []
    for dataset in os.listdir(model_dir):
        full_path = os.path.join(model_dir, dataset)
        if os.path.isdir(full_path):
            results = collect_all_results_for_dataset(full_path)
            for fold, configurations in results.items():
                for configuration, acc in configurations.items():
                    shrinkage, _, rows, splits, depth = configuration
                    matrix.append([dataset, fold, shrinkage, rows, splits, depth, acc])
    pd.DataFrame(matrix, columns=columns).to_csv(os.path.join(model_dir, "all_configurations.txt"),
                                                 index=False)


if __name__ == "__main__":
    # collect_all_results("../experiments/boosting_search")
    collect_results("../experiments/onlyTar_single_tree_search", datasets=["webkb"])
    pass
