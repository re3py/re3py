import subprocess
import os
from evaluation import Accuracy


def try_unzip(directory, files_to_find, tar_gz_archive, tar_archive=None):
    if tar_archive is None:
        tar_archive = tar_gz_archive[:-3]
    files_found = []
    files_full = [os.path.join(directory, file) for file in files_to_find]
    if all(os.path.exists(file_full) for file_full in files_full):
        return True, files_full
    if not os.path.exists(directory):
        return False, []
    already_present = [s for s in os.listdir(directory)]
    seven_zip = r'"C:\Program Files\7-Zip\7z.exe"'
    subprocess.call("{0} e {1}/{2} -o{3} -aoa".format(seven_zip, directory, tar_gz_archive, directory))
    if os.path.exists("{}/{}".format(directory, tar_archive)):
        subprocess.call("{0} e {1}/{2} -o{3} -aoa".format(seven_zip, directory, tar_archive, directory))
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
    return len(files_to_find) == len(files_found), files_found


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
    log_files = []
    for fold in range(10):
        fold_dir = os.path.join(data_dir, "fold{}/results".format(fold))
        files_to_collect = ["experiment_model_fold{}.txt".format(fold),
                            "experiment.log",
                            "experiment_genie3.txt"][1:-1]

        is_ok, file_path = try_unzip(fold_dir, files_to_collect, "experiment.tar.gz")
        if not is_ok:
            print("Something missing for", fold_dir)
            break
        log_files.append(os.path.join(fold_dir, "experiment.log"))
    if len(log_files) != 10:
        return -1
    a_train, a_test = None, None
    for log_file in log_files:
        a_train_part, a_test_part = load_accuracy(log_file)
        if a_train is None:
            a_train = a_test_part
            a_test = a_test_part
        else:
            a_train += a_train_part
            a_test += a_test_part
    # a_test = a_train
    n = a_test.examples
    default_a_test = max(sum(line) / n for line in a_test.confusion_matrix)
    print(data_dir, a_test, a_test.confusion_matrix, default_a_test)


def collect_results(model_dir):
    for dataset in os.listdir(model_dir):
        full_path = os.path.join(model_dir, dataset)
        if os.path.isdir(full_path):
            collect_results_for_dataset(full_path)


if 1:
    collect_results("../experiments/single_tree")

