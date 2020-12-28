import sys
import re
import os
import shutil
import subprocess
import numpy as np
from evaluation import Accuracy
from prepare_for_starling import *


# input:
# - settings (already converted) - actually background file
# - discrete versions of descriptive
# - original target (need to be converted (to binary etc.))
# - jar
#
# actual input:
# - fold
# - algorithm parameters


def creat_background_pointer(directory):
    """
    background-pointer file
    :param directory: either "train" or "test"
    :return:
    """
    with open(os.path.join(directory, f"{directory}_bk.txt"), "w", newline="") as f:
        print('import: "../background.txt".', file=f)


def copy_facts(directory):
    """As above"""
    shutil.copyfile(descriptive, os.path.join(directory, f"{directory}_facts.txt"))


def copy_target(directory, negative_examples, positive_examples):
    shutil.copyfile(negative_examples, os.path.join(directory, f"{directory}_neg.txt"))
    shutil.copyfile(positive_examples, os.path.join(directory, f"{directory}_pos.txt"))


def target_relation_name():
    with open(target) as f:
        line = f.readline()
    return line[:line.find("(")].strip()


def update_settings():
    with open(background) as f:
        lines = [line.strip() for line in f.readlines()]
    pairs = [("maxTreeDepth", depth), ("nodeSize", 2)]
    new_lines = [f"setParam: {param}={value}." for param, value in pairs] + lines
    with open(background, "w", newline="") as f:
        print("\n".join(new_lines), file=f)


def read_predictions():
    prediction_dir = os.path.join(train_dir, "models", "bRDNs")
    prediction_file = ""
    for file in os.listdir(prediction_dir):
        if file.startswith("predictions") and file.endswith(".csv"):
            prediction_file = os.path.join(prediction_dir, file)
    assert prediction_file
    p = []
    with open(prediction_file) as f:
        for line in f:
            if line.strip():
                p.append(float(re.search(", ([^,]+),", line).group(1)))
    return p


arguments = sys.argv[1:]
fold = int(arguments[0])
step = float(arguments[1])
depth = int(arguments[2])

background = "background.txt"
descriptive = 'relations_descriptive.txt'
target = "relation_target.txt"
folds = "folds.txt"

train_dir = "train"
test_dir = "test"

update_settings()
target_relation = target_relation_name()

train_command = f"java -jar BoostSRL.jar -l -train train/ -target {target_relation} -trees 50 -step {step}"
predict_command = f"java -jar BoostSRL.jar -i -model train/models -test test/ -target {target_relation} -trees 50"

true_values, class_values = get_possible_values(target)
predictions = None
test_ids = None
for i, class_value in enumerate(class_values):
    print("Class", i, "of", len(class_values))
    # binarize
    target_neg, target_pos = binarize_target(target, class_value)
    # divide target file
    (train_neg, test_neg), test_ids = train_test_split(target_neg, folds, fold)
    (train_pos, test_pos), _ = train_test_split(target_pos, folds, fold)
    move_neg_test_examples_to_pos(test_pos, test_neg)
    # create standard file structure
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    # background
    creat_background_pointer(train_dir)
    creat_background_pointer(test_dir)
    # facts
    copy_facts(train_dir)
    copy_facts(test_dir)
    # neg and pos
    copy_target(train_dir, train_neg, train_pos)
    copy_target(test_dir, test_neg, test_pos)
    # run training and create predictions
    subprocess.call(train_command, shell=True, stdout=subprocess.DEVNULL)
    subprocess.call(predict_command, shell=True, stdout=subprocess.DEVNULL)
    # read predictions
    class_probabilities = read_predictions()
    if predictions is None:
        predictions = np.zeros((len(class_probabilities), len(class_values)))
    predictions[:, i] = class_probabilities

    shutil.rmtree(train_dir)
    shutil.rmtree(test_dir)
# skip training examples
true_values = [true_values[test_id] for test_id in test_ids]
# aggregate predictions
max_class_indices = np.argmax(predictions, axis=1)
predicted_values = [class_values[i] for i in max_class_indices]
# create standard output file
a_train_test = [Accuracy(class_values), Accuracy(class_values)]
a_train_test[0].add_one(class_values[0], class_values[0])  # fake it
a_train_test[1].add_many(true_values, predicted_values)
with open("experiment.log", "w") as f:
    for name, a in zip(["train", "test"], a_train_test):
        print(name, file=f)
        a.evaluate()
        print(a.measure_value, file=f)
        print(a.class_indices, file=f)
        print(a.confusion_matrix, file=f)


