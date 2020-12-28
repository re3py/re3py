from core import DecisionTree
from random_forest import RandomForest
from boosting import GradientBoosting
from heuristic import *
from data import Dataset
# from table_to_relations import arff_to_relations
# import time
from ensemble_ranking import EnsembleRanking
from evaluation import Accuracy
from cross_validation import create_folds
import sys

print("Starting GS experiment")

port = None  # int(sys.argv[1])
fold = int(sys.argv[2])
shrinkage = float(sys.argv[3])
step_size = float(sys.argv[4])
chosen_examples = float(sys.argv[5])
max_relative_number_of_evaluated_tests_per_node = "sqrt"
try:
    max_relative_number_of_evaluated_tests_per_node = float(sys.argv[6])
except ValueError:
    pass
depth = int(sys.argv[7])

descriptive = 'relations_descriptive.txt'
target = "relation_target.txt"
s_file = 'settings.s'
s = Settings(s_file)
d = Dataset(s_file, descriptive, target)
tree_params = {'heuristic': HeuristicGini(),
               # 'max_number_internal_nodes': 500,
               'max_number_atom_tests': 2,
               'allowed_atom_tests': s.get_atom_tests_structured(),
               'allowed_aggregators': s.get_aggregates(),
               'minimal_examples_in_leaf': 1,
               # 'max_number_of_evaluated_tests_per_node': 1000,
               'max_relative_number_of_evaluated_tests_per_node': max_relative_number_of_evaluated_tests_per_node,
               'java_port': port,
               'max_depth': depth,
               # "only_existential": True
               }

target_values = list({element.get_target() for element in d.get_target_data()})
a_train_test = [Accuracy(target_values), Accuracy(target_values)]

dataset_folds = list(create_folds(d, folds_file="folds.txt"))
train_set_all, _ = dataset_folds[fold]


for train_set, test_set in create_folds(train_set_all, n_folds=3):
    learner = GradientBoosting(50, shrinkage=shrinkage, step_size=step_size, chosen_examples=chosen_examples,
                               **tree_params)
    learner.build(train_set)

    true_values_train = [e.get_target() for e in train_set.get_target_data()]
    true_values_test = [e.get_target() for e in test_set.get_target_data()]

    true = [true_values_train, true_values_test]
    predicted_values_train = [learner.predict(e) for e in train_set.get_target_data()]
    predicted_values_test = [learner.predict(e) for e in test_set.get_target_data()]
    predicted = [predicted_values_train, predicted_values_test]
    for a, t, p in zip(a_train_test, true, predicted):
        a.add_many(t, p)

with open("experiment.log", "w") as f:
    for name, a in zip(["train", "test"], a_train_test):
        print(name, file=f)
        a.evaluate()
        print(a.measure_value, file=f)
        print(a.class_indices, file=f)
        print(a.confusion_matrix, file=f)
