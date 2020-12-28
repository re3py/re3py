# from core import DecisionTree
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


print("Starting experiment")

port = int(sys.argv[1])
fold = int(sys.argv[2])

descriptive = 'relations_descriptive.txt'
target = "relation_target.txt"
s_file = 'settings.s'
s = Settings(s_file)
d = Dataset(s_file, descriptive, target)
tree_params = {'heuristic': HeuristicGini(),
               'max_number_internal_nodes': 500,
               'max_number_atom_tests': 1,
               'allowed_atom_tests': s.get_atom_tests_structured(),
               'allowed_aggregators': s.get_aggregates(),
               'minimal_examples_in_leaf': 1,
               'max_number_of_evaluated_tests_per_node': 1000,
               'java_port': port,
               'max_depth': 50}
target_values = list({element.get_target() for element in d.get_target_data()})
a_train_test = [Accuracy(target_values), Accuracy(target_values)]

dataset_folds = list(create_folds(d, folds_file="folds.txt"))
train_set, test_set = dataset_folds[fold]

# rf = GradientBoosting(3, 1.0, **tree_params)  # DecisionTree(**tree_params)  #
# b.build(d)
# b.print_model('output_boosting.txt')
# b.save_model('output_boosting.sav')
# b = GradientBoosting.load_model('output_boosting.sav')
# tree_params['max_depth'] = 7
rf = RandomForest(50, **tree_params)  # DecisionTree(**tree_params)  #
rf.build(train_set)
rf.print_model('experiment_model_fold{}.txt'.format(fold))
rf.save_model('experiment_model_fold{}.sav'.format(fold))
rf.compute_ranking(EnsembleRanking.genie3).print_ranking("experiment_genie3.txt")

true_values_train = [e.get_target() for e in train_set.get_target_data()]
true_values_test = [e.get_target() for e in test_set.get_target_data()]

true = [true_values_train, true_values_test]
predicted_values_train = [rf.predict(e) for e in train_set.get_target_data()]
predicted_values_test = [rf.predict(e) for e in test_set.get_target_data()]
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
