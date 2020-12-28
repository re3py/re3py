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
import argparse


print("Starting experiment")


parser = argparse.ArgumentParser(description='Option parser',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-port", "--port", default=None)
parser.add_argument("-fold", "--fold", type=int)
parser.add_argument("-model", "--model", choices=["RF", "GB", "DT"])
parser.add_argument("-rs", "--random_seed", type=int)
parser.add_argument("-shrinkage", "--shrinkage", type=float)
parser.add_argument("-step", "--step_size", type=float)
parser.add_argument("-ce", "--chosen_examples", type=float)
parser.add_argument("-mret", "--max_relative_number_of_evaluated_tests_per_node")
parser.add_argument("-d", "--depth", type=int)
parser.add_argument("-at", "--atom_tests", type=int)


arguments = parser.parse_args(sys.argv[1:])

port = arguments.port
if port == "None":
    port = None
if port is not None:
    port = int(port)
fold = arguments.fold
model = arguments.model
random_seed = arguments.random_seed
atom_tests = arguments.atom_tests


max_relative_number_of_evaluated_tests_per_node = "sqrt"
try:
    max_relative_number_of_evaluated_tests_per_node = float(arguments.max_relative_number_of_evaluated_tests_per_node)
except ValueError:
    pass

if model == "GB":
    shrinkage = arguments.shrinkage
    step_size = arguments.step_size
    chosen_examples = arguments.chosen_examples
    depth = arguments.depth
else:
    shrinkage = None
    step_size = None
    chosen_examples = None
    depth = 9999999

descriptive = 'relations_descriptive.txt'
target = "relation_target.txt"
s_file = 'settings.s'
s = Settings(s_file)
d = Dataset(s_file, descriptive, target)
tree_params = {'heuristic': HeuristicGini(),
               'max_number_internal_nodes': float("inf"),
               'max_number_atom_tests': atom_tests,
               'allowed_atom_tests': s.get_atom_tests_structured(),
               'allowed_aggregators': s.get_aggregates(),
               'minimal_examples_in_leaf': 1,
               'max_number_of_evaluated_tests_per_node': float("inf"),
               'max_relative_number_of_evaluated_tests_per_node': max_relative_number_of_evaluated_tests_per_node,
               'java_port': port,
               'max_depth': depth,
               'random_seed': random_seed,
               'per_class_bootstrap': True,
               'only_existential': True
               }
target_values = list({element.get_target() for element in d.get_target_data()})
a_train_test = [Accuracy(target_values), Accuracy(target_values)]

if os.path.exists("folds.txt"):
    dataset_folds = list(create_folds(d, folds_file="folds.txt"))
else:
    dataset_folds = list(create_folds(d, n_folds=10))
train_set, test_set = dataset_folds[fold]

if model == "RF":
    learner = RandomForest(50, **tree_params)
elif model == "GB":
    learner = GradientBoosting(50, shrinkage=shrinkage, step_size=step_size, chosen_examples=chosen_examples,
                               **tree_params)
elif model == "DT":
    print("Building tree with seed {}".format(random_seed))
    train_set = train_set.bootstrap_replicate(random_seed, per_class=tree_params["per_class_bootstrap"])
    learner = DecisionTree(**tree_params)
else:
    raise ValueError("Wrong model: {}".format(model))

parameters = {x: y for x, y in learner.__dict__.items()}
learner.build(train_set)
learner.print_model('experiment_model_fold{}.txt'.format(fold))
# learner.save_model('experiment_model_fold{}.sav'.format(fold))
try:
    learner.compute_ranking(EnsembleRanking.genie3).print_ranking("experiment_genie3.txt")
except AttributeError:
    print("Learner for the model {} does not support ranking.".format(model))
except:
    print("Something else is wrong")

true_values_train = [e.get_target() for e in train_set.get_target_data()]
true_values_test = [e.get_target() for e in test_set.get_target_data()]

true = [true_values_train, true_values_test]
predicted_values_train = [learner.predict(e, True).get_per_class_probabilities() for e in train_set.get_target_data()]
predicted_values_test = [learner.predict(e, True).get_per_class_probabilities() for e in test_set.get_target_data()]
predicted = [predicted_values_train, predicted_values_test]
class_values = learner.predict(test_set.get_target_data()[0], True).get_class_names()

with open("experiment.log", "w") as f:
    print(class_values, file=f)
    for name, p in zip(["train", "test"], predicted):
        print(name, file=f)
        print(p, file=f)
    print(parameters, file=f)
