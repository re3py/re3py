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

print("Starting experiment")

max_relative_number_of_evaluated_tests_per_node = "sqrt"

port = None  # int(sys.argv[1])
fold = int(sys.argv[2])
model = sys.argv[3]
assert model in ["RF", "GB", "DT"]
if len(sys.argv) <= 5:
    try:
        max_relative_number_of_evaluated_tests_per_node = float(sys.argv[4])
    except ValueError:
        pass
    depth = 100000
    shrinkage = None
    step_size = None
    chosen_examples = None
elif len(sys.argv) == 9:
    assert model == "GB", "Wrong model: {}".format(model)
    shrinkage = float(sys.argv[4])
    step_size = float(sys.argv[5])
    chosen_examples = float(sys.argv[6])

    try:
        max_relative_number_of_evaluated_tests_per_node = float(sys.argv[7])
    except ValueError:
        pass
    depth = int(sys.argv[8])
else:
    raise ValueError("Wrong arguments")

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
               "only_existential": True
               }
target_values = list({element.get_target() for element in d.get_target_data()})
a_train_test = [Accuracy(target_values), Accuracy(target_values)]

if os.path.exists("folds.txt"):
    dataset_folds = list(create_folds(d, folds_file="folds.txt"))
else:
    dataset_folds = list(create_folds(d, n_folds=10))
train_set, test_set = dataset_folds[fold]

# rf = GradientBoosting(3, 1.0, **tree_params)  # DecisionTree(**tree_params)  #
# b.build(d)
# b.print_model('output_boosting.txt')
# b.save_model('output_boosting.sav')
# b = GradientBoosting.load_model('output_boosting.sav')
# tree_params['max_depth'] = 7

if model == "RF":
    learner = RandomForest(50, **tree_params)
elif model == "GB":
    learner = GradientBoosting(50, shrinkage=shrinkage, step_size=step_size, chosen_examples=chosen_examples,
                               **tree_params)
elif model == "DT":
    learner = DecisionTree(**tree_params)
else:
    raise ValueError("Wrong model: {}".format(model))

# rf = RandomForest(50, **tree_params)  # DecisionTree(**tree_params)  #
learner.build(train_set)
learner.print_model('experiment_model_fold{}.txt'.format(fold))
learner.save_model('experiment_model_fold{}.sav'.format(fold))
try:
    learner.compute_ranking(EnsembleRanking.genie3).print_ranking("experiment_genie3.txt")
except AttributeError:
    print("Learner for the model {} does not support ranking.".format(model))
except:
    print("Something else is wrong")

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
