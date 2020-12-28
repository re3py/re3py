from typing import Dict
from .relation import *
from .task_settings import Settings
import random
from ..utilities.my_utils import arg_max
import copy
import numpy as np
import os


class Datum:
    def __init__(self, descriptive_part, target_part, weight, identifier):
        self.descriptive_part = descriptive_part
        self.target_part = target_part
        self.weight = weight
        self.identifier = identifier

    def __repr__(self):
        return "Datum({}, {}, {})".format(self.descriptive_part,
                                          self.target_part, self.weight)

    def get_weight(self):
        return self.weight

    def set_weight(self, w):
        self.weight = w

    def get_descriptive(self):
        return self.descriptive_part

    def get_target(self):
        return self.target_part

    def set_target(self, t):
        self.target_part = t


class Dataset:
    def __init__(self,
                 s_file=None,
                 data_file=None,
                 target_file=None,
                 settings=None,
                 descriptive_relations=None,
                 target_data=None,
                 statistics=None,
                 nb_target_instances=float('inf'),
                 target_type=None):
        self.settings = settings
        self.descriptive_relations = descriptive_relations  # type: Dict[str, Relation]
        self.target_data = [] if target_data is None else target_data  # type: List['Datum']
        self.statistics = statistics
        self.number_target_instances = nb_target_instances
        self.data_file = os.path.abspath(
            data_file) if data_file is not None else None
        self.target_file = os.path.abspath(
            target_file) if target_file is not None else None

        # settings i.e., meta data
        if s_file is not None:
            self.settings = Settings(s_file)
            self.descriptive_relations = {
                r.get_name(): r
                for r in self.settings.get_relations()
            }
        # target type
        if self.settings is None:
            self.target_type = target_type
        else:
            self.target_type = self.settings.get_relations()[0].types[-1]
            if target_type is not None:
                assert self.target_type == target_type
        # data
        all_relations_empty = not any(
            r.all_tuples for r in self.descriptive_relations.values())
        if data_file is not None and all_relations_empty:
            self.read_relations_from_file(data_file)
        if target_file is not None and all_relations_empty:
            self.target_data = []
            self.read_target_from_file(target_file,
                                       self.get_target_relation().get_name())
        # statistics

        if len(self.target_data) > 0:
            if self.statistics is None:
                self.compute_statistics_for_data()
            else:
                self.statistics.reset()
                self.statistics.add_examples(self.target_data)

    def __iter__(self):
        yield from self.target_data

    def __getitem__(self, i):
        return self.target_data[i]

    def get_target_data(self):
        return self.target_data

    def get_descriptive_data(self):
        return self.descriptive_relations

    def get_target_relation(self):
        return self.settings.get_relations()[0]

    def get_copy_statistics(self):
        return self.statistics.get_copy()

    def set_statistics(self, s):
        self.statistics = s

    def add_example(self, x):
        self.target_data.append(x)

    def add_examples(self, xs):
        self.target_data += xs

    def read_relations_from_file(self, file):
        with open(file) as f:
            for line_raw in f:
                i = line_raw.find('//')
                if i >= 0:
                    line = line_raw[:i]
                else:
                    line = line_raw.strip()
                if line:
                    r_name = parse_relation_name(line)
                    r = self.descriptive_relations[r_name]
                    r.try_add_tuple(line)

    def read_target_from_file(self, file, target_relation_name):
        added = 0
        with open(file) as f:
            for line_raw in f:
                i = line_raw.find('//')
                if i >= 0:
                    line = line_raw[:i]
                else:
                    line = line_raw.strip()
                if line:
                    r_name = parse_relation_name(line)
                    assert r_name == target_relation_name
                    target_type = self.descriptive_relations[
                        target_relation_name].types[-1]
                    example_target = parse_relation_arguments(line, r_name)
                    example, target = example_target[:-1], example_target[-1]
                    self.add_example(
                        Datum(tuple(example),
                              Relation.intelligent_parse(target_type, target),
                              1, added))
                    added += 1
                    if added == self.number_target_instances:
                        break

    def bootstrap_replicate(self, random_seed=25061991, per_class=False):
        r = random.Random(random_seed)
        c1 = isinstance(self.statistics, NodeStatisticsClassification)
        c2 = isinstance(self.statistics, NodeStatisticsClassificationBoosting)
        if per_class and (c1 or c2):
            classes = {}
            for i, d in enumerate(self.get_target_data()):
                t = d.get_target()
                if t not in classes:
                    classes[t] = []
                classes[t].append(i)
            classes = [classes[c]
                       for c in sorted(classes)]  # always the same order ...
        else:
            classes = [list(range(len(self.get_target_data())))]

        n = len(self.target_data)
        successes = [0] * n
        for class_indices in classes:
            for chosen in Dataset._bootstrap_replicate_one_class(
                    class_indices, r):
                successes[chosen] += 1
        new_target_data = []
        for i, nb_successes in enumerate(successes):
            if nb_successes > 0:
                datum = self.target_data[i]
                new_datum = Datum(datum.get_descriptive(), datum.target_part,
                                  datum.get_weight() * nb_successes,
                                  datum.identifier)
                new_target_data.append(new_datum)
        return Dataset(settings=self.settings,
                       data_file=self.data_file,
                       descriptive_relations=self.descriptive_relations,
                       target_data=new_target_data,
                       statistics=self.get_copy_statistics())

    @staticmethod
    def _bootstrap_replicate_one_class(indices, random_generator):
        n = len(indices)
        return [indices[int(n * random_generator.random())] for _ in range(n)]

    def compute_statistics_for_data(self):
        if Relation.is_nominal_type(self.target_type):
            s = NodeStatisticsClassification
        elif Relation.is_numeric_type(self.target_type):
            s = NodeStatisticsRegression
        elif Relation.is_multi_target_type(self.target_type):
            inner_type = Relation.get_inner_type_of_multi_target(
                self.target_type)
            if Relation.is_numeric_type(inner_type):
                s = NodeStatisticsMultitargetRegression
            else:
                raise TypeError("Wrong target type: {}".format(
                    self.target_type))
        else:
            raise TypeError("Wrong target type: {}".format(self.target_type))
        self.set_statistics(s.compute_stats(self.get_target_data()))


def compute_all_values_of_types(relations):
    # From descriptive data only
    values_per_type = {}
    for r in relations:
        for i, t in enumerate(r.types):
            if t not in values_per_type:
                values_per_type[t] = set()
            values_per_type[t] |= set(r.get_all_values(i))
    return values_per_type


def get_all_target_values(target_data):
    return {d.get_target() for d in target_data}


def target_data_weight(data, indices=None):
    if indices is None:
        indices = range(len(data))
    return sum(data[i].get_weight() for i in indices)


class NodeStatistics:
    def __init__(self,
                 total_nb_examples=0,
                 branch_frequencies=None,
                 variability=None):
        self.total_nb_examples = total_nb_examples
        self.branch_frequencies = branch_frequencies
        self.variability = variability
        self.prediction = None

    def set_total_number_examples(self, n):
        self.total_nb_examples = n

    def get_total_number_examples(self):
        return self.total_nb_examples

    def set_branch_frequencies(self, fs):
        self.branch_frequencies = fs

    def get_branch_frequencies(self):
        return self.branch_frequencies

    def set_variability(self, v):
        self.variability = v

    def get_variability(self):
        return self.variability

    def get_prediction(self):
        return self.prediction

    def get_prediction_for_ensemble(self):
        return self

    def get_copy(self):
        return copy.deepcopy(self)

    def set_prediction(self, p):
        self.prediction = p

    def reset(self):
        self.total_nb_examples = 0
        self.branch_frequencies = None
        self.variability = None
        self.prediction = None

    @staticmethod
    def compute_stats(data: List[Datum]):
        raise NotImplementedError("This should be implemented by a subclass.")

    @staticmethod
    def construct_from_parent(parent_stats: 'NodeStatistics'):
        raise NotImplementedError("This should be implemented by a subclass.")

    def add_example_during_split_eval(self, datum: Datum):
        raise NotImplementedError("This should be implemented by a subclass.")

    def remove_example_during_split_eval(self, datum: Datum):
        raise NotImplementedError("This should be implemented by a subclass.")

    def after_split_evaluation_update(self):
        raise NotImplementedError("This should be implemented by a subclass.")

    def create_predictions(self):
        raise NotImplementedError("This should be implemented by a subclass.")

    def add_examples(self, data: List[Datum]):
        raise NotImplementedError("This should be implemented by a subclass.")

    def add_other_for_ensemble_prediction(self, other: 'NodeStatistics',
                                          other_weight):
        raise NotImplementedError("This should be implemented by a subclass.")


class NodeStatisticsClassification(NodeStatistics):
    def __init__(self, class_names, **node_statistic_args):
        super().__init__(**node_statistic_args)
        self.class_names = sorted(class_names)
        self.nb_examples_per_class = []
        self.per_class_probabilities = []
        self.class_to_index = {}
        self.initialize_from_class_names()

    def __str__(self):
        prediction_str = "" if self.prediction is None else "return {}".format(
            self.prediction)
        return "{} ({}: {})".format(prediction_str, self.class_names,
                                    self.nb_examples_per_class)

    def initialize_from_class_names(self):
        self.nb_examples_per_class = [0] * len(self.class_names)
        self.per_class_probabilities = [0] * len(self.class_names)
        self.class_to_index = {n: i for i, n in enumerate(self.class_names)}

    def reset(self):
        super().reset()
        self.initialize_from_class_names()

    def add_other_for_ensemble_prediction(
            self, other: 'NodeStatisticsClassification', other_weight):
        assert self.class_names == other.class_names
        # update for proportion voting
        for i in range(len(self.class_names)):
            self.per_class_probabilities[
                i] += other.per_class_probabilities[i] * other_weight
        # update for majority voting
        self.nb_examples_per_class[self.class_to_index[
            other.prediction]] += other_weight

    @staticmethod
    def construct_from_parent(parent_stats: 'NodeStatisticsClassification'):
        return NodeStatisticsClassification(parent_stats.get_class_names())

    def get_nb_examples_per_class(self):
        return self.nb_examples_per_class

    def set_nb_examples_per_class(self, counts):
        self.nb_examples_per_class = counts

    def get_per_class_probabilities(self):
        return self.per_class_probabilities

    def get_class_names(self):
        return self.class_names

    def get_class_to_index(self):
        return self.class_to_index

    def update_per_class_probabilities(self):
        if self.total_nb_examples == 0:
            self.per_class_probabilities = [0] * len(self.class_names)
        else:
            self.per_class_probabilities = [
                c / self.total_nb_examples for c in self.nb_examples_per_class
            ]

    def update_total_nb_examples(self):
        self.total_nb_examples = sum(self.nb_examples_per_class)

    def update_dependent(self):
        self.update_total_nb_examples()
        self.update_per_class_probabilities()

    @staticmethod
    def compute_stats(data: List[Datum]):
        # at least the possible ones
        class_names = []
        counters = {c: 0.0 for c in class_names}
        for datum in data:
            c = datum.get_target()
            if c not in counters:
                class_names.append(c)
                counters[c] = 0.0
            counters[c] += datum.get_weight()
        class_names = sorted(class_names)
        counters = [counters[c] for c in class_names]
        s = NodeStatisticsClassification(class_names)
        s.set_nb_examples_per_class(counters)
        s.update_dependent()
        return s

    def add_examples(self, data: List[Datum]):
        for datum in data:
            self.add_example_during_split_eval(datum)
        self.update_dependent()

    def add_example_during_split_eval(self, datum: Datum):
        # We do as little as possible, since this will be called many times.
        # The other statistics are updated after split evaluation in one step.
        self.update_with_delta_weight(self.class_to_index[datum.get_target()],
                                      datum.get_weight())

    def remove_example_during_split_eval(self, datum: Datum):
        # Same comment apply :)
        self.update_with_delta_weight(self.class_to_index[datum.get_target()],
                                      -datum.get_weight())

    def update_with_delta_weight(self, class_index, weight):
        self.nb_examples_per_class[class_index] += weight
        self.total_nb_examples += weight
        self.update_per_class_probabilities()

    def after_split_evaluation_update(self):
        self.update_dependent()

    def create_predictions(self):
        self.prediction = self.class_names[arg_max(self.nb_examples_per_class)]


class NodeStatisticsRegression(NodeStatistics):
    def __init__(self, **node_statistic_args):
        super().__init__(**node_statistic_args)
        self.sum1 = 0.0  # sum of w_i y_i over the examples Datum(x_i, y_i, w_i)
        self.sum2 = 0.0  # sum of w_i y_i^2 over the same examples

    def __str__(self):
        prediction_str = "" if self.prediction is None else "return {}".format(
            self.prediction)
        return "{} ({} examples)".format(prediction_str,
                                         self.total_nb_examples)

    def reset(self):
        super().reset()
        self.sum1 = 0.0
        self.sum2 = 0.0

    def get_sum_of_values(self):
        return self.sum1

    def get_sum_of_squared_values(self):
        return self.sum2

    @staticmethod
    def compute_stats(data: List[Datum]):
        s = NodeStatisticsRegression()
        s.add_examples(data)
        return s

    @staticmethod
    def construct_from_parent(parent_stats: 'NodeStatisticsRegression'):
        return NodeStatisticsRegression()

    def add_example_during_split_eval(self, datum: Datum):
        self.update_with_delta_weight(datum.get_target(), datum.get_weight())

    def remove_example_during_split_eval(self, datum: Datum):
        self.update_with_delta_weight(datum.get_target(), -datum.get_weight())

    def update_with_delta_weight(self, target_value, weight):
        y = weight * target_value
        self.sum1 += y
        self.sum2 += y * target_value
        self.total_nb_examples += weight

    def after_split_evaluation_update(self):
        pass

    def create_predictions(self):
        self.prediction = self.sum1 / self.total_nb_examples

    def add_examples(self, data: List[Datum]):
        for datum in data:
            self.add_example_during_split_eval(
                datum)  # the same update happens

    def add_other_for_ensemble_prediction(self,
                                          other: 'NodeStatisticsRegression',
                                          other_weight):
        self.sum1 += other.sum1
        self.sum2 += other.sum2
        self.total_nb_examples += other_weight


class NodeStatisticsMultitargetRegression(NodeStatisticsRegression):
    # TODO
    def __init__(self, nb_targets, **node_statistic_args):
        super(NodeStatisticsRegression, self).__init__(**node_statistic_args)
        self.nb_targets = nb_targets
        self.sum1 = np.zeros(
            nb_targets
        )  # sum of w_i y_i over the examples Datum(x_i, y_i, w_i), for each target
        self.sum2 = np.zeros(
            nb_targets
        )  # sum of w_i y_i^2 over the same examples, for each target

    def __str__(self):
        prediction_str = "" if self.prediction is None else "return {}".format(
            list(self.prediction))
        return "{} ({} examples)".format(prediction_str,
                                         self.total_nb_examples)

    def reset(self):
        super().reset()
        self.sum1 = np.zeros(self.nb_targets)
        self.sum2 = np.zeros(self.nb_targets)

    @staticmethod
    def compute_stats(data: List[Datum]):
        nb_targets = len(data[0].target_part)
        s = NodeStatisticsMultitargetRegression(nb_targets)
        s.add_examples(data)
        return s

    @staticmethod
    def construct_from_parent(
            parent_stats: 'NodeStatisticsMultitargetRegression'):
        return NodeStatisticsMultitargetRegression(parent_stats.nb_targets)

    def add_other_for_ensemble_prediction(
            self, other: 'NodeStatisticsMultitargetRegression', other_weight):
        self.sum1 += other.sum1
        self.sum2 += other.sum2
        self.total_nb_examples += other_weight


# noinspection PyAbstractClass
class NodeStatisticsClassificationBoosting(NodeStatisticsRegression):
    def __init__(self, **node_statistic_args):
        super().__init__(**node_statistic_args)
        self.sum_abs1 = 0.0

    def reset(self):
        super().reset()
        self.sum_abs1 = 0.0

    def get_sum_of_absolute_values(self):
        return self.sum_abs1

    def update_with_delta_weight(self, target_value, weight):
        super().update_with_delta_weight(target_value, weight)
        self.sum_abs1 += weight * abs(target_value)

    def add_other_for_ensemble_prediction(
            self, other: 'NodeStatisticsBinaryClassificationBoosting',
            other_weight):
        super().add_other_for_ensemble_prediction(other, other_weight)
        self.sum_abs1 += other_weight * other.sum_abs1


# noinspection PyAbstractClass
class NodeStatisticsBinaryClassificationBoosting(
        NodeStatisticsClassificationBoosting):
    @staticmethod
    def construct_from_parent(
            parent_stats: 'NodeStatisticsBinaryClassificationBoosting'):
        return NodeStatisticsBinaryClassificationBoosting()

    @staticmethod
    def compute_stats(data: List[Datum]):
        s = NodeStatisticsBinaryClassificationBoosting()
        s.add_examples(data)
        return s

    def create_predictions(self):
        numerator = 2 * self.sum_abs1 - self.sum2
        if numerator == 0:
            assert abs(self.sum1) > 0
            self.prediction = float('inf')
        elif abs(self.sum1) == float('inf'):
            self.prediction = 0.0
        else:
            self.prediction = self.sum1 / numerator


# noinspection PyAbstractClass
class NodeStatisticsMulticlassClassificationBoosting(
        NodeStatisticsClassificationBoosting):
    def __init__(self, nb_classes, **node_statistic_args):
        self.nb_classes = nb_classes
        super().__init__(**node_statistic_args)

    @staticmethod
    def construct_from_parent(
            parent_stats: 'NodeStatisticsMulticlassClassificationBoosting'):
        return NodeStatisticsMulticlassClassificationBoosting(
            parent_stats.nb_classes)

    @staticmethod
    def compute_stats(data: List[Datum]):
        nb_classes = len(get_all_target_values(data))
        s = NodeStatisticsMulticlassClassificationBoosting(nb_classes)
        s.add_examples(data)
        return s

    def create_predictions(self):
        j = self.nb_classes
        numerator = self.sum_abs1 - self.sum2
        if numerator == 0:
            assert self.sum1 > 0
            self.prediction = float('inf')
        else:
            self.prediction = (j - 1) / j * self.sum1 / numerator


# noinspection PyAbstractClass
class NodeStatisticsRegressionBoosting(NodeStatisticsRegression):
    def __init__(self, **node_statistic_args):
        super().__init__(**node_statistic_args)

    @staticmethod
    def compute_stats(data: List[Datum]):
        s = NodeStatisticsRegressionBoosting()
        s.add_examples(data)
        return s

    @staticmethod
    def construct_from_parent(
            parent_stats: 'NodeStatisticsRegressionBoosting'):
        return NodeStatisticsRegressionBoosting()


# noinspection PyAbstractClass
class NodeStatisticsMultitargetRegressionBoosting(
        NodeStatisticsMultitargetRegression):
    def __init__(self, nb_targets, **node_statistic_args):
        super().__init__(nb_targets, **node_statistic_args)

    @staticmethod
    def compute_stats(data: List[Datum]):
        nb_targets = len(data[0].target_part)
        s = NodeStatisticsMultitargetRegression(nb_targets)
        s.add_examples(data)
        return s

    @staticmethod
    def construct_from_parent(
            parent_stats: 'NodeStatisticsMultitargetRegressionBoosting'):
        return NodeStatisticsMultitargetRegressionBoosting(
            parent_stats.nb_targets)
