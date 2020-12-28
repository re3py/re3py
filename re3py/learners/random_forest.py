from .tree import DecisionTree
from ..data.data_and_statistics import *
from .predictive_model import TreeEnsemble
from ..ranking.ensemble_ranking import EnsembleRanking
from typing import List


class RandomForest(TreeEnsemble):
    zero_one_aggregator = "ZERO-ONE"
    proportions_aggregator = "PROPORTIONS"
    votes_aggregators = [zero_one_aggregator, proportions_aggregator]

    def __init__(self,
                 nb_trees_to_build=100,
                 votes_aggregator=proportions_aggregator,
                 random_seed=314159,
                 **tree_parameters):
        self.trees = []  # type: List[DecisionTree]
        self.nb_trees = nb_trees_to_build
        self.votes_aggregator = votes_aggregator
        self.ensemble_random = EnsembleRandomGenerator(random_seed)
        self.tree_parameters = tree_parameters
        self.sanity_check()

    def sanity_check(self):
        if self.votes_aggregator not in RandomForest.votes_aggregators:
            message = "Wrong votes aggregator: {}. Allowed values: {}."
            raise WrongValueException(
                message.format(self.votes_aggregator,
                               RandomForest.votes_aggregators))

    def build(self, data: Dataset):
        for t in range(self.nb_trees):
            print("Building tree {}".format(t + 1))
            self.tree_parameters[
                'random_seed'] = self.ensemble_random.next_tree_seed()
            self.trees.append(DecisionTree(**self.tree_parameters))
            tree_data = data.bootstrap_replicate(
                self.ensemble_random.next_bootstrap_seed(),
                per_class=self.trees[-1].per_class_bootstrap)
            self.trees[-1].build(tree_data)

    def compute_ranking(self, ranking_type):
        feature_ranking = EnsembleRanking({}, {}, ranking_type, self.nb_trees)
        for i, tree in enumerate(self.trees):
            attribute_scores, aggregate_scores = feature_ranking.compute_tree_contribution(
                tree)
            feature_ranking.update_attributes(attribute_scores,
                                              aggregate_scores, i)
        feature_ranking.normalize()
        return feature_ranking

    def predict(self, d: Datum, nb_trees=None):
        if nb_trees is None:
            nb_trees = len(self.trees)
        predictions_stats = []  # type: List[NodeStatistics]
        for tree in self.trees[:nb_trees]:
            predictions_stats.append(tree.predict(d, True))
        statistics_class = predictions_stats[0].__class__
        ensemble_stats = statistics_class.construct_from_parent(
            predictions_stats[0])
        for s in predictions_stats:
            ensemble_stats.add_other_for_ensemble_prediction(s, 1)
        if statistics_class == NodeStatisticsClassification:
            if self.votes_aggregator == RandomForest.zero_one_aggregator:
                values = ensemble_stats.get_nb_examples_per_class()
            elif self.votes_aggregator == RandomForest.proportions_aggregator:
                values = ensemble_stats.get_per_class_probabilities()
            else:
                raise WrongValueException("Wrong vote aggregator: {}".format(
                    self.votes_aggregator))
            class_names = ensemble_stats.get_class_names()
            return class_names[arg_max(values)]
        elif statistics_class in [
                NodeStatisticsRegression, NodeStatisticsMultitargetRegression
        ]:
            ensemble_stats.create_predictions()
            return ensemble_stats.get_prediction()
        else:
            raise NotImplementedError(":DD")

    def print_model(self, file_name):
        f = open(file_name, "w")
        for i, tree in enumerate(self.trees):
            print("Tree {}:".format(i + 1), file=f)
            print(str(tree), file=f)
            print("", file=f)
        f.close()
