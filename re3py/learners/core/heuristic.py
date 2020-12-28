from ...data.data_and_statistics import *
from typing import List
import numpy as np

class Heuristic:
    def compute_variability(self, tree_node_stat):
        raise NotImplementedError("This should be implemented by a subclass.")

    def evaluate_split(self, parent_stats: NodeStatistics,
                       children_stats: List[NodeStatistics]):
        # TODO: make this efficient (h_parent is not necessary for arg max etc.)
        examples_p = parent_stats.get_total_number_examples()
        examples_cs = [c.get_total_number_examples() for c in children_stats]
        branch_freq = [examples_c / examples_p for examples_c in examples_cs]
        # h_p = self.compute_variability(parent_stats)
        h_cs = [self.compute_variability(c) for c in children_stats]
        # branch_freq = parent_stats.get_branch_frequencies()
        # return h_p - sum(p * h_c for p, h_c in zip(branch_freq, h_cs))
        return sum(p * h_c for p, h_c in zip(branch_freq, h_cs))


class HeuristicGini(Heuristic):
    def compute_variability(self,
                            tree_node_stat: NodeStatisticsClassification):
        return gini(tree_node_stat.get_per_class_probabilities())


def gini(probabilities):
    return 1 - sum(p**2 for p in probabilities)


class HeuristicVariance(Heuristic):
    def compute_variability(self, tree_node_stat: NodeStatisticsRegression):
        """
        Computes the estimate of the variance E[Y^2] - E[Y]^2.
        Since the examples are weighted and can have small weights, we compute the biased estimate,
        i.e., instead of ... / (n - 1), we return ... / n.
        :param tree_node_stat:
        :return:
        """
        n = tree_node_stat.get_total_number_examples(
        )  # n > 0 when this is called
        s1 = tree_node_stat.get_sum_of_values()
        s2 = tree_node_stat.get_sum_of_squared_values()
        return max(0, (s2 - s1**2 / n) / n)


class HeuristicMultitargetVariance(Heuristic):
    def compute_variability(
            self, tree_node_stat: NodeStatisticsMultitargetRegression):
        """
        Computes the sum of estimates of the variance E[Y^2] - E[Y]^2, for each target variable.
        Basically, the sum of HeuristicVariance.compute_variability ...
        :param tree_node_stat:
        :return:
        """
        n = tree_node_stat.get_total_number_examples(
        )  # n > 0 when this is called
        s1 = tree_node_stat.get_sum_of_values()[2:]
        s2 = tree_node_stat.get_sum_of_squared_values()[2:]
        v = np.mean(s2 - s1**2 / n) / n
        return v if v > 0 else 0
