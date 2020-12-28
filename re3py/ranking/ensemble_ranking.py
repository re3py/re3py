from typing import Dict, List
from ..learners.tree import DecisionTree, TreeNode
from ..utilities.my_exceptions import WrongValueException
# import re
from ..utilities.my_utils import float_as_string


class EnsembleRanking:
    genie3 = "GENIE3"
    symbolic = "SYMBOLIC"
    prob_dist = "PROBDIST"
    ranking_types = [genie3, symbolic, prob_dist]
    nb_places = 5

    def __init__(self, attributes, aggregates, ranking_type, iterations):
        self.attributes = attributes  # type: Dict[str, List[float]]
        self.attributes_summed = {}
        self.aggregates = aggregates  # type: Dict[str, List[float]]
        self.iterations = iterations
        if ranking_type not in EnsembleRanking.ranking_types:
            raise WrongValueException(
                "Your ranking type: {}. Allowed-ones: {}".format(
                    ranking_type, EnsembleRanking.ranking_types))
        self.ranking_type = ranking_type

    def __str__(self):
        names = ["Attributes", "Attributes summed", "Aggregators"]
        dictionaries = [
            self.attributes, self.attributes_summed, self.aggregates
        ]
        sorted_keys = [
            EnsembleRanking.sorted_keys_dictionary(d) for d in dictionaries
        ]
        max_len = max(
            [0] +
            [len(key) for keys_sums in sorted_keys for key in keys_sums[0]])
        lines = []
        line_pattern = "{{: <{}}}: {{}}; iterations: [{{}}]".format(max_len)
        for name, (keys,
                   importance_sums), dictionary in zip(names, sorted_keys,
                                                       dictionaries):
            lines.append(name)
            for k in keys:
                vs = dictionary[k]
                formatted_sum = float_as_string(importance_sums[k],
                                                EnsembleRanking.nb_places)
                formatted_vs = ", ".join([
                    float_as_string(importance, EnsembleRanking.nb_places)
                    for importance in vs
                ])
                lines.append(
                    line_pattern.format(k, formatted_sum, formatted_vs))
            lines.append("")
        return "\n".join(lines)

    def update_attributes(self, attributes_contributions,
                          aggregates_contributions, iteration):
        for d_one, d_all in zip(
            [attributes_contributions, aggregates_contributions],
            [self.attributes, self.aggregates]):
            for k, v in d_one.items():
                if k not in d_all:
                    d_all[k] = [0.0] * self.iterations
                d_all[k][iteration] += v
        for relation_name_arguments, v in attributes_contributions.items():
            relation_name = relation_name_arguments[:relation_name_arguments.
                                                    find("[")]
            if relation_name not in self.attributes_summed:
                self.attributes_summed[relation_name] = [0.0] * self.iterations
            self.attributes_summed[relation_name][iteration] += v

    def compute_tree_contribution(self, tree: DecisionTree):
        attributes_scores = {}
        aggregates_scores = {}
        for node in tree:
            if not node.is_leaf():
                attributes, aggregates = EnsembleRanking.get_attributes_and_aggregates(
                    node)
                for attribute_list, d in zip(
                    [attributes, aggregates],
                    [attributes_scores, aggregates_scores]):
                    for a in attribute_list:
                        if a not in d:
                            d[a] = 0.0
                if self.ranking_type == EnsembleRanking.genie3:
                    importance = EnsembleRanking.compute_genie3_importance(
                        node)
                elif self.ranking_type == EnsembleRanking.symbolic:
                    nb_examples = tree.root_node.get_stats(
                    ).get_total_number_examples()
                    importance = EnsembleRanking.compute_symbolic_importance(
                        node, nb_examples)
                elif self.ranking_type == EnsembleRanking.prob_dist:
                    importance = EnsembleRanking.compute_weighted_probability_distance_importance(
                        node)
                else:
                    raise NotImplementedError(
                        "Ranking for type {} not implemented.".format(
                            self.ranking_type))
                importance /= len(attributes)  # Democracy for now:)
                for attribute_list, d in zip(
                    [attributes, aggregates],
                    [attributes_scores, aggregates_scores]):
                    for a in attribute_list:
                        d[a] += importance
        return attributes_scores, aggregates_scores

    def normalize(self):
        for d in [self.attributes, self.attributes_summed, self.aggregates]:
            for k in d:
                d[k] = [relevance / self.iterations for relevance in d[k]]

    @staticmethod
    def get_attributes_and_aggregates(node):  # tree: DecisionTree,
        """
        Tests such as
        IF ('attr1', ['X0', 'Y1'], mode) in {'v1'}
        OR
        IF ('attr1', ['X0', 'girl'], sum), ('attr0', ['X0', 2.2, 'Y0'], sum) < 0.5
        are converted to
        ['attr1[X0, Y1]'] and ['mode']
        OR
        ['attr1[X0, girl]', 'attr0[X0, 2.2, Y0]'] and ['sum', 'sum'].
        :param node: a BinaryNode in the tree
        :return: list of atom tests and list of aggregators
        """
        # String approach:
        # test_string = node.split.__str__(tree.all_variables)
        # attribute_name_pattern = "'[{}]+'".format(Relation.allowed_chars)
        # arguments_pattern = "\[([^, \]]+(, )?)+\]"
        # aggregate_pattern = "[^)]+"
        # atom_pattern = "\(({}), ({}), ({})\)".format(attribute_name_pattern,
        #                                              arguments_pattern,
        #                                              aggregate_pattern)
        # for match in re.findall(atom_pattern, test_string):
        #     relation_name, arguments, aggregator = match[0], match[1], match[-1]
        #     etc.
        attributes = []
        aggregates = []
        split = node.split
        # var_dict = tree.all_variables
        for r, vs, a in split.test:
            arguments = []
            for i, v in enumerate(vs):
                # if not var_dict[v].is_unset():
                #     s = str(var_dict[v].get_value())
                # else:
                #     s = v
                s = v[0]  # the first letter --> type of variable
                arguments.append(s)
            arguments = "[{}]".format(",".join(arguments))
            attributes.append("{}{}".format(r.get_name(), arguments))
            aggregates.append(a.get_name())
        return attributes, aggregates

    @staticmethod
    def compute_genie3_importance(node: TreeNode):
        """
        Computes the (by branch frequencies) weighted variability reduction.
        :param node:
        :return:
        """
        var_here = node.get_stats().get_variability()
        var_children = [
            c.get_stats().get_variability() for c in node.get_children()
        ]
        return var_here - sum(p * v for p, v in zip(
            node.get_stats().get_branch_frequencies(), var_children))

    @staticmethod
    def compute_symbolic_importance(node: TreeNode, all_examples):
        """
        Computes the proportion of the examples that come into this node.
        :param node:
        :param all_examples:
        :return:
        """
        examples_here = node.get_stats().get_total_number_examples()
        return examples_here / all_examples

    @staticmethod
    def compute_weighted_probability_distance_importance(node: TreeNode):
        """
        Computes the difference between the vectors
        [p(true), p(false)] for each of the children.
        :param node:
        """
        def distance(v1, v2):
            p = 2
            if p == float("inf"):
                return max(abs(x1 - x2) for x1, x2 in zip(v1, v2))
            else:
                return sum(abs(x1 - x2)**p for x1, x2 in zip(v1, v2))**(1 / p)

        d = distance(*node.get_stats().get_branch_frequencies())
        examples_here = node.get_stats().get_total_number_examples()
        return d * examples_here

    @staticmethod
    def sorted_keys_dictionary(dictionary):
        sums = {k: sum(v) for k, v in dictionary.items()}
        return sorted(dictionary.keys(), key=lambda k: sums[k],
                      reverse=True), sums

    def print_ranking(self, file_name):
        f = open(file_name, "w")
        print(str(self), file=f)
        f.close()
