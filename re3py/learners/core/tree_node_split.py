from typing import List, Tuple, Union, Dict, Set
from ...data.relation import Relation
from .comparators import Comparator
from .variables import Variable
from .aggregators import Aggregator, CRITICAL_VALUES

# from my_exceptions import WrongValueException
# from my_memo import used_comp_memo

# tree node split memo
TEST_VALUE_MEMO = {}


class BinarySplit:
    use_memo = True
    worst_split_score = float('inf')

    def __init__(self, atom_tests: List[Tuple[Relation, List[str],
                                              Aggregator]],
                 comparator: Union[Comparator,
                                   None], threshold: Union[float, str,
                                                           Set[str]],
                 ignore_critical_values, is_variable_free):
        self.test = atom_tests
        self.comparator = comparator
        self.threshold = threshold
        self.ignore_critical_values = ignore_critical_values
        self.fresh_variables = {}  # type: Dict[str, Variable]
        self.is_variable_free = is_variable_free
        self.used_for_relation_computation = False

    def __str__(self, var_dict=None):
        tests_str = []
        for r, vs, a in self.test:
            test_str = (r.get_name(), [None] * len(vs), a)
            for i, v in enumerate(vs):
                if var_dict is not None and not var_dict[v].is_unset():
                    s = var_dict[v].get_value()
                else:
                    s = v
                test_str[1][i] = s
            tests_str.append(str(test_str))
        if type(self.threshold) == set:
            th_for_str = "{{{}}}".format(str(sorted(self.threshold))[1:-1])
        else:
            th_for_str = self.threshold
        return "{} {} {}".format(", ".join(tests_str), self.comparator,
                                 th_for_str)

    def get_test(self):
        return self.test

    def get_fresh_variables(self):
        return self.fresh_variables

    def add_fresh_variables(self, vs):
        for n, v in vs.items():
            self.fresh_variables[n] = v

    def evaluate_all(self, example: Dict[str, Variable],
                     chain_relations: List[Tuple[Relation, List[str]]],
                     chains_aggregators: List[Tuple[Aggregator]],
                     comparators: List[Comparator],
                     thresholds: List[Union[str, float]]):
        values = self.get_test_values(example, chain_relations,
                                      chains_aggregators, None, [], None, -1,
                                      None, None)
        return [
            c.compare(v, t) for c, v, t in zip(comparators, values, thresholds)
        ]

    def evaluate(self, example: Dict[str, Variable]):
        # example contains all name: variable pairs, maybe values of some is unknown
        chain_relations = [(r, vs) for r, vs, _ in self.test]
        chains_agg = [tuple([a for _, _, a in self.test])]
        if self.is_variable_free:
            compare_with = self.threshold
        else:
            compare_with = {example[v].value for v in self.threshold}
        outcome = self.evaluate_all(example, chain_relations, chains_agg,
                                    [self.comparator], [compare_with])
        return outcome[0]

    def get_test_values(self, example: Dict[str, Variable],
                        chain_relations: List[Tuple[Relation, List[str]]],
                        chains_aggregators: List[Tuple[Aggregator]],
                        relation_key, aggregator_keys, tuple_id, nb_fresh_vars,
                        fresh_indices, known_unknown):
        n_as = len(chains_aggregators)
        values = [None] * n_as
        should_memo = BinarySplit.use_memo and relation_key is not None
        d1 = None
        if should_memo:
            filtered_chains_aggregators = []
            filtered_aggregator_keys = []
            d0 = TEST_VALUE_MEMO[tuple_id]
            for i, chain_aggregators in enumerate(chains_aggregators):
                if relation_key in d0:
                    d1 = d0[relation_key]
                else:
                    d1 = {}
                    d0[relation_key] = d1
                a_key = aggregator_keys[i]
                if a_key in d1:
                    values[i] = d1[a_key]
                    # used_comp_memo[0] += 1
                else:
                    filtered_chains_aggregators.append(chain_aggregators)
                    filtered_aggregator_keys.append(a_key)
                    # used_comp_memo[1] += 1
        else:
            filtered_chains_aggregators = chains_aggregators
            filtered_aggregator_keys = [None] * len(chains_aggregators)
        if filtered_chains_aggregators:
            values_partial_all = self.get_test_value_helper(
                example, chain_relations, filtered_chains_aggregators,
                nb_fresh_vars, fresh_indices, known_unknown)
        else:
            values_partial_all = []
        where_to = 0
        for agg_key, chain_aggregators, values_partial in zip(
                filtered_aggregator_keys, filtered_chains_aggregators,
                values_partial_all):
            v = chain_aggregators[0].aggregate_flat(values_partial)
            # assert v is not None
            while values[where_to] is not None:
                where_to += 1
            values[where_to] = v
            if should_memo:
                d1[agg_key] = v
        return values

    def get_test_value_helper(self,
                              example,
                              chain_relations,
                              chains_aggregators,
                              nb_fresh_vars,
                              fresh_indices,
                              known_unknown_list,
                              depth=1):
        def results_generator(related_objects):
            for r in related_objects:
                # set
                for i in unknown:
                    rel_variables[i].set_value(r[i])
                # compute
                yield self.get_test_value_helper(example, chain_relations,
                                                 chains_aggregators,
                                                 nb_fresh_vars, fresh_indices,
                                                 known_unknown_list, depth + 1)
                # unset
                for i in unknown:
                    rel_variables[i].unset_value()

        def should_keep_value(value):
            return not (self.ignore_critical_values
                        and value in CRITICAL_VALUES)

        # for chain_aggregators in chains_aggregators:
        #     assert len(chain_relations) == len(chain_aggregators)
        relation, rel_variables_names = chain_relations[depth - 1]
        rel_variables = [example[n] for n in rel_variables_names]
        # aggregators = [chain[0] for chain in chains_aggregators]
        if known_unknown_list is None:
            known_unknown = [[], []]  # type: List[List[int]]
            for i, v in enumerate(rel_variables_names):
                known_unknown[example[v].is_unset()].append(i)
        else:
            known_unknown = known_unknown_list[depth - 1]
        known, unknown = known_unknown
        related = relation.get_all(rel_variables, known)
        if len(chain_relations) == depth:
            if nb_fresh_vars < 0:
                fresh_indices = [
                    i for i in unknown if rel_variables[i].can_vary()
                ]
                nb_fresh_vars = len(
                    set(rel_variables[i].get_name() for i in fresh_indices))
            if nb_fresh_vars == 0:
                # everything known, more or less, this is only the proof of (non)existence
                # There should be either 0 or 1 results, we pass their number.
                # This can happen in the cases such as isFriend(x, Y), isFriend(Y, z),
                # when Y has already a value after searching for the results isFriend(x, Y).
                # hits = len(related)
                # assert hits <= 1
                to_aggregate = [len(related)]
                # to_aggregate = related TODO: This more sense?
            elif nb_fresh_vars == 1:
                to_aggregate = [r[fresh_indices[0]]
                                for r in related]  # flattened
            else:
                to_aggregate = related  # [[r[i] for i in fresh_indices] for r in related]
                # if not all([aggregator in COMPLEX_ENOUGH_AGGREGATORS for aggregator in aggregators]):
                #     raise WrongValueException("Your aggregators: {}. Allowed: {}".format(aggregators,
                #                                                                          COMPLEX_ENOUGH_AGGREGATORS))
            return [to_aggregate for _ in range(len(chains_aggregators))]
        else:
            next_aggregators = [chain[depth] for chain in chains_aggregators
                                ]  # type: List[Aggregator]
            to_aggregate = []
            for res in results_generator(related):
                to_aggregate.append(res)
            answer = []
            for a_ind, a in enumerate(next_aggregators):
                ls = [neigh[a_ind] for neigh in to_aggregate]
                out = [x for x in a.aggregate(ls) if should_keep_value(x)]
                answer.append(out)
            return answer

    @staticmethod
    def is_better_than_previous(new_score, previous_score):
        return new_score < previous_score
