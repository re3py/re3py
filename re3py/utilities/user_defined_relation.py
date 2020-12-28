from typing import Dict, List
from learners.core.tree_node_split import BinarySplit
from learners.core.variables import ConstantVariable, VariableVariable
from data.relation import Relation, parse_relation_arguments
from learners.core.aggregators import Aggregator, CRITICAL_VALUES, ALL_AGGREGATORS
from learners.core.comparators import ALL_COMPARATORS
from utilities.my_exceptions import WrongValueException
from data.data_and_statistics import compute_all_values_of_types
import itertools


class BinarySplitForRelation(BinarySplit):
    def __init__(self, relation_updaters, *super_args):
        super().__init__(*super_args)
        self.relation_updaters = relation_updaters

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
        updater = self.relation_updaters[depth - 1]
        if updater is not None:
            updater.is_related(
                tuple([rel_variables[i_known].value for i_known in known]))
        related = relation.get_all(rel_variables, known)
        if len(chain_relations) == depth:
            if nb_fresh_vars is None:
                fresh_indices = [
                    i for i in unknown if rel_variables[i].can_vary()
                ]
                nb_fresh_vars = len(
                    set(rel_variables[i].get_name() for i in fresh_indices))
            if nb_fresh_vars == 0:
                to_aggregate = [len(related)]
            elif nb_fresh_vars == 1:
                to_aggregate = [r[fresh_indices[0]]
                                for r in related]  # flattened
            else:
                to_aggregate = [[r[i] for i in fresh_indices] for r in related]
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


class RelationConstructor:
    def __init__(self, relation, variables,
                 binary_splits_for_relation: List['BinarySplitForRelation']):
        self.relation = relation
        self.input_variables = variables
        self.binary_splits_for_relation = binary_splits_for_relation
        self.memo = set()
        self.all_variables = {v.name: v for v in self.input_variables}
        for v in self.input_variables:
            v.unset_value()

    def update_all_variables(self):
        for binary_split in self.binary_splits_for_relation:
            for atom_test in binary_split.test:
                relation_types = atom_test[0].types
                var_names = atom_test[1]
                for n, t in zip(var_names, relation_types):
                    if n not in self.all_variables:
                        self.all_variables[
                            n] = RelationConstructor.create_variable(n, t)

    def is_related(self, input_tuple):
        if input_tuple not in self.memo:
            example = {n: v.make_copy() for n, v in self.all_variables.items()}
            for variable, value in zip(self.input_variables, input_tuple):
                example[variable.name].value = value
            y = False
            for binary_test in self.binary_splits_for_relation:
                y = binary_test.evaluate(example)
                if y:
                    break
            self.memo.add(input_tuple)
            if y:
                self.relation.add_parsed_tuple(input_tuple)

    def populate_relation(self, values_per_type):
        options = [values_per_type[t] for t in self.relation.types]
        for t in itertools.product(*options):
            self.is_related(t)

    @staticmethod
    def create_variable(name, t):
        if name[0] == "C":
            v = ConstantVariable(name, t, None)
        elif name[0] in "XY":
            v = VariableVariable(name, t, None)
        else:
            raise WrongValueException("Wrong variable name {}".format(name))
        return v


def compute_new_relations(known_relations: Dict[str, Relation],
                          new_relations_descriptions: List[Dict]):
    """
    :param known_relations: As in Dataset.descriptive_relations
    :param new_relations_descriptions: Each element of this list is a 3-tuple that describes one of the
     newly defined relations. Examples.

     First, we define the relation sibling via already existing relations sister(Person, Person)
     and brother(Person, Person) as follows: sibling(X0, X1) <==> sister(X0, X1) OR brother(X0, X1).
     The corresponding 3-tuple is ('sibling(Person, Person)', ['X0', 'X1'], [description1, description2], True),
     where
     - 1st component gives new relation name and variable types
     - 2nd component is a list of input variable names
     - 3rd component gives the descriptions of the ways in which sibling(X, Y) is evaluated to True. In the concrete
       example, the descriptions are
       description1 = ([('sister', ['X0', 'X1'], 'SUM')], 'BIGGER', 0),
       description2 = ([('brother', ['X0', 'X1'], 'SUM')], 'BIGGER', 0),
       i.e., one of relations 'sister' and 'brother' should contain a tuple (X0, X1).

       In general, the first component of each description mimics the tests used in the BinarySplits of the trees.
     - 4th component tells whether we should ignore critical values when computing the aggregations from the conditions,
       given in the 3rd component.

    Next, the relation grandParent is defined via existing relation parent(Person, Person) as follows:
    grandParent(X0, X1) <==> exists Y: parent(X0, Y) AND parent(Y, X1), and the corresponding 3-tuple is
    ('grandParent(Person, Person)', ['X0', 'X1'], [description1], True),
    where the 3rd component has now only one element, since there is only one way in which grandParent is evaluated
    to True. This component is now
    description1 = ([('parent', ['X0', 'Y'], 'SUM'), ('parent', ['Y', 'X1'], 'SUM')], 'BIGGER', 0),
    i.e., its first component has now two elements.

    Next, we give the 3-tuple for grandParent given in terms of mother, father, parent relations:
    grandParent(X0, X1) <==>
       (exists Y1: parent(X0, Y1) AND mother(Y1, X1)) OR
       (exists Y1: parent(X0, Y1) AND father(Y1, X1))

    Now, the 3-tuple is ('grandParent(Person, Person)', ['X0', 'X1'], [d1, d2], True),
    where
    d1 = ([('parent', ['X0', 'Y1'], 'SUM'), ('mother', ['Y1', 'X1'], 'SUM')], 'BIGGER', 0),
    d2 = ([('parent', ['X0', 'Y1'], 'SUM'), ('father', ['Y1', 'X1'], 'SUM')], 'BIGGER', 0).

    Note that the formalism does not allow for a (in this case) simpler definition
    grandParent(X0, X1) <==> exists Y1: parent(X0, Y1 AND (mother(Y1, X1) OR father(Y1, X1))

    Next, we give an example of recursively defined relation of ancestor via the relation of parent:
    ancestor(X0, X1) <==> parent(X0, X1) OR (exists Y: parent(Y, X1) AND ancestor(X0, Y)).

    The 3-tuple is ('ancestor(Person, Person)', ['X0', 'X1'], [d1, d2], True),
    where
    d1 = ([('parent', ['X0', 'X1'], 'SUM')], 'BIGGER', 0)
    d2 = ([('parent', ['Y', 'X1'], 'SUM'), ('ancestor', ['X0', 'Y'], 'SUM')], 'BIGGER', 0).

    Finally, we construct a relation hasLessThanThreeKids via relation isChild(Person, Person) as follows:
     hasLessThanThreeKids(X0) <==> |{Y | isChild(Y, X0)}| < 3.

    The 3-tuple is ('hasLessThanThreeKids(Person)', ['X0'], [d1], True),
    where
    d1 = ([('isChild', ['Y', 'X0'], 'COUNT')], 'BIGGER', 3).

    :return: The new relations.
    """
    allowed_comparators = {c.name: c for c in ALL_COMPARATORS}
    allowed_aggregators = {a.name: a for a in ALL_AGGREGATORS}
    fresh = {}
    fresh_helpers = {}
    # define
    for t in new_relations_descriptions:
        name_and_types, variable_names, options, ignore_critical = t
        relation_name = name_and_types[:name_and_types.find("(")]
        relation_types = parse_relation_arguments(name_and_types,
                                                  relation_name)
        Relation.check_has_ok_types(relation_types)
        fresh[relation_name] = Relation(relation_name, set(), None,
                                        relation_types)
        variables = [
            RelationConstructor.create_variable(n, t)
            for n, t in zip(variable_names, relation_types)
        ]
        fresh_helpers[relation_name] = RelationConstructor(
            fresh[relation_name], variables, [])
        # some sanity checks
        for v in variable_names:
            if v[0] != "X":
                raise WrongValueException("Wrong variable name: {}".format(v))
        for description in options:
            atom_tests, comparator_name, threshold = description
            if comparator_name not in allowed_comparators:
                raise WrongValueException("Wrong comparator name: {}".format(
                    description[1]))
            for atom_test in atom_tests:
                if atom_test[2] not in allowed_aggregators:
                    raise WrongValueException(
                        "Wrong aggregator name: {}".format(atom_test[2]))
    all_relations = {}
    for r_name, r in fresh.items():
        all_relations[r_name] = (True, r)
    for r_name, r in known_relations.items():
        all_relations[r_name] = (False, r)
    # properly initialize
    for t in new_relations_descriptions:
        name_and_types, variable_names, options, ignore_critical = t
        relation_name = name_and_types[:name_and_types.find("(")]
        relation_constructor = fresh_helpers[relation_name]
        for description in options:
            atom_tests, comparator_name, threshold = description
            # some conversion
            relation_updaters = []
            bs_atom_tests = []
            bs_comparator = allowed_comparators[comparator_name]
            bs_threshold = threshold
            bs_ignore = ignore_critical
            for atom_test in atom_tests:
                r_name, r_variable_names, aggregator_name = atom_test
                is_fresh, r = all_relations[r_name]
                bs_atom_tests.append((r, r_variable_names,
                                      allowed_aggregators[aggregator_name]))
                updater = fresh_helpers[r_name] if r_name in fresh else None
                relation_updaters.append(updater)
            # TODO: compute return Type etc.
            bs = BinarySplitForRelation(relation_updaters, bs_atom_tests,
                                        bs_comparator, bs_threshold, bs_ignore,
                                        True)
            relation_constructor.binary_splits_for_relation.append(bs)
        relation_constructor.update_all_variables()
    # fill with values
    values_per_type = compute_all_values_of_types(
        list(known_relations.values()))
    for updater in fresh_helpers.values():
        updater.populate_relation(values_per_type)
    return fresh
