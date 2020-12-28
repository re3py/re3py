from .relation import *
import re
from ..learners.core.aggregators import ALL_AGGREGATORS
from typing import List
from ..utilities.my_utils import try_convert_to_number
from ..utilities.my_exceptions import *

class Settings:
    sec_relations = "Relations"
    sec_relations_constructed = "RelationsConstructed"
    # sec_data = "Data"
    sec_aggregates = "Aggregates"
    sec_atom_tests = "AtomTests"
    sec_tree_params = "TreeParameters"

    atom_test_type_constant = "c"
    must_be_known = "old"
    must_not_be_known = "new"
    allowed_test_specifications = [
        must_be_known, must_not_be_known, atom_test_type_constant
    ]
    allowed_aggregators = [a.get_name() for a in ALL_AGGREGATORS]
    # add all possible tests for automatic generation: or even better: leave blank if you want all? :)
    allowed_sections = [
        sec_relations, sec_aggregates, sec_atom_tests, sec_tree_params,
        sec_relations_constructed
    ]
    comment_string = "//"
    default_tree_parameters = {
        "numNodes": float("inf"),
        "minInstancesNode": 1,
        "maxDepth": float("inf"),
        "maxTestLength": 1
    }

    def __init__(self, s_file):
        self.relations = []  # type: List[Relation]
        # self.data_settings = {"train": None, "test": None, "target": None}
        self.aggregates = []
        self.atom_tests = []  # [(relName, (specification1, ...)), ...]
        self.tree_parameters = {
            k: None
            for k in Settings.default_tree_parameters
        }
        self.relations_constructed = []
        # ignore empty lines and parts of the lines after comment string
        with open(s_file) as f:
            lines = []
            section_indices = []
            for raw_line in f:
                i = raw_line.find(Settings.comment_string)
                line = raw_line.strip() if i < 0 else raw_line[:i].strip()
                if line:
                    section_match = re.match("\\[([A-Za-z]+)\\]", line)
                    if section_match is not None:
                        section_indices.append(len(lines))
                        lines.append(section_match.group(1))
                    else:
                        lines.append(line)
            section_indices.append(len(lines))  # sentinel
            for i in range(len(section_indices) - 1):
                self.parse_section(
                    lines[section_indices[i]:section_indices[i + 1]])
        self.sanity_check()

    def get_relations(self):
        return self.relations

    def get_target_relation(self):
        return self.relations[0]

    def get_aggregates(self):
        return self.aggregates

    def get_atom_tests(self):
        return self.atom_tests

    def get_atom_tests_structured(self):
        allowed_specs = {
            s: i
            for i, s in enumerate(Settings.allowed_test_specifications)
        }
        structured = {}
        # Dictionary with keys such as (relName, (old, old, new, c), (Person, Person, Dog, Size))
        # and values such as {Person: [[0, 1], [], []]}, Dog: [[], [2], []], Size: [[], [], [3]]}
        rel_data_types = {r.get_name(): r.get_types() for r in self.relations}
        for r, specifications in self.atom_tests:
            key = (r, specifications, tuple(rel_data_types[r]))
            structured[key] = {t: [[], [], []] for t in rel_data_types[r]}
            for i, (specification,
                    o_type) in enumerate(zip(specifications,
                                             rel_data_types[r])):
                structured[key][o_type][allowed_specs[specification]].append(i)
        return structured

    def get_tree_parameters(self):
        return self.tree_parameters

    def parse_section(self, lines):
        sec_name = lines[0]
        data = lines[1:]
        if sec_name == Settings.sec_relations:
            for line in data:
                relation_name = line[:line.find("(")]
                relation_types = parse_relation_arguments(line, relation_name)
                Relation.check_has_ok_types(relation_types)
                new_relation = Relation(relation_name, set(), None,
                                        relation_types)
                if new_relation not in self.relations:
                    self.relations.append(new_relation)
                else:
                    print(
                        "Warning: relation {} was listed more than once. Ignoring duplicates."
                        .format(new_relation))
        elif sec_name == Settings.sec_aggregates:
            for line in data:
                if line not in Settings.allowed_aggregators:
                    raise WrongValueException(
                        "Wrong aggregator: {}. Allowed: {}".format(
                            line, Settings.allowed_aggregators))
                if line not in self.aggregates:
                    self.aggregates.append(line)
                else:
                    print(
                        "Warning: aggregate {} was listed more than once. Ignoring duplicates."
                        .format(line))
        elif sec_name == Settings.sec_atom_tests:
            for line in data:
                relation_name = line[:line.find("(")]
                test_specifications = parse_relation_arguments(
                    line, relation_name)
                for t in test_specifications:
                    if t not in Settings.allowed_test_specifications:
                        message = "Wrong test specification: {}. Allowed: {}"
                        raise WrongValueException(
                            message.format(
                                t, Settings.allowed_test_specifications))
                new_test = (relation_name, tuple(test_specifications))
                if new_test not in self.atom_tests:
                    self.atom_tests.append(new_test)
                else:
                    print(
                        "Warning: test {} was listed more than once. Ignoring duplicates."
                        .format(new_test))
        elif sec_name == Settings.sec_tree_params:
            for line in data:
                key_value = re.match("([A-Za-z]+) *= *(.+)", line)
                if key_value is None:
                    raise WrongValueException(
                        "Wrong line format for {}.\nUse parameter = value format."
                        .format(line))
                key, value = [key_value.group(i) for i in range(1, 3)]
                value = try_convert_to_number(value)
                if key not in self.tree_parameters:
                    message = "Wrong data type: {}. Allowed: {}"
                    raise WrongValueException(
                        message.format(key, list(self.tree_parameters.keys())))
                if self.tree_parameters[key] is not None:
                    print(
                        "Warning! Tha value for {} is already defined. Will be overridden."
                        .format(key))
                self.tree_parameters[key] = value
            for k, v in self.tree_parameters.items():
                if v is None:
                    self.tree_parameters[k] = Settings.default_tree_parameters[
                        k]
        elif sec_name == Settings.sec_relations_constructed:
            for line in data:
                self.relations_constructed.append(eval(line))
        else:
            message = "Unknown setting section: {}. Allowed: {}"
            raise WrongValueException(
                message.format(sec_name, Settings.allowed_sections))

    def sanity_check(self):
        # # specify everything in data section
        # for key, value in self.data_settings.items():
        #     if value is None:
        #         raise Exception("Value of {} (section {}) not set".format(key, Settings.sec_data))
        # data files must exist: hold your horses, we will exclude this section anyway

        # target relation must be among relations
        if len(self.relations) == 0:
            raise MissingValueException(
                "Target relation unknown, because no relation specified.")
        else:
            # target relation is of form rel(X1, X2, ... , XN, value)
            # hence the last type must be constant (and only the last type)
            target_types = self.relations[0].get_types()
            for i, t in enumerate(target_types):
                is_ok = (i == len(target_types) -
                         1) == Relation.is_constant_type(t)
                help_string = "" if (i == len(target_types) - 1) else "not "
                if not is_ok:
                    message = "The type on position {}/{} of target relation must {}be in {}, but is {}."
                    raise WrongValueException(
                        message.format(i + 1, len(target_types), help_string,
                                       Relation.relation_type_constant, t))
        if len(self.atom_tests) == 0:
            print(
                "No atom tests specified, all possible options will be considered."
            )
        # atom test should contain known relation name, and has the same number
        # of types at the relation itself
        for r, types in self.atom_tests:
            found_r = False
            for r2 in self.relations:
                if r == r2.get_name():
                    found_r = True
                    if len(types) != len(r2.get_types()):
                        message = "Arity of {} differs in relation ({}) and atom test specification ({})."
                        raise WrongValueException(
                            message.format(r, r2.get_types(), types))
            if not found_r:
                raise MissingValueException(
                    "Relation name in atom test: {} not among the relations".
                    format(r))
