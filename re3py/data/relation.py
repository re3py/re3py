from typing import Set, Tuple, List, Union
import re
from ..utilities.my_utils import *
from ..learners.core.variables import Variable
from ..utilities.my_exceptions import WrongValueException
import numpy as np


class Relation:
    constant_nominal = "nominal"
    constant_numeric = "numeric"
    constant_multi_target = "multi_target["
    relation_type_constant = [
        constant_nominal, constant_numeric, constant_multi_target
    ]
    allowed_chars = "-.,+ A-Za-z0-9_\\[\\]"
    # constant_type = "constant"
    tuple_pattern = "{{}}\\(([{} ,]+)\\)".format(allowed_chars)
    time_efficient_search_bound = 3

    def __init__(self, name: str, related_objects: Union[Set[Tuple[str]],
                                                         None],
                 file: Union[str, None], types):
        self.name = name
        self.all_tuples = set()
        self.types = types
        self.different_values = [-1] * len(self.types)
        self.arity = len(self.types)
        self.all_tuples_by_subsets = {}
        self.init_all_tuples_by_subsets()
        self.file = file
        p1 = related_objects is None
        p2 = self.file is None
        assert p1 + p2 == 1
        if p1:
            with open(file) as f:
                for whole_line in f:
                    line = whole_line[:whole_line.find("//")]  # comments
                    if line:
                        self.try_add_tuple(line)
        else:
            self.all_tuples = related_objects
            # TODO: This does not update tuples by subsets?

    def __repr__(self):
        set_part = []
        total_len = 0
        for t in self.all_tuples:
            if total_len > 30:
                set_part.append("...")
                break
            set_part.append(str(t))
            total_len += len(set_part[-1])
        return "Relation({}, {{{}}})".format(self.name, ", ".join(set_part))

    def __eq__(self, other):
        return self.name == other.name  # and self.types == other.types and self.all_tuples == other.all_tuples

    def should_use_tuples_by_subsets(self):
        return 1 <= self.arity <= Relation.time_efficient_search_bound

    def init_all_tuples_by_subsets(self):
        if self.should_use_tuples_by_subsets():
            pattern = "{{:0>{}b}}".format(self.arity)
            subset_codes = [
                pattern.format(i) for i in range(1, 2**self.arity - 1)
            ]
            for subset_code in subset_codes:
                self.all_tuples_by_subsets[subset_code] = {}
            for t in self.all_tuples:
                self.try_add_one_to_tuples_by_subsets(t)

    def try_add_tuple(self, line: str):
        """
        Converts 'r(x,y)' to (x, y) and adds it to all tuples.

        :param line: a string of form <relation name>(obj1, obj2, ...)
        """
        related_list = parse_relation_arguments(line, self.name)
        relation_tuple = tuple(
            Relation.intelligent_parse(v_type, v_value)
            for v_type, v_value in zip(self.types, related_list))
        self.add_parsed_tuple(relation_tuple)

    def add_parsed_tuple(self, t):
        self.all_tuples.add(t)
        if self.should_use_tuples_by_subsets():
            self.try_add_one_to_tuples_by_subsets(t)

    def try_add_one_to_tuples_by_subsets_old(self, relation_tuple):
        pattern = "{{:0>{}b}}".format(self.arity)
        subset_codes = [pattern.format(i) for i in range(1, 2**self.arity - 1)]
        for subset_code in subset_codes:
            subset_dict = self.all_tuples_by_subsets[subset_code]
            key_part = []
            value_part = []
            for tuple_component, code_component in zip(relation_tuple,
                                                       subset_code):
                if code_component == "1":
                    key_part.append(tuple_component)
                else:
                    value_part.append(tuple_component)
            key_part = tuple(key_part)
            value_part = tuple(value_part)
            if key_part not in subset_dict:
                subset_dict[key_part] = set()
            subset_dict[key_part].add(value_part)

    def try_add_one_to_tuples_by_subsets(self, relation_tuple):
        pattern = "{{:0>{}b}}".format(self.arity)
        subset_codes = [pattern.format(i) for i in range(1, 2**self.arity - 1)]
        for subset_code in subset_codes:
            subset_dict = self.all_tuples_by_subsets[subset_code]
            key_part = []
            # value_part = []
            for tuple_component, code_component in zip(relation_tuple,
                                                       subset_code):
                if code_component == "1":
                    key_part.append(tuple_component)
                # else:
                #     value_part.append(tuple_component)
            key_part = tuple(key_part)
            value_part = relation_tuple  # tuple(value_part)
            if key_part not in subset_dict:
                subset_dict[key_part] = []  # set()
            # subset_dict[key_part].add(value_part)
            subset_dict[key_part].append(value_part)

    def get_all_old(self, variables: List[Variable],
                    known_values: List[int]) -> List[Tuple[Variable]]:
        """
        :param variables: e.g., [X0(2.1), X1(?), X2('b')]
        :param known_values: list of indices of the variables that have known value, e.g., [0, 2]
        :return: list of tuples that satisfy the constraints, e.g., all triplets (x, y, z), for which
          x == 2.1 and z = 'b'.
        """
        def check_ok(related):
            for j in known_values:
                if variables[j].get_value() != related[j]:
                    return False
            return True

        def merge(known_part, unknown_part):
            merged = [None] * len(subset_code)
            i_known = 0
            i_unknown = 0
            for j in range(len(merged)):
                if subset_code[j] == "1":
                    merged[j] = known_part[i_known]
                    i_known += 1
                else:
                    merged[j] = unknown_part[i_unknown]
                    i_unknown += 1
            return tuple(merged)

        if not self.should_use_tuples_by_subsets():
            return [t for t in self.all_tuples if check_ok(t)]
        else:
            key_part = tuple([variables[i].get_value() for i in known_values])
            if len(known_values) == self.arity:
                return [key_part] if key_part in self.all_tuples else []
            elif len(known_values) == 0:
                return list(self.all_tuples)
            else:
                subset_code = ["0"] * self.arity
                for i in known_values:
                    subset_code[i] = "1"
                subset_code = "".join(subset_code)
                subset_dict = self.all_tuples_by_subsets[subset_code]
                if key_part in subset_dict:
                    value_parts = subset_dict[key_part]
                else:
                    value_parts = set()
                return [
                    merge(key_part, value_part) for value_part in value_parts
                ]

    def get_all(self, variables: List[Variable],
                known_values: List[int]) -> List[Tuple[Variable]]:
        """
        :param variables: e.g., [X0(2.1), X1(?), X2('b')]
        :param known_values: list of indices of the variables that have known value, e.g., [0, 2]
        :return: list of tuples that satisfy the constraints, e.g., all triplets (x, y, z), for which
          x == 2.1 and z = 'b'.
        """
        def check_ok(related):
            for j in known_values:
                if variables[j].get_value() != related[j]:
                    return False
            return True

        if not self.should_use_tuples_by_subsets():
            return [t for t in self.all_tuples if check_ok(t)]
        else:
            key_part = tuple([variables[i].get_value() for i in known_values])
            if len(known_values) == self.arity:
                return [key_part] if key_part in self.all_tuples else []
            elif len(known_values) == 0:
                return list(self.all_tuples)
            else:
                subset_code = ["0"] * self.arity
                for i in known_values:
                    subset_code[i] = "1"
                subset_code = "".join(subset_code)
                subset_dict = self.all_tuples_by_subsets[subset_code]
                if key_part in subset_dict:
                    return subset_dict[key_part]
                else:
                    return []

    def get_all_values(self, position):
        return sorted({t[position] for t in self.all_tuples})

    def get_nb_all_values(self, position):
        if self.different_values[position] < 0:
            self.different_values[position] = len(
                self.get_all_values(position))
        return self.different_values[position]

    def get_name(self):
        return self.name

    def get_types(self):
        return self.types

    @staticmethod
    def intelligent_parse(variable_type, variable_value):
        if Relation.is_numeric_type(variable_type):
            # constant numeric
            return try_convert_to_number(variable_value)
        elif Relation.is_nominal_type(variable_type):
            # constant nominal
            return variable_value
        elif Relation.is_multi_target_type(variable_type):
            # multi_target[some type]
            type_of_targets = Relation.get_inner_type_of_multi_target(
                variable_type)
            if not (variable_value.startswith('[')
                    and variable_value.endswith(']')):
                raise WrongValueException(
                    "Multi-target value {} wrongly specified!".format(
                        variable_value))
            target_values_string = intelligent_split(variable_value[1:-1])
            target_values = [
                Relation.intelligent_parse(type_of_targets, v)
                for v in target_values_string
            ]
            if Relation.is_numeric_type(type_of_targets):
                return np.array(target_values)
        else:
            # user-defined
            return variable_value

    @staticmethod
    def is_constant_type(variable_type):
        return Relation.is_nominal_type(variable_type) or Relation.is_numeric_type(variable_type) or\
               Relation.is_multi_target_type(variable_type)

    @staticmethod
    def is_nominal_type(variable_type):
        return variable_type.startswith(Relation.constant_nominal)

    @staticmethod
    def is_numeric_type(variable_type):
        return variable_type.startswith(Relation.constant_numeric)

    @staticmethod
    def is_multi_target_type(variable_type):
        return variable_type.startswith(Relation.constant_multi_target)

    @staticmethod
    def get_inner_type_of_multi_target(variable_type):
        assert Relation.is_multi_target_type(variable_type)
        i0, i1 = variable_type.find('['), variable_type.rfind(']')
        if min(i0, i1) < 0:
            raise WrongValueException(
                "Multi-target type {} wrongly specified!".format(
                    variable_type))
        return variable_type[i0 + 1:i1].strip()

    @staticmethod
    def check_has_ok_types(relation_types):
        for t in relation_types:
            if t[0] == t[0].lower() and not Relation.is_constant_type(t):
                message = "Type {} should either start with a capital letter, " \
                          "or be of form <element><appendix>, where <element> is in {}."
                raise WrongValueException(
                    message.format(t, Relation.relation_type_constant))


def parse_relation_arguments(line: str, relation_name: str):
    assert line.startswith(relation_name)
    tuple_pattern = Relation.tuple_pattern.format(relation_name)
    try:
        related_str = re.search(tuple_pattern, line).group(1).strip()
    except AttributeError:
        print("Relation {}: tuple {} does not match pattern {}".format(
            relation_name, line, tuple_pattern))
        raise
    related_list = intelligent_split(related_str)
    object_name = "^[{}]+$".format(Relation.allowed_chars)
    for o in related_list:
        if re.match(object_name, o) is None:
            if o == "":
                print("Empty string detected in {}".format(line))
            else:
                raise WrongValueException(
                    "{} in {} contains forbidden characters or is empty.".
                    format(o, related_str))
    return related_list


def intelligent_split(line):
    inside_list = False
    inside_quotation = False
    elements = []
    start = 0
    for i, c in enumerate(line):
        if c == '"':
            if i > 0 and line[i - 1] == '\\':
                pass  # escaped quotation
            else:
                inside_quotation = not inside_quotation
        if inside_quotation:
            continue
        if c == '[':
            inside_list = True
        elif c == ']':
            inside_list = False
        elif c == ',' and not inside_list:
            elements.append(line[start:i])
            start = i + 1
    elements.append(line[start:])
    return [e.strip() for e in elements]


def parse_relation_name(line):
    name_pattern = "([{}]+)\\(".format(Relation.allowed_chars)
    try:
        return re.match(name_pattern, line).group(1)
    except AttributeError:
        print(f"Pattern {name_pattern} did not match {line}")
        raise


def parse_relation(line):
    relation_name = parse_relation_name(line)
    related_list = parse_relation_arguments(line, relation_name)
    return relation_name, related_list
