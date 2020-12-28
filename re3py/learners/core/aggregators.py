from typing import List
import statistics as st
import itertools

MODE_OF_EMPTY_LIST = "Nothing to see here"
TYPE_NUMERIC = "numeric"
TYPE_NOMINAL = "nominal"
TYPE_TYPE = "type"
TYPE_TUPLE = "tuple"
TYPE_SAME_AS_INPUT = "as input"


class Aggregator:
    def __init__(self, name):
        self.name = name
        self.input_types = set()
        self.output_type = None
        self.is_projection = False

    def __repr__(self):
        return self.name

    def get_name(self):
        return self.name

    def aggregate_flat(self, a_list):
        raise NotImplementedError("Should be implemented by a subclass!")

    def aggregate(self, ls):
        return [self.aggregate_flat(a_list) for a_list in ls]

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.get_name() < other.get_name()


class Count(Aggregator):
    def __init__(self):
        super().__init__("count")
        self.input_types = {TYPE_NUMERIC, TYPE_NOMINAL, TYPE_TYPE, TYPE_TUPLE}
        self.output_type = TYPE_NUMERIC

    def aggregate_flat(self, a_list):
        return len(a_list)


class CountUnique(Aggregator):
    def __init__(self):
        super().__init__("countUnique")
        self.input_types = {TYPE_NUMERIC, TYPE_NOMINAL, TYPE_TYPE,
                            TYPE_TUPLE}  # Should numeric be removed?
        self.output_type = TYPE_NUMERIC

    def aggregate_flat(self, a_list):
        return len(set(a_list))


class Min(Aggregator):
    def __init__(self):
        super().__init__("min")
        self.input_types = {TYPE_NUMERIC}
        self.output_type = TYPE_NUMERIC

    def aggregate_flat(self, a_list):
        return min(a_list) if a_list else float("inf")


class Max(Aggregator):
    def __init__(self):
        super().__init__("max")
        self.input_types = {TYPE_NUMERIC}
        self.output_type = TYPE_NUMERIC

    def aggregate_flat(self, a_list):
        return max(a_list) if a_list else float("-inf")


class Mean(Aggregator):
    def __init__(self):
        super().__init__("mean")
        self.input_types = {TYPE_NUMERIC}
        self.output_type = TYPE_NUMERIC

    def aggregate_flat(self, a_list):
        return st.mean(a_list) if a_list else float(
            "inf")  # slightly better than 0? None?


class Sum(Aggregator):
    def __init__(self):
        super().__init__("sum")
        self.input_types = {TYPE_NUMERIC}
        self.output_type = TYPE_NUMERIC

    def aggregate_flat(self, a_list):
        return sum(a_list)


class Mode(Aggregator):
    def __init__(self):
        super().__init__("mode")
        self.input_types = {TYPE_NOMINAL, TYPE_TYPE}
        self.output_type = TYPE_SAME_AS_INPUT

    def aggregate_flat(self, a_list):
        d = {}
        for j in a_list:
            if j not in d:
                d[j] = 1
            else:
                d[j] += 1
        if d:
            return max(sorted(d),
                       key=lambda t: d[t])  # Every time the same element
        else:
            return MODE_OF_EMPTY_LIST  # float("inf")


class Flatten(Aggregator):
    def __init__(self):
        super().__init__("flatten")
        self.input_types = {TYPE_NUMERIC, TYPE_NOMINAL, TYPE_TYPE}
        self.output_type = TYPE_SAME_AS_INPUT

    def aggregate(self, ls):
        return list(itertools.chain.from_iterable(ls))

    def aggregate_flat(self, a_list):
        return a_list


class FlattenUnique(Aggregator):
    def __init__(self):
        super().__init__("flattenUnique")
        self.input_types = {TYPE_NUMERIC, TYPE_NOMINAL, TYPE_TYPE}
        self.output_type = TYPE_SAME_AS_INPUT

    def aggregate(self, ls):
        return list(set(itertools.chain.from_iterable(ls)))

    def aggregate_flat(self, a_list):
        return a_list


class Project(Aggregator):
    def __init__(self, component, aggregator):
        super().__init__("projection{}".format(component))
        self.input_types = {TYPE_TUPLE}
        self.output_type = "unknown"
        self.is_projection = True
        self.component = component
        self.aggregator = aggregator
        if self.aggregator is not None:
            self.name += "_" + self.aggregator.name
            self.output_type = self.aggregator.output_type

    def aggregate_flat(self, a_list):
        return self.aggregator.aggregate_flat(
            [e[self.component] for e in a_list])


class Project0(Project):
    def __init__(self, aggregator):
        super().__init__(0, aggregator)


class Project1(Project):
    def __init__(self, aggregator):
        super().__init__(1, aggregator)


class Project2(Project):
    def __init__(self, aggregator):
        super().__init__(2, aggregator)


class Project3(Project):
    def __init__(self, aggregator):
        super().__init__(3, aggregator)


class Project4(Project):
    def __init__(self, aggregator):
        super().__init__(4, aggregator)


class Project5(Project):
    def __init__(self, aggregator):
        super().__init__(5, aggregator)


class Project6(Project):
    def __init__(self, aggregator):
        super().__init__(6, aggregator)


class Project7(Project):
    def __init__(self, aggregator):
        super().__init__(7, aggregator)


FLATTEN = Flatten()
FLATTEN_UNIQUE = FlattenUnique()
COUNT = Count()
COUNT_UNIQUE = CountUnique()
MIN = Min()
MAX = Max()
MEAN = Mean()
SUM = Sum()
MODE = Mode()
PROJECT = Project("", None)
PROJECTIONS = [
    Project0, Project1, Project2, Project3, Project4, Project5, Project6,
    Project7
]

ALL_AGGREGATORS = [
    FLATTEN, FLATTEN_UNIQUE, COUNT, COUNT_UNIQUE, MIN, MAX, MEAN, SUM, MODE,
    PROJECT
]  # type: List['Aggregator']
NUMERIC_AGGREGATORS = [
    a for a in ALL_AGGREGATORS if TYPE_NUMERIC in a.input_types
]  # type: List['Aggregator']
NOMINAL_AGGREGATORS = [
    a for a in ALL_AGGREGATORS
    if TYPE_NOMINAL in a.input_types and TYPE_TYPE in a.input_types
]  # type: List['Aggregator']
COMPLEX_ENOUGH_AGGREGATORS = [
    a for a in ALL_AGGREGATORS if TYPE_TUPLE in a.input_types
]
CRITICAL_VALUES = [float("inf"), float("-inf"), MODE_OF_EMPTY_LIST]
