class Comparator:
    def __init__(self, name):
        self.name = name

    def compare(self, v1, v2):
        raise NotImplementedError("This should be implemented by subclass")


class IsSmaller(Comparator):
    def __init__(self):
        super().__init__("SMALLER")

    def __repr__(self):
        return "<"

    def compare(self, v1, v2):
        return v1 < v2


class IsBigger(Comparator):
    def __init__(self):
        super().__init__("BIGGER")

    def __repr__(self):
        return ">"

    def compare(self, v1, v2):
        return v1 > v2


class IsEqual(Comparator):
    def __init__(self):
        super().__init__("EQUAL")

    def __repr__(self):
        return "=="

    def compare(self, v1, v2):
        return v1 == v2


class Contains(Comparator):
    def __init__(self):
        super().__init__("CONTAINS")

    def __repr__(self):
        return "in"

    def compare(self, v1, v2):
        return v1 in v2


class DoesNotContain(Comparator):
    def __init__(self):
        super().__init__("CONTAINS_NOT")

    def __repr__(self):
        return "not in"

    def compare(self, v1, v2):
        return v1 not in v2


SMALLER = IsSmaller()
BIGGER = IsBigger()
EQUAL = IsEqual()
CONTAINS = Contains()
DOES_NOT_CONTAIN = DoesNotContain()

ALL_COMPARATORS = [SMALLER, BIGGER, EQUAL, CONTAINS, DOES_NOT_CONTAIN]
