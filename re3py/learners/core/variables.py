class Variable:
    def __init__(self, name: str, value_type, value):
        self.name = name
        self.value_type = value_type
        self.value = value

    def __repr__(self):
        v = self.value if not self.is_unset() else "?"
        return "{}({})".format(self.name, v)

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)

    def rename(self, n):
        self.name = n

    def get_name(self):
        return self.name

    def set_value(self, v):
        self.value = v

    def unset_value(self):
        self.value = None

    def is_unset(self):
        return self.value is None

    def get_value(self):
        return self.value

    def is_constant(self):
        return False

    def can_vary(self):
        return False

    def make_copy(self):
        raise NotImplementedError("This should be implemented by a subclass.")


class ConstantVariable(Variable):
    def __init__(self, name, variable_type, value):
        super().__init__(name, variable_type, value)
        assert self.name.startswith("C")

    def is_constant(self):
        return True

    def make_copy(self):
        return ConstantVariable(self.name, self.value_type, self.value)


class VariableVariable(Variable):
    def __init__(self, name, variable_type, value):
        super().__init__(name, variable_type, value)
        assert self.name[0] in "XY"  # X for the initial ones, Y for the others

    def can_vary(self):
        return True

    def make_copy(self):
        return VariableVariable(self.name, self.value_type, self.value)
