import random


class EnsembleRandomGenerator:
    upper_bound = 12345

    def __init__(self, random_seed):
        self.meta_random = random.Random(random_seed)
        self.tree_seeds = random.Random(self.next_seed())
        self.bootstrap_seeds = random.Random(self.next_seed())
        self.sample_rows_seeds = random.Random(self.next_seed())

    def next_seed(self):
        return int(self.meta_random.random() *
                   EnsembleRandomGenerator.upper_bound)

    def next_tree_seed(self):
        return int(self.tree_seeds.random() *
                   EnsembleRandomGenerator.upper_bound)

    def next_bootstrap_seed(self):
        return int(self.bootstrap_seeds.random() *
                   EnsembleRandomGenerator.upper_bound)

    def next_sample_rows_seed(self):
        return int(self.sample_rows_seeds.random() *
                   EnsembleRandomGenerator.upper_bound)


def try_convert_to_number(s):
    # try:
    #     return int(s)
    # except ValueError:
    #     pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def union_of_two_dicts(d1, d2):
    """
    Computes the union of two dictionaries whose values are lists.
    :param d1: dictionary, e.g.,  {'Person': {'X1', 'X2'}, 'Animal': {'Y1'}}
    :param d2: same as d1, e.g., {'Person': {'X4'}, 'House': {'X3'}}
    :return: a union of dictionaries, e.g., {'Person': {'X1', 'X2', 'X4'}, 'Animal': {'Y1'}, 'House': {'X3'}}
    """
    d = {k: {n for n in v} for k, v in d1.items()}
    for k, v in d2.items():
        if k not in d:
            d[k] = v
        else:
            d[k] |= v
    return d


def average_of_dictionaries(ds):
    n = len(ds)
    together = {}
    # sum
    for d in ds:
        for k, v in d.items():
            if k not in together:
                together[k] = 0.0
            together[k] += v
    # division
    if n > 1:
        for k in together:
            together[k] /= n
    return together


def canonic_sequences_of_new_variables(n):
    """
    For example, if n == 5, we would get
    1 1 1 1 1
    1 1 1 1 2
    1 1 1 2 1
    ...
    1 2 1 2 1
    ...
    but not 2 2 2 2 2, 2 1 2 1 2 etc.
    :param n: The length of the sequence
    :return: All possible sequences of indices of new variables.
    """
    def helper(k):
        if k == 1:
            yield [1], 1
        else:
            for s, m in helper(k - 1):
                for i in range(1, m + 2):
                    yield s + [i], max(i, m)

    if n == 0:
        yield []
    else:
        for t, _ in helper(n):
            yield t


def subsets_of_list(a_list, max_nb_generated=None):
    n = len(a_list)
    nb_subsets = 2**n
    pattern = "{{:0>{}b}}".format(n)
    if max_nb_generated is None:
        max_nb_generated = nb_subsets
    bound = min(nb_subsets, max_nb_generated)
    for i in range(bound):
        s = pattern.format(i)
        chosen_and_not = [[], []]
        for c, e in zip(s, a_list):
            chosen_and_not[c == "0"].append(e)
        yield chosen_and_not


def generalized_counting(a_list, k):
    n = len(a_list)
    indices = [0] * k
    if n > 0:
        while True:
            yield [a_list[i] for i in indices]
            which = k - 1
            while which >= 0 and indices[which] == n - 1:
                which -= 1
            if which < 0:
                break
            for after in range(which + 1, k):
                indices[after] = 0
            indices[which] += 1
    elif k == 0:
        yield []


def get_an_element_of_set(s):
    return next(iter(s))


def arg_max(values):
    m = -float("inf")
    chosen = None
    for i, v in enumerate(values):
        if v > m:
            m = v
            chosen = i
    return chosen


def float_as_string(x, significant_places):
    float_to_str_pattern = "{{:.{}e}}".format(significant_places)
    x_str = float_to_str_pattern.format(float(x))
    # Different lengths can occur:
    # 1) '-' sign at the beginning
    # 2) Different lengths of exponent: we assume that it is either e+12 or e+100
    has_minus = x_str.startswith('-')
    exponent_len = len(x_str) - x_str.find("e") - 2  # either 2 or 3
    start = "" if has_minus else " "
    end = "" if exponent_len == 3 else " "
    return start + x_str + end
