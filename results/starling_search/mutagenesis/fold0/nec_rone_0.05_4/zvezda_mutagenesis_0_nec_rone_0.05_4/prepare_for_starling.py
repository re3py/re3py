import numpy as np
import re


def remove_operators(line_out):
    def replace(match):
        return f"dot{match.group(1)}"

    def replace2(match):
        return f"{match.group(1)}underscore"

    line_out = re.sub("\\+", "plus", line_out)
    line_out = re.sub("-", "minus", line_out)
    line_out = re.sub("#", "hash", line_out)
    line_out = re.sub("([(, ]|^)_", replace2, line_out)
    line_out = re.sub("\\.(.)", replace, line_out)
    return line_out.lower()


def split_relation(line):
    i0 = line.find('(')
    j0 = line.find(')')
    assert min(i0, j0) > 0, (i0, i0, line)
    relation = line[:i0]
    arguments = line[i0 + 1: j0].split(',')
    arguments = [a.strip() for a in arguments]
    return relation, arguments


def simple_s_parser(s_file):
    s_object = {}
    section = "[]"
    with open(s_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("//") or not line:
                continue
            elif line.startswith("["):
                section = line
                if section == "[Relations]":
                    s_object[section] = {}
                elif section == "[AtomTests]":
                    s_object[section] = []
            elif section in ["[Relations]", "[AtomTests]"]:
                relation, arguments = split_relation(line)
                if section == "[Relations]":
                    if not s_object[section]:
                        s_object["target"] = relation
                    s_object[section][relation] = arguments
                else:
                    s_object[section].append((relation, arguments))
    return s_object


def prepare_background_file(s_file):
    def create_mode(r, r_arguments=None):
        # the system does not allow for more than one new variable ...
        if r_arguments is None:
            assert s_object["target"] == r
            r_arguments = ["old"] * (len(s_object["[Relations]"][r]) - 1)
        argument_types = s_object["[Relations]"][r]
        assert len(arguments) == len(argument_types)
        new_arguments = []
        newly_introduced = []
        for i, (argument, argument_type) in enumerate(zip(r_arguments, argument_types)):
            new_arguments.append([name_conversion[argument], argument_type.lower()])
            if new_arguments[-1][0] == "-":
                newly_introduced.append(i)
        new_modes = []
        if len(newly_introduced) <= 1:
            new_modes.append(new_arguments)
        else:
            for i in newly_introduced:
                new_mode = [pair[:] for pair in new_arguments]
                for j in newly_introduced:
                    if i != j:
                        new_mode[j][0] = "+"
                new_modes.append(new_mode)
        lines = []
        for new_mode in new_modes:
            lines.append(mode_pattern.format(r, ','.join(["".join(pair) for pair in new_mode])))
        return lines

    # relation types + atom tests --> modes
    # use lower-case type-names
    # old --> +, new --> -, c --> #
    # algorithm parameters: will be added later
    name_conversion = {"old": "+", "new": "-", "c": "#"}
    s_object = simple_s_parser(s_file)
    s_out = s_file + ".discretized"
    mode_pattern = "mode: {}({})."
    with open(s_out, "w", newline="") as f:
        # descriptive
        for relation, arguments in s_object["[AtomTests]"]:
            modes = create_mode(relation, arguments)
            for mode in modes:
                print(mode, file=f)
        # target
        modes = create_mode(s_object["target"])
        for mode in modes:
            print(mode, file=f)


def discretize_etc(s_file, relation_file, bins=10):
    # works for both target and descriptive, I guess
    # read which are numeric
    out_file = relation_file + ".discretized"
    s_object = simple_s_parser(s_file)
    relations_with_numeric = {}
    for relation, arguments in s_object["[Relations]"].items():
        for i, a in enumerate(arguments):
            if a.startswith('numeric'):
                if relation not in relations_with_numeric:
                    relations_with_numeric[relation] = []
                relations_with_numeric[relation].append(i)
    domains = {r: [[] for _ in num] for r, num in relations_with_numeric.items()}
    with open(relation_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('//') or not line:
                continue
            relation, arguments = split_relation(line)
            if relation in relations_with_numeric:
                for i, j in enumerate(relations_with_numeric[relation]):
                    domains[relation][i].append(float(arguments[j]))
    mappings = {r: [discretize_range(domain, bins) for domain in domain_list] for r, domain_list in domains.items()}
    indices = {r: 0 for r in domains}  # which value is next
    with open(relation_file) as f:
        with open(out_file, "w", newline="") as g:
            for line in f:
                relation, arguments = split_relation(line)
                if relation in relations_with_numeric:
                    num = relations_with_numeric[relation]
                    for i, j in enumerate(num):
                        argument_type = s_object["[Relations]"][relation][j]
                        next_value = mappings[relation][i][indices[relation]]
                        arguments[j] = f"{argument_type}_bin{next_value}"
                    indices[relation] += 1
                line_out = remove_operators(f"{relation}({','.join(arguments)}).")
                print(line_out, file=g)


def discretize_range(xs, bins):
    n = len(xs)
    bins = min(n, bins)
    q = n // bins
    r = n % bins
    sizes = [0] + [q + int(i < r) for i in range(bins)]
    borders = np.cumsum(sizes)
    pairs = sorted(enumerate(xs), key=lambda pair: pair[1])
    try:
        min_interval_values = [pairs[b][1] for b in borders[:-1]] + [pairs[-1][1] + 1]
    except IndexError:
        print(xs, bins)
        raise
    for i in range(1, len(min_interval_values)):
        if min_interval_values[i] <= min_interval_values[i - 1]:
            min_interval_values[i] = min_interval_values[i - 1] + 1
    # create explicit mapping
    mapping = [-1 for _ in xs]
    i_border = 1
    for i, x in pairs:
        while x >= min_interval_values[i_border]:
            i_border += 1
        mapping[i] = i_border
    return mapping


def binarize_target(target_relation_file, target_value):
    output_file_names = [target_relation_file + ".neg", target_relation_file + ".pos"]
    output_target_files = [open(file, "w", newline="") for file in output_file_names]
    with open(target_relation_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            # does not work for mtr but ... you know ...
            i = line.rfind(',')
            last_argument = line[i + 1: line.find(')')].strip()
            out_f = output_target_files[last_argument == target_value]
            out_line = remove_operators(f"{re.sub(' ', '', line[:i])}).")
            print(out_line, file=out_f)
    for file in output_target_files:
        file.close()
    return output_file_names


def get_possible_values(target_relation_file):
    values = {}
    with open(target_relation_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            # does not work for mtr but ... you know ...
            id_value = [c.strip() for c in line[line.find("(") + 1: line.rfind(")")].split(',')]
            example = remove_operators(",".join(id_value[:-1]))
            value = id_value[-1]
            values[example] = value
    return values, sorted(set(values.values()))


def train_test_split(target_relation_file, folds_file, i_fold):
    folds = []
    fresh_fold = None
    with open(folds_file) as f:
        for line in f:
            if line.startswith("|||"):
                if fresh_fold:
                    folds.append(fresh_fold)
                fresh_fold = []
            elif fresh_fold is not None:
                fresh_fold.append(remove_operators(line.strip()))
    test_ids = set(folds[i_fold])
    with open(target_relation_file) as f:
        target_lines = [line.strip() for line in f.readlines()]
    train_test_lines = [[], []]
    for line in target_lines:
        example_id = line[line.find("(") + 1:line.find(")")].strip()
        train_test_lines[example_id in test_ids].append(line)
    train_test_file_name = [target_relation_file + x for x in [".train", ".test"]]
    for file_name, examples in zip(train_test_file_name, train_test_lines):
        with open(file_name, "w", newline="") as f:
            print("\n".join(examples), file=f)
    return train_test_file_name, folds[i_fold]


def move_neg_test_examples_to_pos(file_pos, file_neg):
    with open(file_pos, "a", newline="") as f:
        with open(file_neg) as g:
            for line in g:
                print(line.strip(), file=f)
    with open(file_neg, "w") as _:
        pass


# discretize_etc("../data/fake_data/fake.s", "../data/fake_data/fake_d.txt", 3)
# prepare_background_file("../data/fake_data/fake.s")
# binarize_target("../data/fake_data/fake_t.txt", "nominalTarget")


DATA = {"movie": ("movie", "movie_descriptive.txt", "movie_target.txt", "movie.s", "movie/folds1.txt"),
        "imdb": ("imdb_big", "imdb_big_descriptive.txt", "imdb_big_target.txt", "imdb_big.s", "imdb_big/folds1.txt"),
        "stack": (
            "stack_big", "stack_big_descriptive.txt", "stack_big_target.txt", "stack_big.s", "stack_big/folds1.txt"),
        "nba": ("basket", "basket_descriptive.txt", "basket_target.txt", "basket.s", "basket/folds1.txt"),
        "yelp": (
            "yelp_small", "yelp_small_descriptive.txt", "yelp_small_target.txt", "yelp_small.s",
            "yelp_small/folds1.txt"),
        "uwcse": ("uwcse", "uwcse_descriptive.txt", "uwcse_target.txt", "uwcse.s", "uwcse/folds1.txt"),
        "webkb": ("webkb", "webkb_descriptive.txt", "webkb_target.txt", "webkb.s", "webkb/folds1.txt")
        # "yelp_b": ("yelp_big", "yelp_big_descriptive.txt", "yelp_big_target.txt", "yelp_big.s", "yelp_big/folds1.txt")
        }


def prepare_all():
    for name, descriptive, _, s_file, _ in DATA.values():
        prepare_background_file(f"../data/{name}/{s_file}")
        discretize_etc(f"../data/{name}/{s_file}", f"../data/{name}/{descriptive}")


# prepare_all()
