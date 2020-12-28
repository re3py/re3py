from utilities.my_exceptions import WrongValueException
from utilities.my_utils import try_convert_to_number
import os
import re

numeric = "numeric"
nominal = "nominal"
key = "key"


def append_to_file_name(f_name, appendix):
    i = f_name.rfind(".")
    if i < 0:
        return f_name + appendix
    else:
        return f_name[:i] + appendix + f_name[i:]


def arff_to_relations(arff_file, target_index, out_dir, key_index=None):
    """
    Converts non-sparse arff to the relational form.
    This results in two files: one for target relation and one for descriptive relations.
    :param arff_file:
    :param target_index: 0-based index of the target attribute
    :param out_dir:
    :param key_index: None of 0-based index of the key attribute
    :return:
    """
    def get_attribute_type(vs):
        if vs.lower() == numeric:
            return numeric
        elif vs[0] == "{" and vs[-1] == "}":
            return nominal
        elif vs.lower() == key:
            return key
        else:
            raise WrongValueException("Don't know the type of {}.".format(vs))

    def get_file_name(path):
        normalized = re.sub("\\\\", "/", path)
        return normalized[normalized.rfind("/") + 1:]

    def make_out_dir_if_necessary(path):
        normalized = re.sub("\\\\", "/", path)
        if "/" not in normalized:
            normalized += "/"
        output_dir = normalized[:normalized.rfind("/")]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # handle the output file names etc.
    arff_name = get_file_name(arff_file)
    out_dir = re.sub("\\\\", "/", out_dir)
    if out_dir[-1] != "/":
        out_dir += "/"
    descriptive_relations_file = out_dir + append_to_file_name(
        arff_name, "_descriptive")
    target_relations_file = out_dir + append_to_file_name(arff_name, "_target")
    make_out_dir_if_necessary(target_relations_file)
    make_out_dir_if_necessary(descriptive_relations_file)
    # convert
    f = open(arff_file)
    attributes = []  # [(attribute name, type), ...]
    data = "@data"
    attribute = "@attribute"
    for raw_line in f:
        if data in raw_line.lower():
            break
        if raw_line.lower().startswith(attribute):
            pattern = "{} +([^ ]+) +(.+)".format(attribute)
            match = re.search(pattern, raw_line, flags=re.I)
            name, values = match.group(1), match.group(2).strip()
            attributes.append((name, get_attribute_type(values)))
    out_descriptive = open(descriptive_relations_file, "w")
    out_target = open(target_relations_file, "w")
    examples = 0
    for raw_line in f:
        line = raw_line.strip()
        if line:
            examples += 1
            values = [v.strip() for v in line.split(",")]
            assert len(values) == len(attributes)
            example_id = "ex{}".format(
                examples) if key_index is None else values[key_index]
            for i, (v, (a_name, a_type)) in enumerate(zip(values, attributes)):
                if i != key_index and v != "?":
                    if a_type == numeric:
                        v = try_convert_to_number(v)
                    out_f = out_descriptive if i != target_index else out_target
                    print("{}({}, {})".format(a_name, example_id, v),
                          file=out_f)
    out_descriptive.close()
    out_target.close()
    f.close()
