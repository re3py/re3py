from ...data.data_and_statistics import Dataset, Datum
from .variables import Variable
# from relation import Relation
from typing import Dict, List, Tuple, Union
from py4j.java_collections import ListConverter, MapConverter
from ...data.relation import Relation
from .aggregators import Aggregator


def send_data(data: Dataset, client, wrapper):
    def l_convert(l0):
        return list_converter.convert(l0, client)

    def m_convert(m0):
        return map_converter.convert(m0, client)

    list_converter = ListConverter()
    map_converter = MapConverter()

    # send target_data: we can send only descriptive parts of the data
    simplified_target = []
    for d in data:
        simplified_target.append([d.identifier])
        simplified_target[-1] += list(d.get_descriptive())
        simplified_target[-1] = l_convert(simplified_target[-1])
    simplified_target = l_convert(simplified_target)
    # target data
    wrapper.load_target_data(simplified_target)
    # relations
    relations = {
        r.name: l_convert(r.get_types())
        for r in data.get_descriptive_data().values()
    }
    wrapper.load_relations(m_convert(relations), data.data_file)

    # wrapper.load(l_convert([m_convert({2: "we", 3: "d"}), m_convert({"3": 2, "21": 21})]))


def send_variables(example: Dict[str, Variable], client, wrapper):
    def l_convert(l0):
        return list_converter.convert(l0, client)

    list_converter = ListConverter()

    variables = []
    for variable in example.values():
        v_name = variable.get_name()
        v_type = variable.value_type
        v_value = variable.get_value()
        v_can_vary = variable.can_vary()
        variables.append(l_convert([v_name, v_type, v_value, v_can_vary]))
    wrapper.load_variables(l_convert(variables))


def compute_test_values(target_data: List[Datum],
                        target_relation_vars: List[str],
                        rc_modified: List[Tuple[Relation, List[str]]],
                        filtered_agg_chains: List[List[Aggregator]],
                        r_key: Tuple[Tuple[str, Tuple[bool],
                                           Tuple[Union[str, int, Variable]]]],
                        a_keys: List[Tuple[str]], nb_fresh_vars: int,
                        fresh_indices: List[int],
                        known_unknown: List[List[List[int]]], client, wrapper):
    def l_convert(l0):
        return list_converter.convert(l0, client)

    # def m_convert(m0):
    #     return map_converter.convert(m0, client)

    list_converter = ListConverter()
    # map_converter = MapConverter()

    # set_example_values_time = 0
    # find_values_time = 0

    # conversion
    target_data_converted = l_convert([d.identifier for d in target_data])

    target_relation_vars_converted = l_convert(target_relation_vars)

    rc_modified_converted = []
    for r, variables in rc_modified:
        rc_modified_converted.append(l_convert([r.get_name()] + variables))
    rc_modified_converted = l_convert(rc_modified_converted)

    filtered_agg_chains_converted = []
    for aggregators in filtered_agg_chains:
        filtered_agg_chains_converted.append(
            l_convert([a.get_name() for a in aggregators]))
    filtered_agg_chains_converted = l_convert(filtered_agg_chains_converted)

    if r_key is None:
        r_key_converted = None
    else:
        r_key_converted = []
        for name, booleans, variables in r_key:
            n = len(booleans)
            assert n == len(variables)
            new_element = ["" for _ in range(1 + 2 * n)]
            new_element[0] = name
            for i in range(n):
                new_element[1 + i] = str(booleans[i]).lower()
                new_element[1 + n + i] = variables[i]
            r_key_converted.append(l_convert(new_element))
        r_key_converted = l_convert(r_key_converted)

    a_keys_converted = []
    for t in a_keys:
        a_keys_converted.append(l_convert(list(t)))
    a_keys_converted = l_convert(a_keys_converted)

    fresh_indices_converted = l_convert(fresh_indices)

    known_unknown_converted = []
    for known, unknown in known_unknown:
        known_unknown_converted.append(
            l_convert([l_convert(known), l_convert(unknown)]))
    known_unknown_converted = l_convert(known_unknown_converted)

    values = wrapper.compute_test_values(
        target_data_converted, target_relation_vars_converted,
        rc_modified_converted, filtered_agg_chains_converted, r_key_converted,
        a_keys_converted, nb_fresh_vars, fresh_indices_converted,
        known_unknown_converted)
    all_test_values = [[x for x in part] for part in values]
    return all_test_values
