from typing import List, Union
from ..data.data_and_statistics import Dataset
import random

def create_folds(data: Dataset,
                 folds_file: Union[str, None] = None,
                 example_ids: Union[List[List[str]], None] = None,
                 random_seed: Union[int, None] = 2864,
                 n_folds=10):
    """
    Creates folds iterator.
    :param data: a Dataset object
    :param folds_file: one can specify their own test folds in a file of the following form:
    |||
    exampleID1_fold1
    ...
    exampleIDn1_fold1
    |||
    ...
    |||
    exampleID1_fold_k
    ...
    exampleIDn_k_fold_k
    |||
    :param example_ids: example IDs can be directly provided as a list of list of strings
    :param random_seed: used for reproducibility when neither folds_file nor example_ids are provided, and the data
    is randomly subdivided into folds
    :param n_folds:
    :return: for each fold, a training and test Dataset objects are yielded
    """
    target_relation = data.get_target_data()
    id_to_datum = {d.descriptive_part[0]: d for d in target_relation}
    folds = []
    fresh_fold = None
    fold_separator = "|||"
    if folds_file is not None:
        with open(folds_file) as f:
            for line in f:
                if line.startswith(fold_separator):
                    if fresh_fold:
                        folds.append(fresh_fold)
                    fresh_fold = []
                elif fresh_fold is not None:
                    fresh_fold.append(line.strip())
    else:
        if example_ids is None:
            example_ids = [[] for _ in range(n_folds)]
            examples = list(id_to_datum.keys())
            if random_seed is not None:
                random.seed(random_seed)
                random.shuffle(examples)
            for i, example in enumerate(examples):
                example_ids[i % n_folds].append(example)
        folds = example_ids
    assert folds is not None
    for i, fold1 in enumerate(folds):
        training_testing_target = [[], []]
        for j, fold2 in enumerate(folds):
            for d in fold2:
                training_testing_target[i == j].append(id_to_datum[d])
        training_data = Dataset(
            settings=data.settings,
            data_file=data.data_file,
            descriptive_relations=data.get_descriptive_data(),
            target_data=training_testing_target[0],
            statistics=data.get_copy_statistics())
        testing_data = Dataset(
            settings=data.settings,
            data_file=data.data_file,
            descriptive_relations=data.get_descriptive_data(),
            target_data=training_testing_target[1],
            statistics=data.get_copy_statistics())
        yield training_data, testing_data
