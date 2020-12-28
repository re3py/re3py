import os
from collect_from_cluster_grid import load_accuracy, try_unzip
import numpy as np
import matplotlib.pyplot as plt


def collect_results_for_dataset(data_dir):
    configurations = {}
    for fold in range(10):
        fold_dir = os.path.join(data_dir, "fold{}".format(fold))
        configuration_dirs = [(os.path.join(fold_dir, d, "results"), d)
                              for d in os.listdir(fold_dir) if os.path.isdir(os.path.join(fold_dir, d))]
        for configuration_dir, configuration in configuration_dirs:
            c = tuple(configuration[len("tocka"):].split('_'))
            if c not in configurations:
                configurations[c] = []
            files_to_collect = ["experiment.log"]
            is_ok, file_path = try_unzip(configuration_dir, files_to_collect, "experiment.tar.gz")
            if not is_ok:
                print("Something missing for", configuration_dir)
                raise ValueError()
            _, a_test = load_accuracy(os.path.join(configuration_dir, "experiment.log"))
            a_test.evaluate()
            configurations[c].append(a_test.measure_value)
    return configurations


def analyze_data():
    for ds in ["basket", "imdb_big", "movie", "stack_big", "yelp_small"]:
        configurations = collect_results_for_dataset("../experiments/boosting_search/{}".format(ds))
        avg_accuracy = [(np.mean(ys), np.var(ys)**0.5, c) for c, ys in configurations.items()]
        avg_accuracy.sort(key=lambda t: (-t[0], t[1]))
        print(ds)
        for i in range(10):
            print(avg_accuracy[i])
        print("\n" * 2)
        plt.plot([y[0] for y in avg_accuracy], label=ds)
    plt.legend()
    plt.show()


analyze_data()