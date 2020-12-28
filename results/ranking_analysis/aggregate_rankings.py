import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import rc, font_manager
from matplotlib.lines import Line2D

font_size = 20
font_properties = {'family': 'serif', 'serif': ['Computer Modern Roman'],
                   'weight': 'normal', 'size': font_size}

font_manager.FontProperties(family='Computer Modern Roman', style='normal',
                            size=font_size, weight='normal', stretch='normal')
rc('text', usetex=True)
rc('font', **font_properties)

accuracy = {
    "basket": (0.978947368421052, 0.6947368421052632),
    "imdb_big": (-1, -1),
    "imdb_big2": (0.631919237749546, 0.5757713248638838),
    "movie2": (0.567510548523206, 0.5098452883263009),
    "movie": (0.567510548523206, 0.5098452883263009),
    "stack_big": (0.951323654995730, 0.3562766865926558),
    "yelp_big": (0.878600905484995, 0.572298569654233),
    "uwcse": (0.8347826086956521, 0.25217391304347825),
    "mutagenesis": (-1, -1),
    "carcinogenesis": (-1, -1),
    "webkb": (-1, -1),
    "webkb1": (-1, 1),
    "webkb2": (-1, 1)
}
top_feature_names_boostingX = {"basket": ['coachSeas_team', 'teamSeas_assists', 'teamSeas_blocks'],
                               "imdb_big": ['moviesUsers_movieID', 'moviesActors_ranking', 'moviesCity_city'],
                               "movie": ['ratings_movieID'],
                               "stack_big": ['posts_parentID', 'posts_answerCount', 'tags_excerptPostID'],
                               "yelp_big": ['review_businessID', 'checkin_day', 'hours_close'],
                               "uwcse": ['taughtByAssistant_taughtBy', 'taughtBy_taughtBy', 'sessions_sessions'],
                               "mutagenesis": [],
                               "carcinogenesis": ['sbond2_atomID', 'sbond1_atomID', 'sbond7_atomID']
                               }

top_feature_names_boosting = {"basket": ['coachSeas_team', 'teamSeas_3PM', 'draft_selection'],
                              "imdb_big2": ['movieArea_area', 'movieCity_city', 'moviesTags_tagID'],
                              "movie2": ['ratings_movieID', 'movies_releaseDate', 'ratings_stars'],
                              "stack_big": ['posts_parentID', 'postHistory_postID', 'tags_excerptPostID'],
                              "yelp_big": ['attributes_attributes', 'hours_close', 'hours_open'],
                              "uwcse": ['taughtBy_taughtBy', 'advisedBy_discipline', 'authors_authors'],
                              "mutagenesis": ['bonds_bondType', 'atoms_atomType', 'atoms_charge'],
                              "carcinogenesis": ['atom_atomType', 'sbond1_atomID', 'atom_charge']
                              }

top_feature_names_baggingX = {"basket": ['coachSeas_team', 'teamSeas_3PA', 'draft_draftFrom'],
                              "imdb_big": ['movieActors_actorID', 'movies_numberFreshCritics', 'movieRegion_region'],
                              "movie": ['ratings_movieID'],
                              "stack_big": ['posts_parentID', 'tags_excerptPostID', 'post_answerCount'],
                              "yelp_big": ['neighborhoods_neighborhoods', 'attributes_attributes', 'hours_close'],
                              "uwcse": ['taughtByAssistant_taughtBy'],
                              "mutagenesis": [],
                              "carcinogenesis": ['sbond2_atomID', 'sbond7_atomID', 'sbond3_atomID'],
                              "webkb1": ['page_hasTerm', 'links_alfaNumeric']
                              }

top_feature_names_bagging = {"basket": ['coachSeas_team', 'teamSeas_3PA', 'draft_draftFrom'],
                             "imdb_big": ['movieArea_area', 'movieCity_city', 'moviesActors_actorID'],
                             "movie": ['ratings_movieID', 'movies_releaseDate', 'ratings_stars'],
                             "stack_big": ['postHistory_postID', 'posts_parentID', 'tags_excerptPostID'],
                             "yelp_big": ['business_reviewCount', 'checkin_day', 'business_stars'],
                             "uwcse": ['taughtBy_taughtBy', 'advisedBy_discipline', 'authors_authors'],
                             "mutagenesis": ['atoms_atomType', 'atoms_charge', 'bonds_bondType'],
                             "carcinogenesis": ['atom_atomType', 'atom_charge', 'sbond1_atomID'],
                             "webkb1": ['page_hasTerm', 'links_alfaNumeric', 'anchors_anchors']
                             }


def parse_ranking_file(file):
    with open(file) as f:
        lines = [line.strip() for line in f.readlines()]
    skip = 0
    for line in lines:
        skip += 1
        if line.startswith("Attributes summed"):
            break
    importance = {}
    for line in lines[skip:]:
        if not line:
            break
        feature = line[:line.find(" ")]
        try:
            importance_list = eval(line[line.find("["):])
        except:
            print(file, line)
            raise
        assert feature not in importance
        importance[feature] = importance_list
    return importance


def aggregate_trees(experiment_name, scores):
    cumulative_sums = np.cumsum(scores)
    n = len(cumulative_sums)
    assert n == 50
    if "boosting" not in experiment_name:
        cumulative_sums = cumulative_sums / np.arange(1, n + 1)
    return cumulative_sums[-1], cumulative_sums.tolist()


def micro_aggregate_rankings(experiment_name, data_sets=None, n_folds=10):
    """
    Aggregates per-tree rankings into a final one.
    :param experiment_name:
    :param data_sets: list of chosen data sets or None
    :param n_folds:
    :return:
    """
    experiment_dir = f"../{experiment_name}"
    all_rankings = {}
    if data_sets is None:
        data_sets = os.listdir(experiment_dir)
    for data_set in data_sets:
        final_ranking = {}
        for fold in range(n_folds):
            files = [
                os.path.join(
                    experiment_dir, data_set,
                    f"fold{fold}", f"tree{tree}", "results", "experiment_genie3.txt"
                )
                for tree in range(50)
            ]
            rankings = [parse_ranking_file(file) for file in files]
            for i, ranking in enumerate(rankings):
                for key in ranking:
                    if key not in final_ranking:
                        final_ranking[key] = [0.0 for _ in range(50)]
                    final_ranking[key][i] += ranking[key][0] / n_folds
        all_rankings[data_set] = {key: aggregate_trees(experiment_name, value)
                                  for key, value in final_ranking.items()}
    experiment_name = experiment_name[experiment_name.rfind("/") + 1:]
    with open(f"{experiment_name}_genie3.txt", "w") as f:
        print(all_rankings, file=f)


def aggregate_rankings(experiment_name, ensemble_type=""):
    n = 50
    importance_all = {d: {} for d in accuracy}
    for d in accuracy:
        if not os.path.exists(f"../{experiment_name}/{d}") or (experiment_name == "playing_around" and "genesis" in d):
            print(f"../{experiment_name}/{d} does not exist")
            del importance_all[d]
            continue
        importance = {}
        if experiment_name == "playing_around":
            genie3_files = ["../{}/{}/genie3_{}Exist.txt".format(experiment_name, d, ensemble_type)]
        else:
            genie3_files = []
            for fold in range(10):
                if fold == 9 and d == "mutagenesis" and experiment_name == "boosting":
                    print("Skipping critical fold for muta")
                    continue
                genie3_files.append("../{}/{}/fold{}/results/experiment_genie3.txt".format(experiment_name, d, fold))
        for genie3 in genie3_files:
            assert os.path.exists(genie3), genie3
            importance_this = parse_ranking_file(genie3)
            for k, v in importance_this.items():
                if k not in importance:
                    importance[k] = [v]
                else:
                    importance[k].append(v)
        for k, v in importance.items():
            assert set(len(i) for i in v) == {n}
            for _ in range(len(v) - 10):
                v.append([0.0 for _ in range(n)])
        for k, v in importance.items():
            aggregated = np.mean(v, axis=0)
            # cumulative = np.cumsum(aggregated).tolist()
            importance_all[d][k] = aggregate_trees(experiment_name, aggregated)  # (sum(aggregated), cumulative)
    with open(f"{experiment_name}{ensemble_type}_genie3.txt", "w") as f:
        print(importance_all, file=f)


def aggregate_movie():
    experiment_name = "playing_around"
    importance_all = {"movie": {}}
    importance = {}
    genie3_files = ["../{}/movie/movie_genie3.txt".format(experiment_name, )]
    for genie3 in genie3_files:
        assert os.path.exists(genie3), genie3
        importance_this = parse_ranking_file(genie3)
        for k, v in importance_this.items():
            if k not in importance:
                importance[k] = [v]
            else:
                importance[k].append(v)
    for k, v in importance.items():
        aggregated = np.mean(v, axis=0)
        importance_all["movie"][k] = aggregate_trees(experiment_name, aggregated)  # (sum(aggregated), cumulative)
    with open(f"movie_bagging_genie3.txt", "w") as f:
        print(importance_all, file=f)


# aggregate_movie()
# aggregate_rankings("bagging", "")
# micro_aggregate_rankings("bagging_by_tree", ["carcinogenesis", "mutagenesis"])
# micro_aggregate_rankings("webkbTakeTwo/webkb_take_twoX_bagging_by_tree", n_folds=1)


def join_files(files, out_file):
    combined = {}
    n_keys = 0
    for file in files:
        with open(file) as f:
            new = eval(f.readline())
            n_keys += len(new)
        combined = {**combined, **new}
    if n_keys != len(combined):
        print("Warning: some keys repeat!")
    print("Data sets:", len(list(combined.keys())), list(combined.keys()))
    with open(out_file, "w") as f:
        print(combined, file=f)


# join_files(["bagging_genie3.txt", "bagging_by_tree_genie3.txt",
#             "webkb_take_two_bagging_by_tree_genie3.txt", "movie_bagging_genie3.txt"],
#            "joined_bagging_genie3.txt")
# join_files(["onlyX_bagging_genie3.txt", "playing_around_onlyX_bagging_genie3.txt",
#             "webkb_take_twoX_bagging_by_tree_genie3.txt"],
#            "joined_onlyX_bagging_genie3.txt")


def multi_line(xs, ys, c, ax=None, feature_names=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax : (optional) Axes to plot on.
    feature_names : (optional) passed to legend
    kwargs : (optional) passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    lc.set_array(np.asarray(c))

    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel("number of trees")
    ax.set_ylabel("feature importance")

    if feature_names is not None:
        proxies = []
        for c_value, _ in zip(c, feature_names):
            color = lc.cmap(lc.norm(c_value / max(c)))
            proxies.append(Line2D([0, 1], [0, 1], color=color))
        ax.legend(proxies, feature_names, loc=1, bbox_to_anchor=(1.8, 1.0))
    return lc


def plot_ranking(rankin_file, d, top_feature_names=None, ensemble_type=""):
    def nicify(n):
        return re.sub("_", "\\\\_", n)

    with open(rankin_file) as f:
        data_all = eval(f.readline())
        if d in data_all:
            data = data_all[d]
        else:
            print("Skipping", d)
            return

    features = list(data.keys())
    features.sort(key=lambda feature: data[feature][0], reverse=True)
    final_importance = [data[feature][0] for feature in features]
    if "boosting" in rankin_file:
        for values in data.values():
            values[1].insert(0, 0.0)  # start from 0
    print(d, features[:15])

    n_lines = len(features)
    x = np.arange(len(data[features[0]][1])) if features else np.arange(50)

    ys = np.array([data[feature][1] for feature in features])
    xs = np.array([x for _ in range(n_lines)])

    plt.figure(figsize=(8, 4))
    # fig, ax = plt.subplots()
    if top_feature_names is not None:
        if d in top_feature_names:
            feature_names = [nicify(f) for f in top_feature_names[d]]
        else:
            print("Extend top_feature_names with", d)
            feature_names = [nicify(f) for f in features][:3]
    else:
        feature_names = [nicify(f) for f in features][:3]
    final_importance = [x / max(final_importance) for x in final_importance]
    multi_line(xs, ys, final_importance, feature_names=feature_names, cmap='Blues', lw=1,
               norm=plt.Normalize(0, 1))  # lc = multi_line ...
    # fig.colorbar(lc)
    plt.tight_layout()
    plt.xticks(range(0, 51, 10))
    # name = d if "_" not in d else d[: d.find("_")]
    # plt.title("Data: {} acc: {:.3f}/{:.3f}".format(name, *accuracy[d]))
    plt.show()
    return
    fig_name = f"plots/{ensemble_type}/{d}.pdf"
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    if not os.path.exists(fig_name):
        plt.savefig(fig_name)
    else:
        print("Skipping", fig_name)
    plt.cla()
    plt.clf()


def plot_all(ranking_file, ensemble_type, top_features):
    for d in accuracy:
        plot_ranking(ranking_file, d, top_feature_names=top_features, ensemble_type=ensemble_type)


plot_all("joined_bagging_genie3.txt", "bagging", top_feature_names_bagging)
# plot_all("joined_onlyX_bagging_genie3.txt", "baggingExist", top_feature_names_baggingX)
# plot_all("boosting_genie3.txt", "boosting", top_feature_names_boosting)
# plot_all("onlyX_boosting_genie3.txt", "boostingExist", top_feature_names_boostingX)
