import unittest
from core import *
from data import *
from heuristic import HeuristicGini


class CoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_const1 = [
            Datum(("irrelevant", ), "a", 2, 1),
        ]
        cls.data_const2 = [
            Datum(("irrelevant", ), "a", 1, 2),
            Datum(("irrelevant", ), "a", 2, 3),
            Datum(("irrelevant", ), "a", 3, 4),
            Datum(("irrelevant", ), "a", 4, 5),
            Datum(("irrelevant", ), "a", 5, 6),
            Datum(("irrelevant", ), "a", 4, 7),
            Datum(("irrelevant", ), "a", 3, 8),
            Datum(("irrelevant", ), "a", 2, 9),
            Datum(("irrelevant", ), "a", 1, 0)
        ]

        cls.father = Relation("fatherOf", None, "test_data/fatherOf.txt", ["Person", "Person"])
        cls.age = Relation("age", None, "test_data/age.txt", ["Person", "numeric"])
        cls.friend = Relation("friend", None, "test_data/friend.txt", ["Person", "Person"])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_find_nominal_const_class(self):
        # Constant class --> only useless splits
        for target_data in [self.data_const1, self.data_const2]:
            stats = NodeStatisticsClassification(["a", "b"])
            stats.add_examples(target_data)
            heuristic = HeuristicGini()
            current_variability = heuristic.compute_variability(stats)
            if len(target_data) == 1:
                correct_score = BinarySplit.worst_split_score
            else:
                correct_score = current_variability
            tree = DecisionTree(HeuristicGini(), stats)
            tree_node = TreeNode("", None, None, None, stats, 0)
            xs = [str(x) for x in range(len(target_data))]
            score, comparator, subset, partition, _ = tree.find_best_nominal(tree_node, xs, target_data,
                                                                             [], TYPE_NOMINAL)
            self.assertEquals(score, correct_score)

    def test_find_nominal_useless_splits_not_constant_class(self):
        target_data = [
            Datum(("irrelevant", ), "a", 2, 0),  # group 0
            Datum(("irrelevant", ), "b", 2, 1),  # group 1
            Datum(("irrelevant", ), "b", 2, 2),  # group 1
            Datum(("irrelevant", ), "a", 5, 3),  # group 1
            Datum(("irrelevant", ), "a", 3, 4),  # group 0
            Datum(("irrelevant", ), "b", 4, 5)   # group 0
        ]
        stats = NodeStatisticsClassification(["a", "b"])
        stats.add_examples(target_data)
        heuristic = HeuristicGini()
        current_variability = heuristic.compute_variability(stats)
        tree_node = TreeNode("", None, None, None, stats, 0)
        xs = ["0", "1", "1", "1", "0", "0"]
        tree = DecisionTree(HeuristicGini(), stats)
        score, comparator, subset, partition, no_vars = tree.find_best_nominal(tree_node, xs, target_data,
                                                                               [], TYPE_NOMINAL)
        self.assertEquals(score, current_variability)

    def test_find_nominal_perfect(self):
        target_data = [
            Datum(("irrelevant", ), "a", 2, 0),
            Datum(("irrelevant", ), "b", 2, 1),
            Datum(("irrelevant", ), "b", 2, 2),
            Datum(("irrelevant", ), "a", 5, 3),
            Datum(("irrelevant", ), "a", 3, 4),
            Datum(("irrelevant", ), "b", 4, 5)
        ]
        stats = NodeStatisticsClassification(["a", "b"])
        stats.add_examples(target_data)
        tree_node = TreeNode("", None, None, None, stats, 0)
        xs = ["0", "3", "4", "5", "1", "2"]
        tree = DecisionTree(HeuristicGini(), stats)
        score, comparator, subset, partition, has_no_variables = tree.find_best_nominal(tree_node,
                                                                                        xs,
                                                                                        target_data,
                                                                                        [],
                                                                                        TYPE_NOMINAL,
                                                                                        float('inf')
                                                                                        )
        self.assertSetEqual(subset, {"2", "3", "4"})
        self.assertEqual(comparator, DOES_NOT_CONTAIN)

    def test_find_nominal_ok_not_perfect(self):
        target_data = [
            Datum(("irrelevant", ), "a", 2, 0),
            Datum(("irrelevant", ), "b", 2, 1),
            Datum(("irrelevant", ), "b", 2, 2),
            Datum(("irrelevant", ), "a", 5, 3),
            Datum(("irrelevant", ), "a", 3, 4),
            Datum(("irrelevant", ), "b", 4, 5)
        ]
        stats = NodeStatisticsClassification(["a", "b"])
        stats.add_examples(target_data)
        tree_node = TreeNode("", None, None, None, stats, 0)
        xs = ["1", "2", "0", "0", "2", "1"]
        tree = DecisionTree(HeuristicGini(), stats)
        score, comparator, subset, partition, _ = tree.find_best_nominal(tree_node, xs, target_data, [], TYPE_NOMINAL)
        self.assertSetEqual(subset, {"1"})
        self.assertEqual(comparator, DOES_NOT_CONTAIN)

    def test_find_nominal_with_target_variables(self):
        target_data = [
            Datum(("x", "1"), "a", 2, 0),
            Datum(("x", "22"), "b", 2, 1),
            Datum(("x", "00"), "b", 2, 2),
            Datum(("x", "00"), "a", 5, 3),
            Datum(("x", "22"), "a", 3, 4),
            Datum(("x", "1"), "b", 4, 5)
        ]
        stats = NodeStatisticsClassification(["a", "b"])
        stats.add_examples(target_data)
        tree_node = TreeNode("", None, None, None, stats, 0)
        xs = ["1", "2", "0", "0", "2", "1"]
        tree = DecisionTree(HeuristicGini(), stats)
        score, comparator, subset, partition, _ = tree.find_best_nominal(tree_node, xs, target_data,
                                                                         ["X1"], TYPE_TYPE)
        self.assertSetEqual(subset, {"X1"})
        self.assertEqual(comparator, DOES_NOT_CONTAIN)

        target_data = [
            Datum(("1", "z", "x"), "a", 2, 0),
            Datum(("x", "z", "x"), "b", 2, 1),
            Datum(("x", "z", "x"), "b", 2, 2),
            Datum(("x", "z", "0"), "a", 5, 3),
            Datum(("x", "z", "2"), "a", 3, 4),
            Datum(("x", "z", "x"), "b", 4, 5)
        ]
        stats = NodeStatisticsClassification(["a", "b"])
        stats.add_examples(target_data)
        tree_node = TreeNode("", None, None, None, stats, 0)
        xs = ["1", "2", "0", "0", "2", "1"]
        tree = DecisionTree(HeuristicGini(), stats)
        score, comparator, subset, partition, _ = tree.find_best_nominal(tree_node, xs, target_data,
                                                                         ["X2", "X0"], TYPE_TYPE)
        self.assertSetEqual(subset, {"X0", "X2"})
        self.assertEqual(comparator, CONTAINS)

    def test_find_numeric_const_class(self):
        # Constant class --> only useless splits
        correct_scores = [BinarySplit.worst_split_score, 0.0]
        for target_data, correct_score in zip([self.data_const1, self.data_const2], correct_scores):
            data = Dataset(target_data=target_data)
            stats = NodeStatisticsClassification(["a", "b"])
            stats.add_examples(target_data)
            tree_node = TreeNode("", None, None, None, stats, 0)
            xs = list(range(len(data.get_target_data())))
            tree = DecisionTree(HeuristicGini(), stats)
            score, comparator, threshold, partition = tree.find_best_numeric(tree_node, xs, target_data)
            self.assertEquals(score, correct_score)

    def test_find_numeric_useless_splits_not_constant_class(self):
        target_data = [
            Datum(("irrelevant", ), "a", 2, 0),  # group 0
            Datum(("irrelevant", ), "b", 2, 1),  # group 1
            Datum(("irrelevant", ), "b", 2, 2),  # group 1
            Datum(("irrelevant", ), "a", 5, 3),  # group 1
            Datum(("irrelevant", ), "a", 3, 4),  # group 0
            Datum(("irrelevant", ), "b", 4, 5)   # group 0
        ]
        stats = NodeStatisticsClassification(["a", "b"])
        stats.add_examples(target_data)
        heuristic = HeuristicGini()
        current_variability = heuristic.compute_variability(stats)
        tree_node = TreeNode("", None, None, None, stats, 0)
        xs = [2.5, 2.6, 2.6, 2.6, 2.5, 2.5]
        tree = DecisionTree(HeuristicGini(), stats)
        score, comparator, threshold, partition = tree.find_best_numeric(tree_node, xs, target_data)
        self.assertEquals(score, current_variability)

    def test_find_numeric_perfect(self):
        target_data = [
            Datum(("irrelevant", ), "a", 2, 0),
            Datum(("irrelevant", ), "b", 2, 1),
            Datum(("irrelevant", ), "b", 2, 2),
            Datum(("irrelevant", ), "a", 5, 3),
            Datum(("irrelevant", ), "a", 3, 4),
            Datum(("irrelevant", ), "b", 4, 5)
        ]
        stats = NodeStatisticsClassification(["a", "b"])
        stats.add_examples(target_data)
        tree_node = TreeNode("", None, None, None, stats, 0)
        # all a on the left, all b s on the right
        xs = [1.1, 10.1, 10.05, 2.0, 1.5, 8.0]
        tree = DecisionTree(HeuristicGini(), stats)
        score, comparator, threshold, partition = tree.find_best_numeric(tree_node, xs, target_data)
        self.assertAlmostEqual(threshold, 5.0, 10)

    def test_find_numeric_ok_not_perfect(self):
        target_data = [
            Datum(("irrelevant", ), "a", 2, 0),
            Datum(("irrelevant", ), "b", 2, 1),
            Datum(("irrelevant", ), "b", 2, 2),
            Datum(("irrelevant", ), "a", 5, 3),
            Datum(("irrelevant", ), "a", 3, 4),
            Datum(("irrelevant", ), "b", 4, 5)
        ]
        stats = NodeStatisticsClassification(["a", "b"])
        stats.add_examples(target_data)
        tree_node = TreeNode("", None, None, None, stats, 0)
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        tree = DecisionTree(HeuristicGini(), stats)
        score, comparator, threshold, partition = tree.find_best_numeric(tree_node, xs, target_data)
        # self.assertAlmostEqual(score, 0.17636684303350963, 10)
        self.assertAlmostEqual(threshold, 5.5, 10)
        xs = xs[::-1]
        score, comparator, threshold, partition = tree.find_best_numeric(tree_node, xs, target_data)
        # self.assertAlmostEqual(score, 0.17636684303350963, 10)
        self.assertAlmostEqual(threshold, 1.5, 10)

    def test_memo_keys(self):
        f1 = VariableVariable("Y1", "Person", None)
        f2 = VariableVariable("Y2", "Person", None)
        f3 = ConstantVariable("C3", "c", "cool")
        b1 = VariableVariable("X1", "Person", None)
        b2 = VariableVariable("X2", "Person", None)
        example = {"Y1": f1, "X1": b1, "Y2": f2, "C3": f3, "X2": b2}
        relation_chain = [(self.father, ["Y1", "X1"]),
                          (self.age, ["Y1", "Y1"]),
                          (self.father, ["X2", "Y2"])]
        aggregator_chain1 = (MEAN, )
        aggregator_chain2 = (MEAN, MIN)
        aggregator_chain3 = (MEAN, MIN, MAX)
        a_chains = [aggregator_chain1, aggregator_chain2, aggregator_chain3]
        solutions = [
            (
                (
                    ('fatherOf', (False, False), (0, 'X1')),
                ),
                [('mean',)],
                [[[1], [0]]]
            ),
            (
                (
                    ('fatherOf', (False, False), (0, 'X1')),
                    ('age', (False, False), (0, 0))
                ),
                [('mean', 'min')],
                [[[1], [0]], [[0, 1], []]]
            ),
            (
                (
                    ('fatherOf', (False, False), (0, 'X1')),
                    ('age', (False, False), (0, 0)),
                    ('fatherOf', (False, False), ('X2', 1))
                ),
                [('mean', 'min', 'max')],
                [[[1], [0]], [[0, 1], []], [[0], [1]]]
            )
        ]
        for i in range(3):
            memo_keys = DecisionTree.test_values_memo_keys(example,
                                                           relation_chain[:i + 1],
                                                           [a_chains[i]])
            self.assertEqual(memo_keys, solutions[i])

        relation_chain = [(self.father, ["X1", "Y1"]),
                          (self.age, ["Y2", "Y1"]),
                          (self.father, ["X2", "Y2"]),
                          (self.age, ["Y1", "C3"])]
        a_chains = [(MEAN, MEAN, MIN, MIN), (MODE, MAX, MODE, MAX)]
        solution = (
                (
                    ('fatherOf', (False, False), ('X1', 0)),
                    ('age', (False, False), (1, 0)),
                    ('fatherOf', (False, False), ('X2', 1)),
                    ('age', (False, True), (0, 'cool'))
                ),
                [
                    ('mean', 'mean', 'min', 'min'),
                    ('mode', 'max', 'mode', 'max')
                ],
                [
                    [[0], [1]],
                    [[1], [0]],
                    [[0, 1], []],
                    [[0, 1], []]
                ]
            )
        memo_keys = DecisionTree.test_values_memo_keys(example,
                                                       relation_chain,
                                                       a_chains)
        self.assertEqual(memo_keys, solution)

    def test_aggregator_chain_empty(self):
        t = DecisionTree(allowed_aggregators=[])
        chains1 = list(t.generate_possible_aggregator_chains(1, [("var name", "numeric1")], [123]))
        chains2 = list(t.generate_possible_aggregator_chains(2, [("var name", "nominal3")], [123]))
        chains3 = list(t.generate_possible_aggregator_chains(3, [("var name", "nominal2")], [123]))
        self.assertListEqual([], chains1)
        self.assertListEqual([], chains2)
        self.assertListEqual([], chains3)

    def test_aggregator_chain_one_fresh(self):
        t = DecisionTree(allowed_aggregators=["sum", "count", "mode"])
        chains1 = list(t.generate_possible_aggregator_chains(1, [("var name", "numeric1")], [12]))
        chains2 = list(t.generate_possible_aggregator_chains(2, [("var name", "nominal3")], [12]))
        chains3 = list(t.generate_possible_aggregator_chains(3, [("var name", "numeric2")], [12]))
        solution1 = [[COUNT], [SUM]]
        solution2 = [[COUNT, MODE], [MODE, MODE], [SUM, COUNT]]
        solution3 = [[SUM, SUM, COUNT], [SUM, SUM, SUM]]
        self.assertListEqual([(c, TYPE_NUMERIC) for c in solution1],
                             sorted(chains1))
        self.assertListEqual([(c, TYPE_NUMERIC) if c[0] != MODE else (c, "nominal3")
                              for c in solution2],
                             sorted(chains2))
        self.assertListEqual([(c, TYPE_NUMERIC) for c in solution3],
                             sorted(chains3))

    def test_aggregator_chain_one_fresh_ends_with_flat(self):
        t = DecisionTree(allowed_aggregators=["flatten", "flattenUnique",
                                              "count", "countUnique",
                                              "min", "max", "mean", "sum",
                                              "mode"]
                         )
        chains1 = list(t.generate_possible_aggregator_chains(3, [("var name", "numeric1")], [0]))
        chains2 = list(t.generate_possible_aggregator_chains(3, [("var name", "nominal")], [78]))
        chains3 = list(t.generate_possible_aggregator_chains(3, [("var name", "Horse")], [21]))
        end_with_flatten = [[COUNT, FLATTEN, FLATTEN], [COUNT_UNIQUE, FLATTEN, FLATTEN],
                            [MIN, FLATTEN, FLATTEN], [MAX, FLATTEN, FLATTEN],
                            [MEAN, FLATTEN, FLATTEN], [SUM, FLATTEN, FLATTEN],
                            [COUNT, FLATTEN_UNIQUE, FLATTEN], [COUNT_UNIQUE, FLATTEN_UNIQUE, FLATTEN],
                            [MIN, FLATTEN_UNIQUE, FLATTEN], [MAX, FLATTEN_UNIQUE, FLATTEN],
                            [MEAN, FLATTEN_UNIQUE, FLATTEN], [SUM, FLATTEN_UNIQUE, FLATTEN],
                            [MIN, COUNT, FLATTEN], [MAX, COUNT, FLATTEN],
                            [MEAN, COUNT, FLATTEN], [SUM, COUNT, FLATTEN],
                            [MIN, COUNT_UNIQUE, FLATTEN], [MAX, COUNT_UNIQUE, FLATTEN],
                            [MEAN, COUNT_UNIQUE, FLATTEN], [SUM, COUNT_UNIQUE, FLATTEN],
                            [MIN, MIN, FLATTEN], [MAX, MIN, FLATTEN], [MEAN, MIN, FLATTEN], [SUM, MIN, FLATTEN],
                            [MIN, MAX, FLATTEN], [MAX, MAX, FLATTEN], [MEAN, MAX, FLATTEN], [SUM, MAX, FLATTEN],
                            [MIN, MEAN, FLATTEN], [MAX, MEAN, FLATTEN], [MEAN, MEAN, FLATTEN], [SUM, MEAN, FLATTEN],
                            [MIN, SUM, FLATTEN], [MAX, SUM, FLATTEN], [MEAN, SUM, FLATTEN], [SUM, SUM, FLATTEN]]
        end_with_flatten_unique = [chain[:2] + [FLATTEN_UNIQUE] for chain in end_with_flatten]
        solution1 = sorted(end_with_flatten + end_with_flatten_unique)
        solution1 = [(x, TYPE_NUMERIC) for x in solution1]
        answer = sorted([c for c in chains1 if c[0][-1] in [FLATTEN, FLATTEN_UNIQUE]])
        self.assertListEqual(solution1, answer)
        solution2 = [[MODE, FLATTEN, FLATTEN], [MODE, FLATTEN_UNIQUE, FLATTEN],
                     [COUNT, FLATTEN, FLATTEN], [COUNT, FLATTEN_UNIQUE, FLATTEN],
                     [COUNT_UNIQUE, FLATTEN, FLATTEN], [COUNT_UNIQUE, FLATTEN_UNIQUE, FLATTEN],
                     [MODE, MODE, FLATTEN], [COUNT, MODE, FLATTEN], [COUNT_UNIQUE, MODE, FLATTEN],
                     [MIN, COUNT, FLATTEN], [MAX, COUNT, FLATTEN], [MEAN, COUNT, FLATTEN], [SUM, COUNT, FLATTEN],
                     [MIN, COUNT_UNIQUE, FLATTEN], [MAX, COUNT_UNIQUE, FLATTEN],
                     [MEAN, COUNT_UNIQUE, FLATTEN], [SUM, COUNT_UNIQUE, FLATTEN]]
        solution2 += [chain[:2] + [FLATTEN_UNIQUE] for chain in solution2]
        solution2 += [[MODE, MODE, MODE], [COUNT, MODE, MODE], [COUNT_UNIQUE, MODE, MODE],
                      [MODE, FLATTEN, MODE], [COUNT, FLATTEN, MODE], [COUNT_UNIQUE, FLATTEN, MODE],
                      [MODE, FLATTEN_UNIQUE, MODE], [COUNT, FLATTEN_UNIQUE, MODE], [COUNT_UNIQUE, FLATTEN_UNIQUE, MODE],
                      [MIN, COUNT, MODE], [MAX, COUNT, MODE], [MEAN, COUNT, MODE], [SUM, COUNT, MODE],
                      [MIN, COUNT_UNIQUE, MODE], [MAX, COUNT_UNIQUE, MODE],
                      [MEAN, COUNT_UNIQUE, MODE], [SUM, COUNT_UNIQUE, MODE]]
        package = [MAX, MIN, MEAN, SUM]
        for a in [MAX, MIN, MEAN, SUM]:
            for b in [COUNT, COUNT_UNIQUE]:
                solution2 += [[x, a, b] for x in package]
        types = [TYPE_NUMERIC if chain[0] != MODE else "nominal" for chain in solution2]
        solution3 = solution2
        solution2 = [(c, t) for c, t in zip(solution2, types)]
        self.assertListEqual(sorted(solution2), sorted(chains2))
        types = [TYPE_NUMERIC if chain[0] != MODE else "Horse" for chain in solution3]
        solution3 = [(c, t) for c, t in zip(solution3, types)]
        self.assertListEqual(sorted(solution3), sorted(chains3))

    def test_aggregator_chain_no_fresh(self):
        t = DecisionTree(allowed_aggregators=["sum"])
        chains1 = list(t.generate_possible_aggregator_chains(4, [], []))
        self.assertListEqual([([SUM, SUM, SUM, SUM], TYPE_NUMERIC)], chains1)
        t = DecisionTree(allowed_aggregators=[])
        chains2 = list(t.generate_possible_aggregator_chains(4, [], []))
        self.assertListEqual([([SUM, SUM, SUM, SUM], TYPE_NUMERIC)], chains2)
        t = DecisionTree(allowed_aggregators=["count", "sum", "mode", "flatten"])
        chains3 = list(t.generate_possible_aggregator_chains(4, [], []))
        self.assertListEqual([([SUM, SUM, SUM, SUM], TYPE_NUMERIC)], chains3)
