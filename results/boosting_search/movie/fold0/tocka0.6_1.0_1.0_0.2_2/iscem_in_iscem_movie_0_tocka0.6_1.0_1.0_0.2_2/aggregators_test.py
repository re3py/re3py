import unittest
import aggregators
import numpy.testing as npt


class AggTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def generate_triplets(self, solutions, flat_solutions):
            for t, s, f_s in zip(self.tasks, solutions, flat_solutions):
                yield t, s, f_s

        cls.solution_generator = generate_triplets
        cls.tasks = [[], [1], [1, 2, 3], [[]], [[], []], [[1, 2, 3]], [[1, 2, 2], [1, 2, 8, 1], [1, 2, 2]]]
        cls.flatten_sol = [[], None, None, [], [], [1, 2, 3], [1, 2, 2, 1, 2, 8, 1, 1, 2, 2]]
        cls.flatten_sol_flat = [[], [1], [1, 2, 3], [[]], [[], []], [[1, 2, 3]], [[1, 2, 2], [1, 2, 8, 1], [1, 2, 2]]]
        cls.flatten_unique_sol = [[], None, None, [], [], [1, 2, 3], [1, 2, 8]]
        cls.flatten_unique_sol_flat = [[], [1], [1, 2, 3], [[]], [[], []],
                                       [[1, 2, 3]], [[1, 2, 2], [1, 2, 8, 1], [1, 2, 2]]]
        cls.count_sol = [[], None, None, [0], [0, 0], [3], [3, 4, 3]]
        cls.count_sol_flat = [0, 1, 3, 1, 2, 1, 3]
        cls.count_unique_sol = [[], None, None, [0], [0, 0], [3], [2, 3, 2]]
        cls.count_unique_sol_flat = [0, 1, 3, 1, 1, 1, 2]
        cls.min_sol = [[], None, None, [float("inf")], [float("inf"), float("inf")], [1], [1, 1, 1]]
        cls.min_sol_flat = [float("inf"), 1, 1, None, None, None, None]
        cls.max_sol = [[], None, None, [float("-inf")], [float("-inf"), float("-inf")], [3], [2, 8, 2]]
        cls.max_sol_flat = [float("-inf"), 1, 3, None, None, None, None]
        cls.mean_sol = [[], None, None, [float("inf")], [float("inf"), float("inf")], [2], [5/3, 3, 5/3]]
        cls.mean_sol_flat = [float("inf"), 1, 2, None, None, None, None]
        cls.sum_sol = [[], None, None, [0], [0, 0], [6], [5, 12, 5]]
        cls.sum_sol_flat = [0, 1, 6, None, None, None, None]
        cls.mode_sol = [[], None, None, [aggregators.MODE_OF_EMPTY_LIST],
                        [aggregators.MODE_OF_EMPTY_LIST, aggregators.MODE_OF_EMPTY_LIST],
                        [1], [2, 1, 2]]
        cls.mode_sol_flat = [aggregators.MODE_OF_EMPTY_LIST, 1, 1, [], [], [1, 2, 3], [1, 2, 2]]

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_flatten(self):
        for t, s, f_s in self.solution_generator(self.flatten_unique_sol, self.flatten_unique_sol_flat):
            if s is not None:
                self.assertListEqual(sorted(aggregators.FLATTEN_UNIQUE.aggregate(t)), sorted(s))
            self.assertListEqual(aggregators.FLATTEN_UNIQUE.aggregate_flat(t), f_s)

    def test_flatten_unique(self):
        for t, s, f_s in self.solution_generator(self.mode_sol, self.mode_sol_flat):
            if s is not None:
                a = aggregators.MODE.aggregate(t)
                self.assertListEqual(a, s)
            if f_s is not None:
                modified = t
                if type(t) == list and t and type(t[0]) == list:
                    modified = [tuple(w) for w in t]
                    f_s = tuple(f_s)
                a = aggregators.MODE.aggregate_flat(modified)
                self.assertEqual(a, f_s)

    def test_count(self):
        for t, s, f_s in self.solution_generator(self.count_sol, self.count_sol_flat):
            if s is not None:
                a = aggregators.COUNT.aggregate(t)
                self.assertListEqual(a, s)
            a = aggregators.COUNT.aggregate_flat(t)
            self.assertEqual(a, f_s)

    def test_count_unique(self):
        for t, s, f_s in self.solution_generator(self.count_unique_sol, self.count_unique_sol_flat):
            if s is not None:
                a = aggregators.COUNT_UNIQUE.aggregate(t)
                self.assertListEqual(a, s)
            a = aggregators.COUNT_UNIQUE.aggregate_flat([tuple(w) if type(w) == list else w for w in t])
            self.assertEqual(a, f_s)

    def test_min(self):
        for t, s, f_s in self.solution_generator(self.min_sol, self.min_sol_flat):
            if s is not None:
                a = aggregators.MIN.aggregate(t)
                self.assertListEqual(a, s)
            if f_s is not None:
                a = aggregators.MIN.aggregate_flat(t)
                self.assertEqual(a, f_s)

    def test_max(self):
        for t, s, f_s in self.solution_generator(self.max_sol, self.max_sol_flat):
            if s is not None:
                a = aggregators.MAX.aggregate(t)
                self.assertListEqual(a, s)
            if f_s is not None:
                a = aggregators.MAX.aggregate_flat(t)
                self.assertEqual(a, f_s)

    def test_mean(self):
        for t, s, f_s in self.solution_generator(self.mean_sol, self.mean_sol_flat):
            if s is not None:
                a = aggregators.MEAN.aggregate(t)
                npt.assert_almost_equal(a, s, decimal=10)
            if f_s is not None:
                a = aggregators.MEAN.aggregate_flat(t)
                self.assertAlmostEqual(a, f_s)

    def test_sum(self):
        for t, s, f_s in self.solution_generator(self.sum_sol, self.sum_sol_flat):
            if s is not None:
                a = aggregators.SUM.aggregate(t)
                self.assertListEqual(a, s)
            if f_s is not None:
                a = aggregators.SUM.aggregate_flat(t)
                self.assertEqual(a, f_s)

    def test_mode(self):
        for t, s, f_s in self.solution_generator(self.mode_sol, self.mode_sol_flat):
            if s is not None:
                a = aggregators.MODE.aggregate(t)
                self.assertListEqual(a, s)
            if f_s is not None:
                modified = t
                if type(t) == list and t and type(t[0]) == list:
                    modified = [tuple(w) for w in t]
                    f_s = tuple(f_s)
                a = aggregators.MODE.aggregate_flat(modified)
                self.assertEqual(a, f_s)


