import unittest
# from comparators import *
# from variables import *
from core import *
from relation import Relation
from aggregators import *


class SplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def generate_triplets(self, solutions, flat_solutions):
            for t, s, f_s in zip(self.tasks, solutions, flat_solutions):
                yield t, s, f_s

        cls.father = Relation("fatherOf", None, "test_data/fatherOf.txt", ["Person", "Person"])
        cls.age = Relation("age", None, "test_data/age.txt", ["Person", "numeric"])
        cls.friend = Relation("friend", None, "test_data/friend.txt", ["Person", "Person"])

        cls.solution_generator = generate_triplets

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_first_no_hits_simple(self):
        f1 = VariableVariable("Y1", "c", None)
        f2 = VariableVariable("Y2", "c", None)
        b1 = VariableVariable("X1", "Person", "Jim7")
        example = {"Y1": f1, "X1": b1, "Y2": f2}
        tests = [[self.age, ["Y1", "X1"], MIN], [self.age, ["Y2", "X1"], COUNT],  [self.age, ["Y1", "X1"], COUNT]]
        ths = [float("inf"), 0, 1]
        comps = [EQUAL, EQUAL, SMALLER]
        for t, th, c in zip(tests, ths, comps):
            s = BinarySplit([t], c, th, True, True)
            self.assertTrue(s.evaluate(example))

    def test_first_no_hits_long(self):
        f1 = VariableVariable("Y1", "c", None)
        f2 = VariableVariable("Y2", "c", None)
        b1 = VariableVariable("X1", "Person", "Jim7")
        example = {"Y1": f1, "X1": b1, "Y2": f2}
        tests = [[self.age, ["Y1", "X1"], MIN], [self.age, ["Y2", "X1"], COUNT],  [self.age, ["Y2", "X1"], COUNT]]
        ths = [float("inf"), 0, 1]
        comps = [EQUAL, EQUAL, SMALLER]
        for i, (t, th, c) in enumerate(zip(tests, ths, comps)):
            s = BinarySplit([tuple(x) for x in tests[i:]], c, th, True, True)
            self.assertTrue(s.evaluate(example))

    def test_then_no_hits_long(self):
        f1 = VariableVariable("Y1", "Person", None)
        f2 = VariableVariable("Y2", "c", None)
        b1 = VariableVariable("X1", "Person", "Jim7")
        example = {"Y1": f1, "X1": b1, "Y2": f2}
        tests = [(self.father, ["Y1", "X1"], MIN), (self.age, ["Y1", "Y2"], MIN)]
        s = BinarySplit(tests, EQUAL, float("inf"), True, True)
        self.assertTrue(s.evaluate(example))

    def test_ages_of_grandparents(self):
        f1 = VariableVariable("Y1", "Person", None)
        f2 = VariableVariable("Y2", "Person", None)
        f3 = VariableVariable("Y3", "c", None)
        b1 = VariableVariable("X1", "Person", "Jim1")
        example = {"Y1": f1, "X1": b1, "Y2": f2, "Y3": f3}
        # unique
        tests = [(self.father, ["Y1", "X1"], MEAN),
                 (self.father, ["Y2", "Y1"], FLATTEN_UNIQUE),
                 (self.age, ["Y2", "Y3"], MIN)]
        aggregated = 16
        s = BinarySplit(tests, SMALLER, aggregated + 10**-9, True, True)
        self.assertTrue(s.evaluate(example))
        self.assertTrue(s.evaluate(example))
        s = BinarySplit(tests, BIGGER, aggregated - 10**-9, True, True)
        self.assertTrue(s.evaluate(example))
        # not unique
        aggregated = 49/3
        tests = [(self.father, ["Y1", "X1"], MEAN),
                 (self.father, ["Y2", "Y1"], FLATTEN),
                 (self.age, ["Y2", "Y3"], MIN)]
        s = BinarySplit(tests, SMALLER, aggregated + 10**-9, True, True)
        self.assertTrue(s.evaluate(example))
        s = BinarySplit(tests, BIGGER, aggregated - 10**-9, True, True)
        self.assertTrue(s.evaluate(example))

    def test_ages_of_grandparents_more_generators(self):
        f1 = VariableVariable("Y1", "Person", None)
        f2 = VariableVariable("Y2", "Person", None)
        f3 = VariableVariable("Y3", "c", None)
        b1 = VariableVariable("X1", "Person", "Jim1")
        example = {"Y1": f1, "X1": b1, "Y2": f2, "Y3": f3}
        chains_relations = [(self.father, ["Y1", "X1"]),
                            (self.father, ["Y2", "Y1"]),
                            (self.age, ["Y2", "Y3"])]
        chains_aggregators = [(MEAN, FLATTEN_UNIQUE, MIN), (MEAN, FLATTEN, MIN)]
        eps = 10**-9
        s = BinarySplit([], None, None, True, True)
        answer = s.evaluate_all(example,
                                chains_relations,
                                chains_aggregators,
                                [SMALLER, SMALLER],
                                [16 + eps, 49/3 + eps])
        self.assertListEqual([True, True], answer)
        answer = s.evaluate_all(example,
                                chains_relations,
                                chains_aggregators,
                                [BIGGER, BIGGER],
                                [16 - eps, 49/3 - eps])
        self.assertListEqual([True, True], answer)

    def test_not_one_new_fresh(self):
        f1 = VariableVariable("Y1", "Person", None)
        f2 = VariableVariable("Y2", "Person", None)
        f3 = VariableVariable("Y3", "Person", None)
        b1 = VariableVariable("X1", "Person", "Jim1")
        b2 = VariableVariable("X2", "Person", "Bob2")
        example = {"Y1": f1, "X1": b1, "Y2": f2, "Y3": f3, "X2": b2}
        # 3-path
        tests = [(self.friend, ["X1", "Y1"], SUM),
                 (self.friend, ["Y1", "X2"], SUM)]
        s = BinarySplit(tests, EQUAL, 2, True, True)
        self.assertTrue(s.evaluate(example))
        # 4-cycle
        tests = [(self.friend, ["X1", "Y1"], SUM),
                 (self.friend, ["Y2", "X2"], SUM),
                 (self.friend, ["Y1", "Y2"], SUM)]
        s = BinarySplit(tests, EQUAL, 2, True, True)
        self.assertTrue(s.evaluate(example))
