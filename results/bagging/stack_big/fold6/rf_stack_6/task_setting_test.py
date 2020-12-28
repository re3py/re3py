import unittest
from task_settings import *
from relation import Relation


class SplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def constructor_caller(c, file_name):
            Settings(file_name)

        cls.caller = constructor_caller
        cls.no_target = "test_data/set_no_target.s"
        cls.empty = "test_data/set_empty.s"
        cls.wrong_agg = "test_data/set_wrong_agg.s"
        cls.ok = "test_data/set_ok.s"

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_no_target(self):
        self.assertRaises(MissingValueException, self.caller, self.no_target)

    def test_empty(self):
        self.assertRaises(MissingValueException, self.caller, self.empty)

    def test_wrong_agg(self):
        self.assertRaises(WrongValueException, self.caller, self.wrong_agg)

    def test_ok(self):
        s = Settings(self.ok)
        r, a, t, p = s.get_relations(), s.get_aggregates(), s.get_atom_tests(), s.get_tree_parameters()
        true_r = [Relation("x", set(), None, ["Person", "c"]),
                  Relation("xY", set(), None, ["PerSon"]),
                  Relation("hasPet", set(), None, ["A", "Bb"]),
                  Relation("isMale", set(), None, ["A"]),
                  Relation("baNaNa", set(), None, ["U", "Nana"]),
                  Relation("hairy", set(), None, ["Harry", "c", "c", "c", "c"]),
                  Relation("someRelation", set(), None, ["A", "B", "A", "C", "C", "B", "A"])]
        self.assertListEqual(r, true_r)
        self.assertListEqual(a, ["count", "sum", "flatten", "countUnique"])
        self.assertListEqual(t, [("hasPet", ("old", "new")),
                                 ("isMale", ("old",)),
                                 ("someRelation", ('old', 'old', 'new', 'old', 'old', 'old', 'old')),
                                 ("someRelation", ('old', 'c', 'new', 'old', 'c', 'old', 'c'))])
        self.assertDictEqual(p, {"numNodes": 21, "minInstancesNode": 1, "maxDepth": 42, "maxTestLength": 9001})

    def test_structured_atom_tests(self):
        s = Settings(self.ok)
        answer = {("hasPet", ("old", "new"), ("A", "Bb")): {"A": [[0], [], []], "Bb": [[], [1], []]},
                  ("isMale", ("old",), ("A",)): {"A": [[0], [], []]},
                  ("someRelation",
                   ('old', 'old', 'new', 'old', 'old', 'old', 'old'),
                   ("A", "B", "A", "C", "C", "B", "A")): {"A": [[0, 6], [2], []],
                                                          "B": [[1, 5], [], []],
                                                          "C": [[3, 4], [], []]},
                  ("someRelation",
                   ('old', 'c', 'new', 'old', 'c', 'old', 'c'),
                   ("A", "B", "A", "C", "C", "B", "A")): {"A": [[0], [2], [6]],
                                                          "B": [[5], [], [1]],
                                                          "C": [[3], [], [4]]}
                  }
        self.assertDictEqual(s.get_atom_tests_structured(), answer)
