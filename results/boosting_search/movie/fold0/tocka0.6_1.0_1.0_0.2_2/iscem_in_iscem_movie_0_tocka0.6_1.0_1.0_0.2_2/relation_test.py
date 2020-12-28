import unittest
from relation import *


class RelationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_parse_relation_arguments_standard(self):
        tasks = [("r1", "r1(x, y,   z)", ["x", "y", "z"]),
                 ("r2", "r2(1.2, Tom, 3.3  )", ["1.2", "Tom", "3.3"]),
                 ("r3", "r3(  3.3,  3.3  , 3.3 )", ["3.3", "3.3", "3.3"])
                 ]
        for relation, line, solution in tasks:
            result = parse_relation_arguments(line, relation)
            self.assertListEqual(solution, result)

    def test_parse_relation_arguments_standard_fail(self):
        tasks1 = [("r1", "r1(x, y,   z&)"),
                  ("r2", "r2(1.2, >Tom, 3.3  )")
                  ]
        tasks2 = [("r3", "r3(  3.3, 3 3.3  , 3.3 )"),
                  ("r4", "r4(1 1)")
                  ]
        for relation, line in tasks1:
            self.assertRaises(AttributeError, parse_relation_arguments, line, relation)
        for relation, line in tasks2:
            parse_relation_arguments(line, relation)  # i.e., not raises WrongValueException ...

    def test_parse_relation_arguments_multi_target_regression(self):
        tasks = [("r1", "r1(x, y,   [z])", ["x", "y", "[z]"]),
                 ("r2", "r2(1.2, Tom, [3.3  ])", ["1.2", "Tom", "[3.3  ]"]),
                 ("r3", "r3(  3.3,  3.3  , [3.3, 2.2, a] )", ["3.3", "3.3", "[3.3, 2.2, a]"])
                 ]
        for relation, line, solution in tasks:
            result = parse_relation_arguments(line, relation)
            self.assertListEqual(solution, result)

    def test_intelligent_parse_standard(self):
        tasks = [(["nominal", "nominal", "nominal"], ["x", "y", "z"], ["x", "y", "z"]),
                 (["numeric", "nominal", "numeric"], ["1.2", "Tom", "3.3"], [1.2, "Tom", 3.3]),
                 (["numeric", "numeric", "numeric"], ["3.3", "3.3", "3.3"], [3.3, 3.3, 3.3])
                 ]
        for types, string_values, solution in tasks:
            result = [Relation.intelligent_parse(t, v) for t, v in zip(types, string_values)]
            self.assertListEqual(solution, result)

    def test_intelligent_parse_multi_target(self):
        tasks = [(["nominal", "nominal", "multi_target[nominal1]"], ["x", "y", "[z]"], ["x", "y", ["z"]]),
                 (["nominal", "nominal", "multi_target[nominal1]"], ["x", "y", "[z, u]"], ["x", "y", ["z", "u"]]),
                 (["numeric", "nominal", "multi_target[numeric]"], ["1.2", "Tom", "[3.3 ]"], [1.2, "Tom", [3.3]]),
                 (["numeric", "numeric", "multi_target[numeric]"], ["3.3", "3.3", "[3.3, 2.2]"], [3.3, 3.3, [3.3, 2.2]])
                 ]
        for types, string_values, solution in tasks:
            result = [Relation.intelligent_parse(t, v) for t, v in zip(types, string_values)]
            self.assertListEqual(solution, result)
