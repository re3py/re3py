import unittest
import my_utils


class UtilsTest(unittest.TestCase):
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

    def test_generate_canonic_sequences_of_new_variables(self):
        def generate_all(n):
            def helper(k):
                if k == 1:
                    for i in range(n):
                        yield [i]
                else:
                    for s in helper(k - 1):
                        for i in range(n):
                            yield s + [i]
            for t in helper(n):
                yield t

        def canonic_sequences_of_new_variables_brute(n):
            def canonic_form(seq):
                perm = {}
                t = 0
                for x in seq:
                    if x not in perm:
                        perm[x] = t
                        t += 1
                return [perm[x] for x in seq]
            for s in generate_all(n):
                if s == canonic_form(s):
                    yield [u + 1 for u in s]

        for size in range(1, 7):
            a = list(my_utils.canonic_sequences_of_new_variables(size))
            b = list(canonic_sequences_of_new_variables_brute(size))
            self.assertListEqual(a, b)

    def test_float_as_string(self):
        solutions = [" 1.0000e+101", "-1.0000e+101", " 0.0000e+00 ", " 1.2335e+00 ", " 2.1321e+05 ", " 2.3100e+02 "]
        floats = [10**101, -10**101, 0, 1.2334545, 213213.332, 231]
        for s, f in zip(solutions, floats):
            self.assertEqual(s, my_utils.float_as_string(f, 4))
