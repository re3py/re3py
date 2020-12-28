class Evaluator:
    def __init__(self):
        self.examples = 0
        self.measure_value = None

    def get_measure_value(self):
        return self.measure_value

    def evaluate(self):
        raise NotImplementedError("This should be implemented by a subclass.")

    def add_one(self, true_value, prediction, weight=1.0):
        self.examples += weight

    def add_many(self, true_values, predictions, weights=None):
        if weights is None:
            weights = (1 for _ in range(len(true_values)))
        for t, p, w in zip(true_values, predictions, weights):
            self.add_one(t, p, w)


class RegressionEvaluator(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self):
        raise NotImplementedError("This should be implemented by a subclass.")

    def add_one(self, true_value, prediction, weight=1.0):
        super().add_one(true_value, prediction, weight)


class MeanSquaredError(RegressionEvaluator):
    def __init__(self):
        super().__init__()
        self.sum_of_squared_differences = 0

    def add_one(self, true_value, prediction, weight=1.0):
        super().add_one(true_value, prediction, weight)
        self.sum_of_squared_differences += weight * (true_value -
                                                     prediction)**2

    def evaluate(self):
        self.measure_value = self.sum_of_squared_differences / self.examples


class RelativeMeanSquaredError(MeanSquaredError):
    def __init__(self):
        super().__init__()
        self.sum_of_true_values = 0
        self.sum_of_squared_true_values = 0

    def add_one(self, true_value, prediction, weight=1.0):
        super().add_one(true_value, prediction, weight)
        u = weight * true_value
        self.sum_of_true_values += weight * true_value
        self.sum_of_squared_true_values += u * true_value

    def evaluate(self):
        super().evaluate()
        normalization = self.sum_of_squared_true_values - self.examples * self.sum_of_true_values**2
        self.measure_value /= normalization


class MeanAbsoluteError(RegressionEvaluator):
    def __init__(self):
        super().__init__()
        self.sum_of_differences = 0

    def add_one(self, true_value, prediction, weight=1.0):
        super().add_one(true_value, prediction, weight)
        self.sum_of_differences += weight * abs(true_value - prediction)

    def evaluate(self):
        self.measure_value = self.sum_of_differences / self.examples


class RelativeMeanAbsoluteError(MeanAbsoluteError):
    def __init__(self):
        super().__init__()
        # incremental computation not possible
        self.true_values = []
        self.weights = []

    def add_one(self, true_value, prediction, weight=1.0):
        super().add_one(true_value, prediction, weight)
        self.true_values.append(true_value)
        self.weights.append(weight)

    def evaluate(self):
        super().evaluate()
        average = sum(self.true_values) / self.examples
        normalization = sum(
            w * abs(t - average)
            for w, t in zip(self.weights, self.true_values)) / self.examples
        self.measure_value /= normalization


class RootMeanSquaredError(MeanSquaredError):
    def __init__(self):
        super().__init__()

    def evaluate(self):
        super().evaluate()
        self.measure_value **= 0.5


class RelativeRootMeanSquaredError(RelativeMeanSquaredError):
    def __init__(self):
        super().__init__()

    def evaluate(self):
        super().evaluate()
        self.measure_value **= 0.5


class ClassificationEvaluator(Evaluator):
    def __init__(self,
                 class_values,
                 positive_class=None,
                 confusion_matrix=None):
        """
        :param class_values: an iterable of possible class values
        :param positive_class: the positive class value
        :return:
        """
        super().__init__()
        self.class_indices = {}
        for v in class_values:
            if v not in self.class_indices:
                self.class_indices[v] = len(self.class_indices)
        c = len(self.class_indices)
        self.positive_class = positive_class
        # When we will be interested in area under the curve errors, we will conf. matrices to a subclass
        self.confusion_matrix = [[0 for _ in range(c)] for _ in range(c)
                                 ]  # matrix[true class][predicted class]
        if confusion_matrix is not None:
            # safer than self.confusion_matrix = confusion_matrix
            true_values = []
            predicted_values = []
            index_to_class = {v: k for k, v in self.class_indices.items()}
            for i in range(c):
                true_value = index_to_class[i]
                for j in range(c):
                    predicted_value = index_to_class[j]
                    n = confusion_matrix[i][j]
                    true_values += [true_value for _ in range(n)]
                    predicted_values += [predicted_value for _ in range(n)]
            self.add_many(true_values, predicted_values)

    def __str__(self):
        value = str(
            self.measure_value) if self.measure_value is not None else "None"
        return "{}: {}".format(self.__class__.__name__, value)

    def __add__(self, other):
        """
        Adds confusion matrices and computes the measure from the new matrix. Returns a new object.
        :param other:
        :return:
        """
        if sorted(self.class_indices.keys()) != sorted(
                other.class_indices.keys()):
            print(self.class_indices, other.class_indices)
            exit(-1)
        if self.positive_class != other.positive_class:
            print(self.positive_class != other.positive_class)
            exit(-2)
        m = self.__class__(sorted(self.class_indices.keys()),
                           self.positive_class)
        # add matrices
        for term in [self, other]:
            for true_c in term.class_indices:
                i_true_m = m.class_indices[true_c]
                i_true_o = term.class_indices[true_c]
                for predicted_c in term.class_indices:
                    i_predicted_m = m.class_indices[predicted_c]
                    i_predicted_o = term.class_indices[predicted_c]
                    m.confusion_matrix[i_true_m][
                        i_predicted_m] += term.confusion_matrix[i_true_o][
                            i_predicted_o]
        # add examples
        m.examples = self.examples + other.examples
        m.evaluate()
        return m

    def add_one(self, true_value, predicted_value, weight=1.0):
        """
        :param true_value
        :param predicted_value either actual predictions, such as 'pos', 'healthy' etc.
        or probabilistic scores, such as 0.213, 0.0 or 1.0.
        :param weight
        :return:
        """
        super().add_one(true_value, predicted_value, weight)
        i_true = self.class_indices[true_value]
        i_predicted = self.class_indices[predicted_value]
        self.confusion_matrix[i_true][i_predicted] += weight

    def evaluate(self):
        raise NotImplementedError("This should be implemented by a subclass.")


class Accuracy(ClassificationEvaluator):
    def __init__(self,
                 class_values,
                 positive_class=None,
                 confusion_matrix=None):
        super().__init__(class_values, positive_class, confusion_matrix)

    def evaluate(self):
        d = sum(x[i] for i, x in enumerate(self.confusion_matrix))
        self.measure_value = d / self.examples


class Precision(ClassificationEvaluator):
    def __init__(self,
                 class_values,
                 positive_class=None,
                 confusion_matrix=None):
        super().__init__(class_values, positive_class, confusion_matrix)

    def evaluate(self):
        i_tp = self.class_indices[self.positive_class]
        true_positives = self.confusion_matrix[i_tp][i_tp]
        predicted_positives = sum(x[i_tp] for x in self.confusion_matrix)
        self.measure_value = true_positives / predicted_positives


class Recall(ClassificationEvaluator):
    def __init__(self,
                 class_values,
                 positive_class=None,
                 confusion_matrix=None):
        super().__init__(class_values, positive_class, confusion_matrix)

    def evaluate(self):
        i_tp = self.class_indices[self.positive_class]
        true_positives = self.confusion_matrix[i_tp][i_tp]
        positives = sum(self.confusion_matrix[i_tp])
        self.measure_value = true_positives / positives


class F1(ClassificationEvaluator):
    def __init__(self,
                 class_values,
                 positive_class=None,
                 confusion_matrix=None):
        super().__init__(class_values, positive_class, confusion_matrix)

    def evaluate(self):
        i_tp = self.class_indices[self.positive_class]
        true_positives = self.confusion_matrix[i_tp][i_tp]
        predicted_positives = sum(x[i_tp] for x in self.confusion_matrix)
        positives = sum(self.confusion_matrix[i_tp])
        self.measure_value = 2 * true_positives / (predicted_positives +
                                                   positives)
