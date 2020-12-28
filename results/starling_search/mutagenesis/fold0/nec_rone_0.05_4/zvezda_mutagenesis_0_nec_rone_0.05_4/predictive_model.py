from data import Datum
import pickle


class PredictiveModel:
    def build(self, *args):
        raise NotImplementedError("This should be implemented by a subclass.")

    def predict(self, d: Datum):
        raise NotImplementedError("This should be implemented by a subclass.")

    def print_model(self, file_name):
        raise NotImplementedError("This should be implemented by a subclass.")

    def save_model(self, file_name):
        pickle.dump(self, open(file_name, 'wb'))

    @staticmethod
    def load_model(file_name):
        return pickle.load(open(file_name, 'rb'))


# noinspection PyAbstractClass
class TreeEnsemble(PredictiveModel):
    def compute_ranking(self, ranking_type):
        raise NotImplementedError("This should be implemented by a subclass.")


# class ConstantModel(PredictiveModel):
#     def __init__(self):
#         self.prediction = None
#
#     def build(self, p):
#         self.prediction = p
#
#     def predict(self, d: Datum):
#         return self.prediction
#
#     def print_model(self, file_name):
#         f = open(file_name, "w")
#         s = "None" if self.prediction is None else str(self.prediction)
#         print("return {}".format(s), file=f)
#         f.close()
