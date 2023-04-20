from ..data.data_and_statistics import Datum
import pickle


class PredictiveModel:
    def fit(self, *args):
        raise NotImplementedError("This should be implemented by a subclass.")

    def predict(self, d: Datum):
        raise NotImplementedError("This should be implemented by a subclass.")

    def dump_to_text(self, file_name):
        raise NotImplementedError("This should be implemented by a subclass.")

    def dump_to_bin(self, file_name):
        pickle.dump(self, open(file_name, 'wb'))

    @staticmethod
    def load(file_name):
        return pickle.load(open(file_name, 'rb'))


# noinspection PyAbstractClass
class TreeEnsemble(PredictiveModel):
    def compute_ranking(self, ranking_type):
        raise NotImplementedError("This should be implemented by a subclass.")
