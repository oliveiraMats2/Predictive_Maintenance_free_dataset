from neuralprophet import NeuralProphet
import pickle
class TrainNeuralProphet:
    def __init__(self, **configs) -> None:
        self.neural_prophet = NeuralProphet(**configs)

    def save(self, name:str ="neural_prophet_tmp.pkl") -> None:
        with open(name, "wb") as f:
            pickle.dump(self.neural_prophet, f)

    def load(self, name:str ="neural_prophet_tmp.pkl") -> None:
        with open(name, "rb") as f:
            self.neural_prophet = pickle.load(f)