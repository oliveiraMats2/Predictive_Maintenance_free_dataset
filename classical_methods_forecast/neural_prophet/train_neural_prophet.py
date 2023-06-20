from neuralprophet import NeuralProphet
from neuralprophet import save, load
import pickle
class TrainNeuralProphet:
    def __init__(self, **configs) -> None:
        self.neural_prophet = NeuralProphet(**configs)

    def save(self, name:str ="neural_prophet_tmp.pkl") -> None:
        save(self.neural_prophet, name)

    def load(self, name:str ="temperature_neural_prophet.pkl") -> None:
        self.neural_prophet = load(name)