from neuralprophet import NeuralProphet
from neuralprophet import save, load
import pickle


class TrainNeuralProphet:
    __instance = None

    # def __new__(cls, *args, **kwargs):
    #     if not TrainNeuralProphet.__instance:
    #         TrainNeuralProphet.__instance = super(TrainNeuralProphet, cls).__new__(cls)
    #         return TrainNeuralProphet.__instance

    def __init__(self, **configs) -> None:
        self.neural_prophet = NeuralProphet(**configs)

    def save(self, name: str = "neural_prophet_tmp.np") -> None:
        save(self.neural_prophet, name)

    def load(self, name: str = "temperature_neural_prophet.np") -> None:
        self.neural_prophet = load(name)
