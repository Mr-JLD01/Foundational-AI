import numpy as np
import random
import math
from .Weights import Weights

class GlorotUniform(Weights):
	@classmethod
	def generate(cls, in_neurons: int, out_neurons: int) -> np.ndarray:
		bound = math.sqrt(6 / (in_neurons + out_neurons))
		weights = np.zeros((in_neurons, out_neurons), dtype=float)

		for row in range(in_neurons):
			for col in range(out_neurons):
				weights[row][col] = random.uniform(-bound, bound)

		return weights