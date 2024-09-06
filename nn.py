from autodiff import Variable
import numpy as np


class SimpleNN:
    def __init__(self, x, weights, biases):
        self.x = x
        self.weights = weights
        self.biases = biases

    def forward(self):
        # z = x * weights + biases
        self.z = self.x.dot(self.weights) + self.biases
        return self.z


def mse_loss(prediction, target):
    return (prediction - target) ** 2


x = Variable(np.array([[0.5, 1.0]]))  
weights = Variable(np.random.randn(2, 1))  
biases = Variable(np.array([[0.1]]))  
target = Variable(np.array([[1.0]]))  

nn = SimpleNN(x, weights, biases)
prediction = nn.forward()

loss = mse_loss(prediction, target)

loss.backward()

print(f"Predicción: {prediction.value}")
print(f"Pérdida: {loss.value}")
print(f"Gradiente con respecto a x: {x.grad}")
print(f"Gradiente con respecto a los pesos: {weights.grad}")
print(f"Gradiente con respecto a los sesgos: {biases.grad}")








