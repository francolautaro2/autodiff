from autodiff import Variable
import numpy as np

# Red neuronal simple con una capa completamente conectada
class SimpleNN:
    def __init__(self, x, weights, biases):
        self.x = x
        self.weights = weights
        self.biases = biases

    def forward(self):
        # Calcular z = x * weights + biases
        self.z = self.x.dot(self.weights) + self.biases
        return self.z

# Función de pérdida MSE
def mse_loss(prediction, target):
    return (prediction - target) ** 2

# Crear variables para la entrada, pesos, sesgos y objetivo (target)
x = Variable(np.array([[0.5, 1.0]]))  # Entrada de forma (1, 2)
weights = Variable(np.random.randn(2, 1))  # Pesos de forma (2, 1)
biases = Variable(np.array([[0.1]]))  # Sesgos de forma (1, 1)
target = Variable(np.array([[1.0]]))  # Valor esperado (target)

# Instanciar la red neuronal y realizar el forward
nn = SimpleNN(x, weights, biases)
prediction = nn.forward()

# Calcular la pérdida
loss = mse_loss(prediction, target)

# Realizar el backward para calcular gradientes
loss.backward()

# Mostrar los resultados
print(f"Predicción: {prediction.value}")
print(f"Pérdida: {loss.value}")
print(f"Gradiente con respecto a x: {x.grad}")
print(f"Gradiente con respecto a los pesos: {weights.grad}")
print(f"Gradiente con respecto a los sesgos: {biases.grad}")








