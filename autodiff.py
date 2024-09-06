import numpy as np

import numpy as np

class Variable:
    def __init__(self, value, _children=(), _op=''):
        self.value = np.array(value, dtype=float)
        self.grad = np.zeros_like(self.value)  # Gradiente acumulado
        self._backward = lambda: None  # Función para retropropagar
        self._prev = set(_children)  # Variables padre en el grafo
        self._op = _op  # Operación que creó esta variable (para depuración)

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value + other.value, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value - other.value, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward

        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        return other - self

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value * other.value, (self, other), '*')

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value / other.value, (self, other), '/')

        def _backward():
            self.grad += (1 / other.value) * out.grad
            other.grad -= (self.value / (other.value**2)) * out.grad
        out._backward = _backward

        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        return other / self

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Solo se soportan potencias enteras o flotantes."
        out = Variable(self.value ** power, (self,), f'**{power}')

        def _backward():
            self.grad += (power * self.value ** (power - 1)) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Variable(np.exp(self.value), (self,), 'exp')

        def _backward():
            self.grad += np.exp(self.value) * out.grad
        out._backward = _backward

        return out

    def sin(self):
        out = Variable(np.sin(self.value), (self,), 'sin')

        def _backward():
            self.grad += np.cos(self.value) * out.grad
        out._backward = _backward

        return out

    def cos(self):
        out = Variable(np.cos(self.value), (self,), 'cos')

        def _backward():
            self.grad += -np.sin(self.value) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.value))
        out = Variable(sig, (self,), 'sigmoid')

        def _backward():
            self.grad += sig * (1 - sig) * out.grad
        out._backward = _backward

        return out

    def dot(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(np.dot(self.value, other.value), (self, other), 'dot')

        def _backward():
            # La forma de out.grad es igual a la de out.value
            self.grad += np.dot(out.grad, other.value.T)
            other.grad += np.dot(self.value.T, out.grad)
        out._backward = _backward

        return out

    def mean(self):
        out = Variable(np.mean(self.value), (self,), 'mean')

        def _backward():
            self.grad += (1.0 / self.value.size) * np.ones_like(self.value) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # Construir orden topológico
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.value)

        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad})"
