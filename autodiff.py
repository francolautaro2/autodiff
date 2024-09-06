import numpy as np

class Variable:
    def __init__(self, value, derivative=0):
        self.value = value
        self.derivative = derivative
        self._grad = 0  # Gradient for backward accumulation
        self._backward = lambda: None  # Function to compute gradient
    
    def __add__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)
        result = Variable(self.value + other.value)
        result._backward = lambda: (self._grad + other._grad)
        result.derivative = self.derivative + other.derivative
        return result

    def __sub__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)
        result = Variable(self.value - other.value)
        result._backward = lambda: (self._grad - other._grad)
        result.derivative = self.derivative - other.derivative
        return result

    def __mul__(self, other):
        if isinstance(other, Variable):
            result = Variable(self.value * other.value)
            result._backward = lambda: (self._grad * other.value + self.value * other._grad)
            result.derivative = self.derivative * other.value + self.value * other.derivative
        else:
            result = Variable(self.value * other)
            result._backward = lambda: (self._grad * other)
            result.derivative = self.derivative * other
        return result

    def __truediv__(self, other):
        if isinstance(other, Variable):
            result = Variable(self.value / other.value)
            result._backward = lambda: ((self._grad * other.value - self.value * other._grad) / (other.value**2))
            result.derivative = (self.derivative * other.value - self.value * other.derivative) / (other.value**2)
        else:
            result = Variable(self.value / other)
            result._backward = lambda: (self._grad / other)
            result.derivative = self.derivative / other
        return result

    def __pow__(self, power):
        if isinstance(power, Variable):
            result = Variable(self.value**power.value)
            result._backward = lambda: (power.value * self.value**(power.value - 1) * self._grad)
            result.derivative = power.value * self.value**(power.value - 1) * self.derivative
        else:
            result = Variable(self.value**power)
            result._backward = lambda: (power * self.value**(power - 1) * self._grad)
            result.derivative = power * self.value**(power - 1) * self.derivative
        return result

    def sin(self):
        result = Variable(np.sin(self.value))
        result._backward = lambda: (np.cos(self.value) * self._grad)
        result.derivative = np.cos(self.value) * self.derivative
        return result

    def cos(self):
        result = Variable(np.cos(self.value))
        result._backward = lambda: (-np.sin(self.value) * self._grad)
        result.derivative = -np.sin(self.value) * self.derivative
        return result

    def exp(self):
        result = Variable(np.exp(self.value))
        result._backward = lambda: (np.exp(self.value) * self._grad)
        result.derivative = np.exp(self.value) * self.derivative
        return result

    def backward(self):
        self._grad = 1  # Set the gradient of the output variable to 1
        self._backward()  # Start backward pass

# Define the variable
x = Variable(1.0)

# Define the function f(x) = (x^2 + 2x) * sin(x)
x_squared = x ** 2
two_x = Variable(2) * x
intermediate = x_squared + two_x
y = intermediate.sin()

# Compute the gradients
y.backward()

# Print the results
print(f"Value of x: {x.value}")
print(f"Value of y: {y.value}")
print(f"Derivative of y with respect to x: {x._grad}")

