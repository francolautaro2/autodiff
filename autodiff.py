import numpy as np

class Variable:
    def __init__(self, value, derivative=0):
        self.value = value
        self.derivative = derivative

    def __add__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)
        return Variable(self.value + other.value, self.derivative + other.derivative)

    def __sub__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)
        return Variable(self.value - other.value, self.derivative - other.derivative)

    
    def __mul__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)
        return Variable(self.value * other.value, self.derivative * other.value + self.value * other.derivative)

    
    def __truediv__(self, other):
        if not isinstance(other, Variable):
            other = Variable(other)
        return Variable(self.value / other.value, (self.derivative * other.value - self.value * other.derivative) / (other.value**2))

    
    def __pow__(self, power):
        return Variable(self.value**power, power * self.value**(power - 1) * self.derivative)

    
    def sin(self):
        return Variable(np.sin(self.value), np.cos(self.value) * self.derivative)

    def cos(self):
        return Variable(np.cos(self.value), -np.sin(self.value) * self.derivative)

    def exp(self):
        return Variable(np.exp(self.value), np.exp(self.value) * self.derivative)


x = Variable(2, 1)
y = x**2 + x.sin()
z = y**2 + y.sin()

print(f"Value of y: {y.value}")
print(f"Derivative of y with respect to x: {y.derivative}")
print(f"Value of z: {z.value}")

