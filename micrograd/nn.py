from micrograd.engine import Value
import random

class Module:
    # Будет вызываться в обучении, для того чтобы обнулять градиенты перед их след расчетом
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        """

        :param nin: кол-во инпутов
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        # w @ x + b
        activ = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = activ.relu() if self.nonlin else activ
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout):
        '''

        :param nin: сколько параметров на вход каждого нейрона
        :param nout: сколько нейронов надо создать
        '''
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # Альтернативный вариант через list_comprehension
        # params = [p for neuron in self.neurons for p in neuron.parameters()]
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params