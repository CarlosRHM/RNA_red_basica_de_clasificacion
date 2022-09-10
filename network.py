# %load network.py

"""
Un modulo para implementar el algoritmo de aprendizaje stochastic
gradient descent para una red neuronal feedforward.
Los gradientes son calculados usando backpropagation
"""

#### Libraries
import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        "sizes" es el tamaño de la red, en este caso 3 neuronas, de
        784 entrada, 30 oculta y 10 salida.
        Los biases y los pesos de la red son inicializados aleatoriamente,
        usando una distribucion Gaussiana con media 0 y varianza 1.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Devuelve la salida de la red"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Entrena la red usando mini-batch stochastic gradient descent.
        El "training_data" es una lista de tuplas "(x,y)", representa
        las entradas de entrenamiento y las salidas deseadas.

        Si se proporciona ``test_data``, la red se evaluará con los
        datos de prueba después de cada época y se imprimirá el progreso
        parcial.
        """

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))
                plt.plot(epocas, costo)

    def update_mini_batch(self, mini_batch, eta):
        """
        Actualiza los pesos y biases de la red mediante la aplicación
        de descenso de gradiente mediante la retropropagación a un único
        mini-batch.
        El "mini_batch" es una lista de tuplas "(x,y)", y "eta" es la tasa
        de aprendizaje.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Devuelve una tupla "(nabla_b, nabla_w)" representa el gradiente
        de la funcion C_x.
        "nabla_b" y "nabla_w" son listas de matrices numpy capa por capa.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # Lista de las activaciones capa por capa
        zs = [] # Lista de vectores z capa por capa
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Devuelve el número de entradas de prueba para las que la
        red neuronal genera el resultado correcto.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Devuelve el vector de derivadas parciales
        \frac{\partial C_x}{\partial activacion de salida}
        """
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """La funcion sigmoide"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivada de la funcion sigmoide"""
    return sigmoid(z)*(1-sigmoid(z))
