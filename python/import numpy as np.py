import numpy as np

# Simulación de una capa simple de RNN
class SimpleRNNLayer:
    def __init__(self, input_size, hidden_size):
        self.Wx = np.random.randn(hidden_size, input_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev):
        # Operaciones matriciales clave:
        z = np.dot(self.Wx, x) + np.dot(self.Wh, h_prev) + self.b
        h_next = np.tanh(z)
        return h_next

# Ejemplo de uso
input_size = 3
hidden_size = 2
rnn = SimpleRNNLayer(input_size, hidden_size)

# Secuencia de entrada (3 pasos de tiempo, cada uno de tamaño input_size)
seq = [np.array([[1], [2], [3]]),
       np.array([[4], [5], [6]]),
       np.array([[7], [8], [9]])]

h = np.zeros((hidden_size, 1))
for t in range(len(seq)):
    h = rnn.forward(seq[t], h)
    print(f"Paso {t+1}: h = \n{h}\n")