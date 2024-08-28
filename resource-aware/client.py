import json
import numpy as np
import requests
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Example Dense layer class for demonstration purposes
class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.rand(input_dim, output_dim)
        self.biases = np.random.rand(output_dim)
    
    def forward(self, x):
        return np.dot(x, self.weights) + self.biases

def measure_flops(layer):
    return 2 * layer.input_dim * layer.output_dim

def profile_network(layers, sample_input):
    layer_complexities = []
    layer_sizes = []

    for layer in layers:
        flops = measure_flops(layer)
        layer_complexities.append(flops)

        output = layer.forward(sample_input)
        layer_size = output.size * output.itemsize
        layer_sizes.append(layer_size)

        sample_input = output

    return layer_complexities, layer_sizes

def find_optimal_split(L, C_client, C_server, M_client, M_server, B_link, L_latency, layer_complexities, layer_sizes, alpha):
    min_cost = float('inf')
    best_split = 0

    for k in range(1, L):
        # Client side
        M_c = sum(layer_sizes[:k])
        P_c = sum(layer_complexities[:k])

        # Server side
        M_s = sum(layer_sizes[k:])
        P_s = sum(layer_complexities[k:])

        # JSON bandwidth usage
        S_transfer = layer_sizes[k-1] * alpha / B_link + L_latency

        if (M_c <= M_client) and (P_c <= C_client) and (M_s <= M_server) and (P_s <= C_server) and (S_transfer <= 100):
            cost = max(M_c / M_client, P_c / C_client, M_s / M_server, P_s / C_server, S_transfer)

            if cost < min_cost:
                min_cost = cost
                best_split = k

    return best_split

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(-1, 28*28) / 255.0
test_images = test_images.reshape(-1, 28*28) / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Sample input for profiling
sample_input = train_images[0].reshape(1, 28*28)

# Define the layers of the neural network
layers = [
    DenseLayer(28*28, 128),
    DenseLayer(128, 64),
    DenseLayer(64, 32),
    DenseLayer(32, 16),
    DenseLayer(16, 10)
]

L = len(layers)
C_client = 1e9  # FLOPs available on client
C_server = 10e9  # FLOPs available on server
M_client = 1024 * 1024 * 512  # 512MB available on client
M_server = 1024 * 1024 * 8192  # 8GB available on server
B_link = 100 * 1e6  # 100 Mbps
L_latency = 5  # 5 ms latency
alpha = 1.3  # JSON overhead factor

layer_complexities, layer_sizes = profile_network(layers, sample_input)

best_split = find_optimal_split(L, C_client, C_server, M_client, M_server, B_link, L_latency, layer_complexities, layer_sizes, alpha)

client_layers = layers[:best_split]
server_layers = layers[best_split:]

# Send the necessary data to the server for its layers
output = sample_input
for layer in client_layers:
    output = layer.forward(output)

json_data = {"activations": output.tolist()}
headers = {'Content-Type': 'application/json'}
response = requests.post("http://127.0.0.1:5002/process", json=json_data, headers=headers)

print(f"Optimal split: {best_split} layers on client, {L - best_split} layers on server")
print(f"Server response: {response.json()}")
