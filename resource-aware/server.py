from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Example Dense layer class for demonstration purposes
class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.rand(input_dim, output_dim)
        self.biases = np.random.rand(output_dim)
    
    def forward(self, x):
        return np.dot(x, self.weights) + self.biases

# Initialize the server layers
server_layers = [
    DenseLayer(16, 10)  # Ensure this matches the output of the last client layer
]

@app.route('/process', methods=['POST'])
def process():
    try:
        # Ensure the incoming request is parsed as JSON
        data = request.get_json(force=True)

        # Extract the activations from the JSON
        activations = np.array(data["activations"])

        if activations is None:
            raise ValueError("No 'activations' found in the input data.")

        # Forward pass through the server layers
        output = activations
        for layer in server_layers:
            output = layer.forward(output)

        # Return the final output as a JSON response
        return jsonify({"output": output.tolist()})

    except Exception as e:
        # Handle exceptions and print error for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)
