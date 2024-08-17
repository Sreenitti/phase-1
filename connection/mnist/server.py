import socket
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
import pickle

# Define the middle layers of the model (Server)
def create_server_model():
    input_layer = Input(shape=(64,))
    x = Dense(128, activation='relu')(input_layer)
    x = BatchNormalization()(x) 
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x) 
    model = Model(inputs=input_layer, outputs=x)
    return model

# Create the server model
server_model = create_server_model()

# Define an optimizer for updating the server model's weights
optimizer = tf.keras.optimizers.Adam()

# Storage for activations received from the client
received_activations = None

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '0.0.0.0'  # Bind to all available interfaces
port = 5005

# Bind the socket to a public host, and a well-known port
server_socket.bind((host, port))
server_socket.listen(5)

print("Server listening on port", port)

while True:
    client_socket, addr = server_socket.accept()
    print('Got connection from', addr)
    
    # Receive data from client
    data = client_socket.recv(4096)
    received_activations = np.array(pickle.loads(data))
    
    # Process activations through the server model
    server_activations = server_model.predict(received_activations)
    
    # Send processed activations back to the client
    client_socket.sendall(pickle.dumps(server_activations))
    
    # Receive gradients from client
    grads_data = client_socket.recv(4096)
    grads = np.array(pickle.loads(grads_data))
    
    # Convert gradients to TensorFlow tensor
    grads_tensor = tf.convert_to_tensor(grads, dtype=tf.float32)

    # Create a TensorFlow tensor for the received activations
    received_activations_tensor = tf.convert_to_tensor(received_activations, dtype=tf.float32)

    # Use the activations to calculate gradients with respect to the server model
    with tf.GradientTape() as tape:
        tape.watch(received_activations_tensor)
        outputs = server_model(received_activations_tensor)
    
    # Calculate gradients of the server model with respect to the received activations
    grads_input = tape.gradient(outputs, received_activations_tensor, output_gradients=grads_tensor)
    
    # Update the server model with the gradients
    with tf.GradientTape() as tape:
        tape.watch(server_model.trainable_variables)
        server_outputs = server_model(received_activations_tensor)
        loss = tf.reduce_mean(tf.losses.mean_squared_error(outputs, server_outputs))
    
    grads_model = tape.gradient(loss, server_model.trainable_variables)
    optimizer.apply_gradients(zip(grads_model, server_model.trainable_variables))
    
    # Send processed gradients back to the client
    client_socket.sendall(pickle.dumps(grads_input.numpy()))

    # Close the connection with the client
    client_socket.close()
