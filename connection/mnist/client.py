import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
import socket
import pickle
import logging

# Configure logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the initial layers of the model (Client)
def create_initial_model():
    input_layer = Input(shape=(784,))
    x = Dense(512, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images and normalize pixel values
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Create the initial model
initial_model = create_initial_model()

# Compile the initial model
initial_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')

train_acc = []
server_ip = '192.168.0.101'  # Change this to the server's IP address
port = 5005

# Training loop
for epoch in range(100):  # Adjust epochs as needed
    logging.info('Epoch %d', epoch + 1)
    
    # Forward pass through initial model
    logging.info('Performing forward pass through initial model...')
    initial_activations = initial_model(x_train, training=False)
    logging.info('Initial model forward pass complete.')
    
    # Send activations to server using socket
    logging.info('Sending activations to server...')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((server_ip, port))
        data = pickle.dumps(initial_activations.numpy())
        client_socket.sendall(data)
        
        # Receive processed activations from server
        server_activations_data = client_socket.recv(4096)
        server_activations = np.array(pickle.loads(server_activations_data))
        logging.info('Server responded with processed activations.')

    # Convert server activations to a TensorFlow tensor
    server_activations_tensor = tf.convert_to_tensor(server_activations, dtype=tf.float32)

    # Receive processed activations from server and calculate loss
    logging.info('Calculating loss...')
    with tf.GradientTape() as tape:
        tape.watch(server_activations_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_train, server_activations_tensor)
    
    grads = tape.gradient(loss, server_activations_tensor)
    
    # Send gradients back to server using socket
    logging.info('Sending gradients back to server...')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((server_ip, port))
        data = pickle.dumps(grads.numpy())
        client_socket.sendall(data)
        
        # Receive gradients processed by server
        grads_received_data = client_socket.recv(4096)
        grads_received = np.array(pickle.loads(grads_received_data))
        logging.info('Server responded with gradients processing.')

    # Perform backpropagation on initial model
    logging.info('Performing backpropagation on initial model...')
    grads_received_tensor = tf.convert_to_tensor(grads_received, dtype=tf.float32)
    with tf.GradientTape() as tape:
        initial_outputs = initial_model(x_train, training=True)
    
    grads_initial_model = tape.gradient(initial_outputs, initial_model.trainable_variables, output_gradients=grads_received_tensor)
    initial_model.optimizer.apply_gradients(zip(grads_initial_model, initial_model.trainable_variables))
    logging.info('Initial model gradients applied.')

    # Compute accuracy
    logging.info('Evaluating on training data...')
    predictions = tf.argmax(server_activations, axis=1)
    train_accuracy = np.mean(predictions.numpy() == y_train)
    logging.info('Epoch: %d, Training Accuracy: %.4f', epoch + 1, train_accuracy)
    train_acc.append(train_accuracy)

# Evaluate final model on training set
logging.info('Evaluating final model on training set...')
train_predictions = initial_model.predict(x_train)
train_accuracy = np.mean(np.argmax(train_predictions, axis=1) == y_train)
logging.info('Training Accuracy: %.4f', train_accuracy)
print("TRAINING ACCURACY ARRAY : ", train_acc)

# Evaluate final model on test set
try:
    logging.info('Evaluating final model on test set...')
    x_test_client = initial_model.predict(x_test)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((server_ip, port))
        data = pickle.dumps(x_test_client)
        client_socket.sendall(data)
        
        # Receive processed test activations from server
        x_test_server_data = client_socket.recv(4096)
        x_test_server = np.array(pickle.loads(x_test_server_data))
    
    x_test_final = tf.convert_to_tensor(x_test_server, dtype=tf.float32)
    
    test_accuracy = np.mean(np.argmax(x_test_final, axis=1) == y_test)
    logging.info('Test Accuracy: %.4f', test_accuracy)
except Exception as e:
    logging.error("An error occurred: %s", e)
