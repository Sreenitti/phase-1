import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
import requests
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def create_final_model():
    input_layer = Input(shape=(64,))
    x = Dense(64, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Split the training data between two clients
split_index = len(x_train) // 2
client1_x_train = x_train[:split_index]
client1_y_train = y_train[:split_index]
print("Client 1 x_train shape : ", client1_x_train.shape)

# Create initial and final models
initial_model = create_initial_model()
final_model = create_final_model()

# Compile the models
initial_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mean_squared_error')
final_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training loop
for epoch in range(1):
    logging.info('Epoch %d', epoch + 1)
    
    logging.info('Performing forward pass through initial model...')
    initial_activations = initial_model(client1_x_train, training=False)
    logging.info('Initial model forward pass complete.')
    
    logging.info('Sending activations to server...')
    print("Client initial activations shape : ", initial_activations.shape)
    response = requests.post("http://localhost:5002/process_activations", json={"client_id": 1, "activations": initial_activations.numpy().tolist()})
    
    while 'status' in response.json() and response.json()['status'] == 'waiting for other client':
        logging.info('Server is waiting for other client. Retrying in 1 second...')
        time.sleep(1)
        response = requests.post("http://localhost:5002/process_activations", json={"client_id": 1, "activations": initial_activations.numpy().tolist()})
    
    if 'activations' in response.json():
        server_activations = np.array(response.json()["activations"])
        logging.info('Server responded with processed activations.')
    else:
        logging.error('Unexpected server response: %s', response.json())
        continue

    logging.info('Training the final model...')
    final_model.fit(server_activations, client1_y_train[:len(server_activations)], epochs=1, verbose=1)
    logging.info('Final model training step complete.')
    
    with tf.GradientTape() as tape:
        predictions = final_model(server_activations, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(client1_y_train[:len(server_activations)], predictions)
    
    gradients = tape.gradient(loss, final_model.trainable_variables)
    final_model.optimizer.apply_gradients(zip(gradients, final_model.trainable_variables))
    logging.info('Final model gradients applied.')

    logging.info('Computing gradients of server activations...')
    with tf.GradientTape() as tape:
        tape.watch(initial_activations)
        predictions = final_model(initial_activations, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(client1_y_train[:len(initial_activations)], predictions)
    
    grads_initial_activations = tape.gradient(loss, initial_activations)
    
    if grads_initial_activations is None:
        logging.error('Error: Gradients for server activations are None')
    else:
        logging.info('Server activations gradients computed.')
        grads_initial_activations = tf.convert_to_tensor(grads_initial_activations, dtype=tf.float32)
        
        logging.info('Sending gradients of server activations to server...')
        print("Client gradients shape : ", grads_initial_activations.shape)
        response = requests.post("http://localhost:5002/backpropagate", json={"client_id": 1, "grads": grads_initial_activations.numpy().tolist()})

        while 'status' in response.json() and response.json()['status'] == 'waiting for all gradients':
            logging.info('Server is waiting for other client. Retrying in 1 second...')
            time.sleep(1)
            response = requests.post("http://localhost:5002/backpropagate", json={"client_id": 1, "grads": grads_initial_activations.numpy().tolist()})
        
        logging.info('Server responded with gradients processing.')

        try:
            grads_initial_activations = np.array(response.json()["grads"])
            logging.info('Received gradients from server for initial model.')
        except KeyError:
            logging.error("Response does not contain 'grads' key")
            logging.error("Full response: %s", response.json())
            continue
        
        logging.info('Performing backpropagation on initial model...')
        with tf.GradientTape() as tape:
            initial_outputs = initial_model(client1_x_train, training=True)
        
        grads_initial_activations = tf.convert_to_tensor(grads_initial_activations, dtype=tf.float32)
        
        grads = tape.gradient(initial_outputs, initial_model.trainable_variables, output_gradients=grads_initial_activations)
        if initial_model.optimizer is not None:
            initial_model.optimizer.apply_gradients(zip(grads, initial_model.trainable_variables))
            logging.info('Initial model gradients applied.')
        else:
            logging.error("Error: Initial model optimizer is None")

    logging.info('Evaluating on training data...')
    train_loss, train_accuracy = final_model.evaluate(server_activations, client1_y_train[:len(server_activations)], verbose=0)
    logging.info('Epoch: %d, Training Accuracy: %.4f', epoch+1, train_accuracy)

# Evaluate final model on test set
try:
    logging.info('Evaluating final model on test set...')
    x_test_client = initial_model.predict(x_test)
    logging.info("TEST - Sending client activations")
    print("Client activations shape : ", x_test_client.shape)
    response = requests.post("http://localhost:5002/process_activations", json={"client_id": 1, "activations": x_test_client.tolist()})
    
    while 'status' in response.json() and response.json()['status'] == 'waiting for other client':
        logging.info('Server is waiting for other client. Retrying in 1 second...')
        time.sleep(1)
        response = requests.post("http://localhost:5002/process_activations", json={"client_id": 1, "activations": x_test_client.tolist()})
    
    if 'activations' in response.json():
        x_test_server = np.array(response.json()["activations"])
        x_test_final = final_model.predict(x_test_server)
        test_accuracy = np.mean(np.argmax(x_test_final, axis=1) == y_test)
        logging.info('Test Accuracy: %.4f', test_accuracy)
    else:
        logging.error('Unexpected server response: %s', response.json())
except requests.exceptions.RequestException as e:
    logging.error("Request failed: %s", e)
except (ValueError, KeyError) as e:
    logging.error("Error decoding response: %s", e)
