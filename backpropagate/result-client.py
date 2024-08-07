import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the initial layers of the CNN (Client) with varying layers
def create_initial_model(num_dense_layers=1):
    input_layer = Input(shape=(784,))
    x = BatchNormalization()(input_layer)
    for _ in range(num_dense_layers):
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

# Define the final layers of the CNN (Client) with varying layers
def create_final_model(num_dense_layers=1):
    input_layer = Input(shape=(64,))
    x = BatchNormalization()(input_layer)
    for _ in range(num_dense_layers):
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Experiment with different numbers of layers
num_dense_layers_list = [8, 9, 10]  # Number of dense layers to experiment with

# Store results
results = []

for num_dense_layers in num_dense_layers_list:
    print(f"Testing with {num_dense_layers} dense layers")

    # Create initial and final models
    initial_model = create_initial_model(num_dense_layers)
    final_model = create_final_model(num_dense_layers)

    # Compile the final model
    final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training loop
    for epoch in range(15):  # Training for 10 epochs
        # Forward pass through initial model
        initial_activations = initial_model.predict(x_train)

        # Send activations to server
        response = requests.post(
            "http://localhost:5001/process_activations",
            json={"activations": initial_activations.tolist()}
        )
        server_activations = np.array(response.json()["activations"])

        # Train the final model using true labels
        final_model.fit(server_activations, y_train, epochs=1, verbose=0)

    # Evaluate final model on training set
    train_predictions = final_model.predict(server_activations)
    train_accuracy = accuracy_score(y_train, np.argmax(train_predictions, axis=1))
    train_precision = precision_score(y_train, np.argmax(train_predictions, axis=1), average='weighted', zero_division=0)
    train_recall = recall_score(y_train, np.argmax(train_predictions, axis=1), average='weighted', zero_division=0)
    train_f1 = f1_score(y_train, np.argmax(train_predictions, axis=1), average='weighted', zero_division=0)

    print(f'Training Metrics with {num_dense_layers} dense layers:')
    print(f'Accuracy: {train_accuracy:.4f}')
    print(f'Precision: {train_precision:.4f}')
    print(f'Recall: {train_recall:.4f}')
    print(f'F1 Score: {train_f1:.4f}')

    # Evaluate final model on test set
    try:
        x_test_client = create_initial_model(num_dense_layers).predict(x_test)
        response = requests.post(
            "http://localhost:5001/process_activations",
            json={"activations": x_test_client.tolist()}
        )
        x_test_server = np.array(response.json()["activations"])
        x_test_final = final_model.predict(x_test_server)

        test_accuracy = accuracy_score(y_test, np.argmax(x_test_final, axis=1))
        test_precision = precision_score(y_test, np.argmax(x_test_final, axis=1), average='weighted', zero_division=0)
        test_recall = recall_score(y_test, np.argmax(x_test_final, axis=1), average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, np.argmax(x_test_final, axis=1), average='weighted', zero_division=0)

        results.append({
            "num_dense_layers": num_dense_layers,
            "train_accuracy": train_accuracy,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        })

        print(f'Test Metrics with {num_dense_layers} dense layers:')
        print(f'Accuracy: {test_accuracy:.4f}')
        print(f'Precision: {test_precision:.4f}')
        print(f'Recall: {test_recall:.4f}')
        print(f'F1 Score: {test_f1:.4f}')

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except (ValueError, KeyError) as e:
        print(f"Error decoding response: {e}")

# Print all results after testing all combinations
print("\nSummary of all tests:")
for result in results:
    print(result)
