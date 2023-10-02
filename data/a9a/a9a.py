import numpy as np
import json

def create_client_data(X, y):
    # Assuming X and y have the same number of samples
    num_samples = len(X)
    
    # Create a list of client IDs
    clients = ["f_{:05d}".format(i) for i in range(num_samples)]
    
    # Initialize dictionaries for train_data and test_data
    train_data = {}
    test_data = {}
    
    # Populate train_data and test_data
    for client_id, x_values, label in zip(clients, X, y):
        client_data = {"x": x_values.tolist(), "y": int(label)}  # Assuming label is an integer
        train_data[client_id] = client_data
        test_data[client_id] = client_data
    
    # Create the final dictionary
    output_data = {"users": clients, "user_data": train_data}
    output_test = {"users": clients, "user_data": test_data}
    
    # Save the data as a .json file
    with open('client_data.json', 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    with open('client_data_test.json', 'w') as json_file:
        json.dump(output_test, json_file, indent=4)

    return clients, [], train_data, test_data

# Example usage:
X = np.random.rand(100, 14)  # Example X array with 100 samples and 14 features
y = np.random.randint(2, size=100)  # Example y array with binary labels

clients, groups, train_data, test_data = create_client_data(X, y)
