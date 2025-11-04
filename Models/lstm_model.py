import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import os

# Data Preprocessing


def read_and_combine_data(directory):
    # Limit to the first file
    all_files = sorted(glob.glob(os.path.join(directory, "*_tracks.csv")))[:1]
    data = []

    for file in all_files:
        df = pd.read_csv(file)
        data.append(df)

    combined_data = pd.concat(data, ignore_index=True)
    return combined_data


def preprocess_data(data, track_id):
    data = data[data['trackId'] == track_id]
    relevant_columns = ['xCenter', 'yCenter', 'xVelocity',
                        'yVelocity', 'xAcceleration', 'yAcceleration', 'heading']
    data = data[relevant_columns]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data)

    return data_normalized, scaler


def create_sliding_windows(data, window_size=30, prediction_size=10):
    X, y = [], []
    for i in range(len(data) - window_size - prediction_size):
        X.append(data[i:i+window_size])
        # Predicting only xCenter, yCenter
        y.append(data[i+window_size:i+window_size+prediction_size, :2])
    return np.array(X), np.array(y)


# Load and preprocess the data
# Replace with your data directory path
data_directory = "C:/Users/skspa/Downloads/inD-dataset-v1.1/data"
data = read_and_combine_data(data_directory)
first_track_id = data['trackId'].unique()[0]
data_normalized, scaler = preprocess_data(data, first_track_id)

# Split the data into training and testing sets
train_data, test_data = train_test_split(
    data_normalized, test_size=0.2, shuffle=False)

# Create sliding windows for the training and testing data
X_train, y_train = create_sliding_windows(train_data)
X_test, y_test = create_sliding_windows(test_data)

# LSTM Model with sequence output


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, prediction_length):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_length = prediction_length

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Get only the last 'prediction_length' time steps
        out = out[:, -self.prediction_length:, :]

        out = self.fc(out)
        return out


# Initialize the model, loss function, and optimizer
input_size = 7
hidden_size = 64
num_layers = 2
output_size = 2  # Predicting xCenter and yCenter
sequence_length = y_train.shape[1]  # This should match the prediction_size

model = LSTMModel(input_size, hidden_size, num_layers, output_size, sequence_length).to(
    torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'))
        y_batch = y_batch.to(torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'))

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation


def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    predictions, actuals = [], []

    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))
            y_batch = y_batch.to(torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    return avg_loss, predictions, actuals


test_loss, predictions, actuals = evaluate_model(model, test_loader)


def plot_comparisons(predictions, actuals):
    # Flatten the predictions and actuals arrays
    predictions_flat = predictions.reshape(-1, predictions.shape[-1])
    actuals_flat = actuals.reshape(-1, actuals.shape[-1])

    # Generate the correct number of time steps
    time_steps = np.arange(predictions_flat.shape[0])

    plt.figure(figsize=(12, 8))

    # Plot for positions
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, actuals_flat[:, 0], 'b-', label='Actual X Position')
    plt.plot(time_steps, predictions_flat[:, 0],
             'r--', label='Predicted X Position')
    plt.xlabel('Time Step')
    plt.ylabel('X Position')
    plt.title('X Position Comparison')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_steps, actuals_flat[:, 1], 'b-', label='Actual Y Position')
    plt.plot(time_steps, predictions_flat[:, 1],
             'r--', label='Predicted Y Position')
    plt.xlabel('Time Step')
    plt.ylabel('Y Position')
    plt.title('Y Position Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Call the visualization function
plot_comparisons(predictions, actuals)
