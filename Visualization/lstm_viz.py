import matplotlib.pyplot as plt
import pytorch_lightning as torch

def predict_and_visualize(model, X, y, scaler, num_samples=5):
    model.eval()
    predictions = model(torch.tensor(X, dtype=torch.float32).to(torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))).cpu().detach().numpy()

    for i in range(num_samples):
        plt.figure()
        actual = scaler.inverse_transform(np.concatenate(
            [y[i], np.zeros((y.shape[1], X.shape[2] - 2))], axis=1))[:, :2]
        predicted = scaler.inverse_transform(np.concatenate(
            [predictions[i:i+1], np.zeros((1, X.shape[2] - 2))], axis=1))[:, :2]
        plt.plot(actual[:, 0], actual[:, 1], label="Actual")
        plt.plot(predicted[:, 0], predicted[:, 1], label="Predicted")
        plt.legend()
        plt.title(f'Sample {i+1}')
        plt.show()


# Example prediction and visualization
predict_and_visualize(model, X[-num_samples:], y[-num_samples:], scaler)
