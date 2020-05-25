import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.animation import FuncAnimation
import torch

from Linear_regression import LinearRegression


def animate_learning(i, preds_arr, x_arr, line_):
    current_predicts = preds_arr[i]
    current_x = x_arr[i]
    line_.set_data(current_x, current_predicts)
    ax.legend()
    return line,


fig, ax = plt.subplots(figsize=(12, 8))


X, Y = datasets.make_regression(n_samples=1500, n_targets=1, n_features=1, noise=13)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# plt.scatter(X_train, Y_train, c='green', alpha=0.5, label="Train Data")
plt.scatter(X_test, Y_test, c='darkcyan', alpha=0.5, label="Test Data")


X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).float()

X_test = torch.from_numpy(X_test).float()
Y_test = torch.from_numpy(Y_test).float()


estimator = LinearRegression()
_, predictions_array, x_array, frames = estimator.fit(X_train, Y_train)

predicts = estimator.predict(X_test)

accuracy = 1-np.mean(abs(Y_test.detach().numpy() - predicts.detach().numpy()))/100
print(accuracy, '\n\n')


line, = ax.plot([], [], color="orangered", label='Predicted line')


plt.xlabel('X-axis\n\nAccuracy: {0}'.format(round(accuracy, 4)))
plt.ylabel('Y-axis')
plt.legend()

animation = FuncAnimation(
    fig,
    func=animate_learning,
    frames=frames,
    fargs=(predictions_array, x_array, line),
    interval=500,
    blit=True,
    repeat=True
)

# animation.save('Animated_Linear_regression_learning_test_only.gif', writer='imagemagick', fps=2)
plt.show()
