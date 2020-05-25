import torch


class LinearRegression:
    def __init__(self):
        self.W = None
        self.b = None
        self.epsilon = 1e-1

    def get_loss(self, preds, y):
        """
            @param preds: предсказания модели
            @param y: истиные значения
            @return mse: значение MSE на переданных данных
        """

        mse = torch.nn.functional.mse_loss(preds, y)
        return mse

    def init_weights(self, input_size, output_size):
        """
            Инициализирует параметры модели
            W - матрица размерности (input_size, output_size)
            инициализируется рандомными числами из
            uniform распределения (torch.rand())
            b - вектор размерности (1, output_size)
            инициализируется нулями
        """
        self.W = torch.rand(1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def fit(self, X, y, num_epochs=20000, lr=0.001):
        """
            Обучение модели линейной регрессии методом градиентного спуска
            @param X: размерности (num_samples, input_shape)
            @param y: размерности (num_samples, output_shape)
            @param num_epochs: количество итераций градиентного спуска
            @param lr: шаг градиентного спуска
            @return metrics: вектор значений MSE на каждом шаге градиентного
            спуска.
        """
        predictions = []
        x_array = []
        self.init_weights(X.shape[0], y.shape[0])
        metrics = []
        frames = 0
        for _ in range(num_epochs):
            preds = self.predict(X)
            loss = self.get_loss(preds, y)
            loss.backward()

            with torch.no_grad():
                self.W -= lr * self.W.grad
                self.b -= lr * self.b.grad

            if _ % 100 == 0:
                frames += 1
                predictions.append(preds.detach().numpy())
                x_array.append(X.detach().numpy())

            metrics.append(self.get_loss(preds, y).data)

            if self.W.grad.norm() <= self.epsilon and self.b.grad.norm() <= self.epsilon:
                 break

            self.W.grad.data.zero_()
            self.b.grad.data.zero_()

        print(_)
        return metrics, predictions, x_array, frames

    def predict(self, X):
        """

        """
        y_predicted = torch.matmul(X, self.W) + self.b
        return y_predicted
