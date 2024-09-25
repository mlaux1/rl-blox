import numpy as np
from torchgen.executorch.api.et_cpp import return_type
from tqdm import trange
import torch
from torch.nn import functional as F
from torch.autograd import Variable
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
from sklearn.utils import check_random_state


class ProbRegressionNN:
    def __init__(self, n_inputs, n_outputs,
                 random_state=None, verbose=0, n_hidden_units=300,
                 n_features=50):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.model = ProbRegression(self.n_inputs, self.n_outputs, n_hidden_units, n_features)

    def fit(self, X, Y, context_transform=None, batch_size=16, n_epochs=100):
        X = Tensor(X)
        Y = Tensor(Y)

        self.model = ProbRegression(self.n_inputs, self.n_outputs)
        opt = torch.optim.Adam(self.model.parameters())

        n_samples = X.shape[0]
        assert n_samples == Y.shape[0]
        indices = np.arange(n_samples)

        if self.verbose:
            pbar = trange(n_epochs)
        else:
            pbar = range(n_epochs)

        for _ in pbar:
            for i in range(n_samples // batch_size):
                batch_indices = self.random_state.choice(indices, batch_size, False)
                batch_X = X[batch_indices]
                batch_Y = Y[batch_indices]
                Y_pred, Y_log_std = self.model(batch_X)
                l = heteroscedastic_aleatoric_uncertainty_loss(
                    Y_pred, Y_log_std, batch_Y)
                opt.zero_grad()
                l.backward()
                opt.step()
                loss_value = l.item()
                if self.verbose:
                    pbar.set_description("Error: %.3f; Epochs" % loss_value)

        return self

    def features(self, X):
        X = Tensor(X)
        features = self.model.compute_features(X)
        return features.detach().cpu().numpy()

    def predict(self, X, return_std=False):
        X = Tensor(X)
        Y_pred, Y_log_std = self.model(X)
        Y_std = torch.exp(Y_log_std.detach())
        if return_std:
            return Y_pred.detach().cpu().numpy(), Y_std.cpu().numpy()
        else:
            return Y_pred.detach().cpu().numpy()

    def sample(self, X):
        Y_pred, Y_std = self.predict(X)
        eps = self.random_state.randn(*Y_pred.shape)
        return Y_pred + Y_std * eps

    def score(self, X, Y):
        X = Tensor(X)
        Y = Tensor(Y)
        Y_pred, Y_log_std = self.model(X)

        loss = heteroscedastic_aleatoric_uncertainty_loss(
            Y_pred, Y_log_std, Y)
        return loss.item()


class ProbRegression(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden_units=300, n_features=10):
        super(ProbRegression, self).__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_hidden_units),
            torch.nn.ReLU(),
        )
        self.features = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_units, n_hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_units, n_features),
        )
        self.mean = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(n_features, n_outputs),
        )
        self.log_std = torch.nn.Linear(n_features, n_outputs)
        if torch.cuda.is_available():
            self.cuda()

    def compute_features(self, x):
        return self.features(self.body(x))

    def forward(self, x):
        x = self.body(x)
        return self.mean(self.features(x)), self.log_std(self.features(x))


def heteroscedastic_aleatoric_uncertainty_loss(
        Y_pred, Y_log_std, Y):
    squared_errors = (Y - Y_pred) ** 2
    variance = torch.exp(Y_log_std) ** 2
    per_sample_losses = 0.5 * (squared_errors / variance + Y_log_std)
    return torch.mean(per_sample_losses)


def example():
    import matplotlib.pyplot as plt


    random_state = np.random.RandomState(0)

    n_samples = 200
    x = np.linspace(0, 4, n_samples)
    y = np.exp(x)
    y += (max(x) - x) ** 2 * random_state.randn(x.shape[0])

    X = x[:, np.newaxis]
    Y = y[:, np.newaxis]

    model = ProbRegressionNN(1, 1, n_features=10, verbose=1)

    model.fit(X, Y)
    Y_pred, Y_std = model.predict(X, return_std=True)
    print(model.score(X, Y))

    plt.scatter(X[:, 0], Y[:, 0])
    plt.plot(X[:, 0], Y_pred[:, 0])
    for factor in [1, 1.5, 2]:
        plt.fill_between(
            X[:, 0],
            (Y_pred - factor * Y_std)[:, 0],
            (Y_pred + factor * Y_std)[:, 0],
            alpha=0.1
        )
    plt.show()


example()
