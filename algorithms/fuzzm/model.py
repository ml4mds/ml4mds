# -*- coding: utf-8 -*-
"""Stream handlers.

Stream handlers.
"""
import numpy as np
from scipy.stats import mannwhitneyu
import torch
import torch.nn as nn


# rank sum test
def rank_sum_test(memb1, memb2, alpha=0.05):
    """Rank sum test."""
    _, p = mannwhitneyu(memb1, memb2)
    if p < alpha:
        return True
    return False


# membership functions
class SMF:
    """sigmoid membership function."""

    def __init__(self):
        """__init__ method."""
        self.func = nn.Sequential(nn.Linear(1, 1, dtype=torch.float64),
                                  nn.Sigmoid())
        self.optimizer = torch.optim.SGD(self.func.parameters(), lr=1e-2)

    def membership(self, x):
        """Return the membership of a ndarray."""
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        x = x.reshape((-1, 1))
        return self.func(x)

    def fit(self, x1, x2, epochs=5):
        """Estimate the parameters of membership.

        x1, x2: ndarray of size (n, d)
        """
        x1 = torch.from_numpy(x1)
        n1 = x1.shape[0]
        y1 = torch.ones(n1, dtype=torch.float64)
        x2 = torch.from_numpy(x2)
        n2 = x2.shape[0]
        y2 = torch.zeros(n2, dtype=torch.float64)
        shuff = torch.randperm(n1+n2)
        X = torch.cat((x1, x2))
        X = X[shuff]
        Y = torch.cat((y1, y2))
        Y = Y[shuff]

        for _ in range(epochs):
            for i in range((n1+n2)//100):
                i1 = i * 100
                i2 = i1 + 100
                xi = X[i1:i2]
                yi = Y[i1:i2]
                yhat = self.membership(xi).reshape(-1)
                loss = nn.functional.mse_loss(yhat, yi)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


class StreamHandler:
    """
    Handle a stream in dynamic environment.

    Stream handler, each stream has a learner and a membership fucntion
    """

    def __init__(self, base_learner, random_state=None):
        """__init__ for StreamHandler."""
        self.learner = base_learner
        self.mf = SMF()
        self.hist_memb = None

    def fit(self, x, y, x1, y1, epochs=100, sample_weight=None):
        """Fit method."""
        self.learner.fit(x, y, sample_weight=sample_weight)
        yhat = self.learner.predict(x)
        loss = (yhat - y) ** 2
        yhat1 = self.learner.predict(x1)
        loss1 = (yhat1 - y1) ** 2
        self.mf.fit(loss, loss1, epochs=epochs)
        self.hist_memb = self.mf.membership(loss).detach().cpu().numpy()

    def score(self, x, y, return_memb=False):
        """Score method.

        x: ndarray of size (n, d)
        y: ndarray of size (n,)
        """
        yhat = self.learner.predict(x)
        loss = (yhat - y) ** 2
        memb = self.mf.membership(loss).detach().cpu().numpy()
        if return_memb:
            return memb.mean()
        drift = rank_sum_test(self.hist_memb, memb, 0.001)
        return loss.mean(), drift


class MultiStreamRetrainer:
    """Handle multiple data streams by re-training in dynamic environment."""

    def __init__(self, m, base_learner, random_state=None):
        """__init__ for MultiStreamHandler."""
        self.handlers = [StreamHandler(base_learner) for _ in range(m)]

    def fit(self, x, y):
        """Fit method."""
        for i, hdlr in enumerate(self.handlers):
            _, m, _ = x.shape
            hdlr.fit(x[:, i, :], y[:, i],
                     x[:, (i+1) % m, :], y[:, (i+1) % m])

    def score(self, x, y):
        """Score method."""
        dlist = []
        n, m, d = x.shape
        result = np.zeros(m)
        for i in range(m):
            result[i], drift = self.handlers[i].score(x[:, i, :], y[:, i])
            if drift:
                dlist.append(i)
        for i in dlist:
            xx = x[:, i, :].reshape(-1, d)
            yy = y[:, i].reshape(-1)
            self.handlers[i].fit(xx, yy, x[:, (i+1) % m, :], y[:, (i+1) % m])
        return result, dlist


class MultiStreamHandler:
    """Handle multiple data streams in dynamic environment."""

    def __init__(self, m, base_learner, random_state=None):
        """__init__ for MultiStreamHandler."""
        self.handlers = [StreamHandler(base_learner) for _ in range(m)]

    def fit(self, x, y):
        """Fit method."""
        for i, hdlr in enumerate(self.handlers):
            _, m, _ = x.shape
            hdlr.fit(x[:, i, :], y[:, i],
                     x[:, (i+1) % m, :], y[:, (i+1) % m])

    def score(self, x, y):
        """Score method."""
        dlist = []
        nlist = []
        n, m, d = x.shape
        result = np.zeros(m)
        for i in range(m):
            result[i], drift = self.handlers[i].score(x[:, i, :], y[:, i])
            if drift:
                dlist.append(i)
            else:
                nlist.append(i)
        for i in dlist:
            tlist = [j for j in nlist]
            tlist.append(i)
            xx = x[:, tlist, :].reshape(-1, d)
            yy = y[:, tlist].reshape(-1)
            weight = np.ones(len(tlist)*n)
            for ji, j in enumerate(tlist):
                if j == i:
                    break
                j1 = ji * n
                j2 = j1 + n
                weight[j1:j2] *= self.handlers[j].score(x[:, i, :],
                                                        y[:, i],
                                                        True)
            self.handlers[i].fit(xx, yy, x[:, (i+1) % m, :], y[:, (i+1) % m],
                                 sample_weight=weight)
        return result, dlist


if __name__ == "__main__":
    mf = SMF()
    x1 = np.random.normal(1, 1, 1000)
    x2 = np.random.normal(2, 1, 1000)
    mf.fit(x1, x2, 10000)
    print(mf.membership(x1).reshape(-1))
    print(mf.membership(x2).reshape(-1))
