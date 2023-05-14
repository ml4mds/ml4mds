# -*- coding: utf-8 -*-
"""Baseline.

Training at first and do nothing."""
import numpy as np


class BaselineStreamHandler:
    """
    Handle a stream in dynamic environment.

    Train a model at first and do nothing.
    """

    def __init__(self, base_learner):
        """__init__ for BaselineStreamHandler."""
        self.learner = base_learner

    def fit(self, x, y):
        """Fit method."""
        self.learner.fit(x, y)

    def score(self, x, y):
        """Score method."""
        yhat = self.learner.predict(x)
        loss = (yhat - y) ** 2
        return loss.mean()


class BaselineMultiStreamHandler:
    """Handle multiple data streams in dynamic environment."""

    def __init__(self, m, base_learner):
        """__init__ for BaselineMultiStreamHandler."""
        self.handlers = [BaselineStreamHandler(base_learner) for _ in range(m)]

    def fit(self, x, y):
        """Fit method."""
        for i, hdlr in enumerate(self.handlers):
            hdlr.fit(x[:, i, :], y[:, i])

    def score(self, x, y):
        """Score method."""
        _, m, _ = x.shape
        result = np.zeros(m)
        for i in range(m):
            result[i] = self.handlers[i].score(x[:, i, :], y[:, i])
        return result
