# -*- coding: utf-8 -*-
"""Baseline.

Training at first and do nothing.
"""

from copy import copy

import numpy as np


class BaselineStreamHandler:
    """Handle a stream in dynamic environment.

    Train a model at first and do nothing.
    """

    def __init__(self, base_learner):
        """__init__ for BaselineStreamHandler."""
        self.learner = base_learner

    def fit(self, x, y):
        """Fit method."""
        self.learner.fit(x, y)

    def predict(self, x):
        """Score method."""
        yhat = self.learner.predict(x)
        return yhat


class BaselineMultiStreamHandler:
    """Handle multiple data streams in dynamic environment."""

    def __init__(self, m, base_learner):
        """__init__ for BaselineMultiStreamHandler."""
        self.handlers = [BaselineStreamHandler(copy(base_learner))
                         for _ in range(m)]

    def fit(self, x, y):
        """Fit method."""
        for i, hdlr in enumerate(self.handlers):
            hdlr.fit(x[:, i, :], y[:, i])

    def predict(self, x):
        """Predict method."""
        n, m, _ = x.shape
        result = np.zeros(n, m)
        for i in range(m):
            result[:, i] = self.handlers[i].predict(x[:, i, :])
        return result
