# -*- coding: utf-8 -*-
"""Introduction to fuzzmdd and fuzzmda and their parameters."""


def fuzzm_intro(col_algo1, col_algo2):
    """Introduction to fuzzmdd and fuzzmda and their parameters."""
    col_fig, col_txt = col_algo1.columns(2)
    col_fig.image('algorithms/fuzzm/intro.png')
    col_txt.markdown("""
                     **Fuzzy Membership-based Drift Detection** and
                     **Fuzzy Membership-based Drift Adaptation** jointly
                     constitute a novel multi-stream concept drift handling
                     framework.

                     A **base learner** is initialized for each data stream
                     using the begining part of the data
                     as the **training set**.
                     Then data arrive **chunk by chunk**.
                     A stream fuzzy set is defined for each data stream
                     to model the correlations between streams.
                     If a concept drift is detected,
                     data from other streams are used to retrain a new model,
                     with their fuzzy membership as weight.
                     """)
    base_learner = col_algo2.selectbox('Base learner:',
                                       ('tree model',
                                        'linear model'))
    training_size = col_algo2.number_input('Training set size (>0):',
                                           value=100)
    batch_size = col_algo2.number_input('Chunk size (>0):', value=100)
    return training_size, batch_size, base_learner, True
