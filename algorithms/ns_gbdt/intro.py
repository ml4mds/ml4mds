# -*- coding: utf-8 -*-
"""Introduction to fuzzmdd and fuzzmda and their parameters."""


def nsgbdt_intro(col_algo1, col_algo2):
    """Introduction to N_S_GBDT and their parameters."""
    col_fig, col_txt = col_algo1.columns(2)
    # col_fig.image('algorithms/fuzzm/intro.png')
    col_txt.markdown("""
                     """)
    training_size = col_algo2.number_input('Training set size (>0):',
                                           value=100)
    batch_size = col_algo2.number_input('Chunk size (>0):', value=100)
    max_iter = col_algo2.number_input('Max iteration: ', value=50)
    sample_rate = col_algo2.number_input('Sample rate: ', value=50)
    learn_rate = col_algo2.number_input('learning rate: ', value=50)
    max_depth = col_algo2.number_input('Max depth of trees: ', value=50)
    return training_size, batch_size, max_iter, sample_rate, \
        learn_rate, max_depth, False
