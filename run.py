import numpy as np
from nodestimation.pipeline import pipeline
import nilearn.plotting as nplt
import matplotlib.pyplot as plt
from nodestimation.project.actions import read

subjects = pipeline(
    methods=['wpli', 'psd', 'envelope', 'imcoh', 'ciplv'],
    freq_bands=[(0.5, 4), (4, 7), (7, 14), (14, 30), (30, 70)],
    centrality_metrics=['eigen', 'close', 'between']
)


