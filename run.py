import numpy as np
from mne.minimum_norm import compute_source_psd

from nodestimation.pipeline import pipeline
import nilearn.plotting as nplt
import matplotlib.pyplot as plt
from nodestimation.project.actions import read

subjects = pipeline(
    methods=['wpli', 'envelope'],
    freq_bands=[(0.5, 4), (4, 7), (7, 14), (14, 30), (30, 70)],
    centrality_metrics=['eigen', 'close', 'between']#, 'degree', 'katz', 'info', 'harmonic']
)

# for subject in subjects:
#     epochs = read['epo'](subject.data['epo'])
#     epochs.plot_psd(picks='meg')
