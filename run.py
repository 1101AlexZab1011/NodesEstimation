import numpy as np
from mne.minimum_norm import compute_source_psd

from nodestimation.pipeline import pipeline
import nilearn.plotting as nplt
import matplotlib.pyplot as plt
from nodestimation.project.actions import read

subjects = pipeline(
    methods=['wpli', 'envelope'],
    freq_bands=(7.5, 12),
    centrality_metrics=['eigen', 'close', 'between', 'degree', ],# 'katz', 'info', 'harmonic']
    subjects_specificity={
        'M2S2': {
            'freq_bands': (7.5, 12.5)
        },
        'R1D2': {
            'freq_bands': (7.5, 11)
        },
        'S1A2': {
            'freq_bands': (5, 10)
        },
        'S1H1': {
            'freq_bands': (8, 13)
        },
        'K1V1': {
            'freq_bands': (7.5, 11)
        },
        'L1P1': {
            'freq_bands': (5, 10)
        },
        'M1G2': {
            'freq_bands': (7, 11)
        },
        'G1V2': {
            'freq_bands': (7, 11)
        },
        'G1R1': {
            'freq_bands': (12.5, 16.5)
        },
        'M1N2': {
            'freq_bands': (10, 15)
        },
        'B1R1': {
            'freq_bands': (6, 11)
        },
        'B1C2': {
            'freq_bands': (7.5, 12.5)
        },
        'J1T2': {
            'freq_bands': (11, 15)
        },
        'O1O2': {
            'freq_bands': (5.5, 9.5)
        },
    }
)

# for subject in subjects:
#     epochs = read['epo'](subject.data['epo'])
#     epochs.plot_psd(picks='meg')
