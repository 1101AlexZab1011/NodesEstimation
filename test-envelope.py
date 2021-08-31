import mne
from abc import ABC
from typing import *

import mne
import os
import pickle

import numpy as np

import nodestimation.project as proj
from nodestimation.project import SubjectTree
from nodestimation.project.subject import Subject
from nodestimation.project.structures import subject_data_types
from nodestimation.processing.features import \
    prepare_features, \
    prepare_data, \
    prepare_connectomes
from nodestimation.processing.mneraw import \
    read_original_raw, \
    first_processing, \
    src_computation, \
    bem_computation, \
    read_original_trans, \
    forward_computation, \
    events_computation, \
    epochs_computation, \
    evokeds_computation, \
    noise_covariance_computation, \
    inverse_computation, \
    source_estimation, \
    coordinates_computation, \
    read_original_resec, \
    resection_area_computation, \
    features_computation, \
    nodes_creation, \
    read_original_resec_txt
from nodestimation.pipeline import pipeline

SUBJECTS = pipeline(
    methods=['wpli', 'envelope', 'coh', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'pli2_unbiased', 'wpli2_debiased'],
    freq_bands=[(4, 8), (8, 14), (6, 8), (8, 10)],
)

subjects_dir, _ = proj.find_subject_dir()


def correlation_connectivity_computation(label_ts, fmin=4, fmax=8, sfreq=200) -> np.ndarray:
    if isinstance(label_ts, list):
        label_ts = np.array(label_ts)

    label_ts = mne.filter.filter_data(label_ts, sfreq, l_freq=fmin, h_freq=fmax, method='fir', copy=True)

    return mne.connectivity.envelope_correlation(label_ts)


subject = SUBJECTS[0]
labels = mne.read_labels_from_annot(subject.name, parc='aparc.a2009s', subjects_dir=subjects_dir)
src = proj.actions.read['src'](subject.data['src'])
stc = proj.actions.read['stc'](subject.data['stc'])
label_ts = mne.extract_label_time_course(stc, labels, src, mode='pca_flip')

con1 = correlation_connectivity_computation(label_ts)
con2 = correlation_connectivity_computation(label_ts)
diff = np.abs(con1-con2)
print(
    f'AVERAGE 1: {np.mean(np.mean(con1))}'
    f'AVERAGE 2: {np.mean(np.mean(con2))}'
    f'MEAN DIFF: {np.round(np.mean(np.mean(diff)), 5)}\n'
)
