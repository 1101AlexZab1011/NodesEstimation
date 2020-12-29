import re
import os
import numpy as np
import mne
import nilearn.plotting as nplt
import pickle
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
from sklearn.metrics import roc_auc_score
from node_estimate import Node
from timewindow import TimeWindow, sliding_window
from parcellation import freesurf_dict

root= './'

tree = os.walk(os.path.join(root, 'Source'))

raw_files = []
src_files = []
inv_files = []
bem_files = []
fwd_files = []
trans_files = []
epochs_files = []
ave_files = []
stc_files = []

subjects_found = False

for walk in tree:
    for file in walk[2]:
        if re.search(r'.*raw\.fif', file):
            raw_files.append(os.path.join(walk[0], file))
        if re.search(r'.*src.*\.fif', file) or re.search(r'.*source.*space.*\.fif', file):
            src_files.append(os.path.join(walk[0], file))
        if re.search(r'.*inv.*\.fif', file) or re.search(r'.*inverse.*\.fif', file):
            inv_files.append(os.path.join(walk[0], file))
        if re.search(r'.*bem.*\.fif', file):
            bem_files.append(os.path.join(walk[0], file))
        if re.search(r'.*fwd.*\.fif', file) or re.search(r'.*forward.*\.fif', file):
            fwd_files.append(os.path.join(walk[0], file))
        if re.search(r'.*trans.*\.fif', file):
            trans_files.append(os.path.join(walk[0], file))
        if re.search(r'.*epo.*\.fif', file):
            epochs_files.append(os.path.join(walk[0], file))
        if re.search(r'.*ave.*\.fif', file):
            ave_files.append(os.path.join(walk[0], file))
        if re.search(r'.*stc.*\.fif.*', file):
            stc_files.append(os.path.join(walk[0], file))

    for subdir in walk[1]:
        if subdir == 'Subjects' or subdir == 'subjects' and not subjects_found:
            subjects_found = True
            subjects_dir = os.path.join(walk[0], subdir)
        elif subdir == 'Subjects' or subdir == 'subjects' and subjects_found:
            raise OSError("There are two subjects directories: {}, {}; Only one must be".format(
                subjects_dir, os.path.join(walk[0], subdir)
            ))

if not raw_files:
    raise OSError("No one of raw files are found. Raw file must have extension \'.fif\' and ends with \'raw\'")

if not subjects_found:
    raise OSError("Subjects directory not found!")
subjects = os.listdir(subjects_dir)

subject_dirs = []

for subject in subjects:
    subject_dirs.append(os.path.join(subjects_dir, subject))

raw_file = raw_files[0]
bem_file = bem_files[0]
src_file = src_files[0]
fwd_file = fwd_files[0]
trans_file = trans_files[0]

res_folder = os.path.join(root, 'Pipeline', subjects[0])

res_raw_folder = os.path.join(res_folder, 'Raw')
res_bem_folder = os.path.join(res_folder, 'Bem')
res_src_folder = os.path.join(res_folder, 'Src')
res_fwd_folder = os.path.join(res_folder, 'Fwd')
res_events_folder = os.path.join(res_folder, 'Events')
res_epochs_folder = os.path.join(res_folder, 'Epochs')
res_evoked_folder = os.path.join(res_folder, 'Evoked')
res_cov_folder = os.path.join(res_folder, 'Cov')
res_inv_folder = os.path.join(res_folder, 'Inv')
res_sLORETA_folder = os.path.join(res_folder, 'sLORETA')
res_nodes_folder = os.path.join(res_folder, 'NodesEstimate')

res_raw_file = os.path.join(res_raw_folder, 'raw.fif')
res_bem_file = os.path.join(res_bem_folder, 'raw_bem.fif')
res_src_file = os.path.join(res_src_folder, 'raw_src_ico5.fif')
res_fwd_file = os.path.join(res_fwd_folder, 'raw_fwd_ico5.fif')
res_events_file = os.path.join(res_events_folder, 'raw_eve.fif')
res_epochs_file = os.path.join(res_epochs_folder, 'raw_epo.fif')
res_evoked_file = os.path.join(res_evoked_folder, 'raw_ave.fif')
res_cov_file = os.path.join(res_cov_folder, 'noise_cov.fif')
res_inv_file = os.path.join(res_inv_folder, 'raw_inv.fif')
res_sLORETA_file = os.path.join(res_sLORETA_folder, 'sLORETA_raw_ave_inv.fif')
res_sLORETA_file_lh = os.path.join(res_sLORETA_folder, 'sLORETA_raw_ave_inv.fif-lh.stc')
res_sLORETA_file_rh = os.path.join(res_sLORETA_folder, 'sLORETA_raw_ave_inv.fif-rh.stc')
res_nodes_strength_file = os.path.join(res_nodes_folder, 'nodes_strength.dat')

subject_dir = subject_dirs[0]
subject = subjects[0]
print(subject_dir)


conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
epochs_tmin, epochs_tmax = -30, 30
crop_time = 120
snr = 0.5  # use SNR smaller than 1 for raw data
lambda2 = 1.0 / snr ** 2
method = "sLORETA"
nfreq = 50
lfreq = 1
hfreq = 70


def mkdir(path):

    try:
        os.mkdir(path)

    except OSError:
        print("PIPELINE: creation of the directory %s failed" % path)

    else:
        print("PIPELINE: successfully created the directory %s " % path)


@sliding_window(1200, 0.5)
def pearson(signals):

    nsigmals, lsignals = signals.shape
    out = np.zeros((nsigmals, nsigmals))

    for i in range(nsigmals):
        for j in range(nsigmals):
            out[i, j] = np.corrcoef(signals[i, :], signals[j, :])[0, 1]

    return out


def mean_across_tw(twlist):

    l, w = twlist[0].data.shape
    voxel = voxel_from_tw(twlist)
    out = np.zeros((l, w))

    for i in range(w):
        for j in range(l):
            out[i, j] = np.mean(voxel[i, j, :])

    return out


def voxel_from_tw(twlist):

    l, w = twlist[0].data.shape
    h = len(twlist)
    voxel = np.zeros((l, w, h))

    for i in range(h):
        voxel[:, :, i] = twlist[i].data

    return voxel


def notchfir(raw, lfreq, nfreq, hfreq):

    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False)
    raw_filtered = raw \
        .load_data() \
        .notch_filter(nfreq, meg_picks) \
        .filter(l_freq=lfreq, h_freq=hfreq)

    return raw_filtered


def artifacts_clean(raw):

    ica = ICA(n_components=15, random_state=97)
    ica.fit(raw)
    ica.exclude = ica.find_bads_eog(raw)[0] + \
                  ica.find_bads_ecg(raw, method='correlation', threshold=3.0)[0]

    ica.apply(raw)

    return raw


def first_processing(raw):

    raw_cropped = raw.crop(tmax=crop_time)
    raw_filtered = notchfir(raw_cropped, lfreq, nfreq, hfreq)
    # raw_reconstructed = artifacts_clean(raw_filtered)
    raw_out = raw_filtered.pick_types(meg=True, eeg=False)

    del raw, raw_filtered

    return raw_out


def nodes_strength(label_tc):

    pearson_matrices = pearson(label_tc)
    pears_mean = mean_across_tw(pearson_matrices)
    n_strength = np.array([])

    for i in range(pears_mean.shape[0]):
        n_strength = np.append(n_strength, np.sum(pears_mean[i, :]))

    return n_strength


if not os.path.exists('./Pipeline'):
    mkdir('./Pipeline')

if not os.path.exists(res_folder):
    mkdir(res_folder)

if os.path.isfile(res_raw_file):
    raw = mne.io.read_raw_fif(res_raw_file)

elif os.path.isfile(raw_file):
    raw = mne.io.read_raw_fif(raw_file)
    raw = first_processing(raw)
    path = res_raw_folder

    mkdir(path)

    raw.save(res_raw_file)

else:
    raise OSError('PIPELINE: Raw-file not found')


print(raw.info)

if os.path.isfile(res_src_file):
    src = mne.read_source_spaces(res_src_file)

elif os.path.isfile(src_file):
    src = mne.read_source_spaces(src_file)
    path = res_src_folder

    mkdir(path)

    src.save(res_src_file)

else:
    print('PIPELINE: Source spaces not found, creating a new one...')
    src = mne.setup_source_space(subject, spacing='ico5', add_dist='patch', subjects_dir=subjects_dir)
    path = res_src_folder

    mkdir(path)

    src.save(res_src_file)

if os.path.isfile(res_bem_file):
    bem = mne.read_bem_solution(res_bem_file)

elif os.path.isfile(bem_file):
    bem = mne.read_bem_solution(bem_file)
    path = res_bem_folder

    mkdir(path)

    mne.write_bem_solution(res_bem_file, bem)

else:
    print('PIPELINE: BEM-surface not found, creating a new one...')
    model = mne.make_bem_model(subject=subject, ico=4, conductivity=conductivity, subjects_dir=subject_dir)
    bem = mne.make_bem_solution(model)
    path = res_bem_folder

    mkdir(path)

    mne.write_bem_solution(res_bem_file, bem)

if os.path.isfile(res_fwd_file):
    fwd = mne.read_forward_solution(res_fwd_file)

elif os.path.isfile(fwd_file):
    fwd = mne.read_forward_solution(fwd_file)
    path = res_fwd_folder

    mkdir(path)

    mne.write_forward_solution(res_fwd_file, fwd)

else:
    print('PIPELINE: Forward solution not found, creating a new one...')
    fwd = mne.make_forward_solution(res_raw_file, trans=trans_file, src=src, bem=bem, meg=True, eeg=False,
                                    mindist=5.0, n_jobs=1, verbose=True)
    path = res_fwd_folder

    mkdir(path)

    mne.write_forward_solution(res_fwd_file, fwd)

events = np.array([[
        raw.first_samp + raw.time_as_index(crop_time/2 - 30)[0],
        0,
        1
    ]])

raw.plot(events=events, start=0, duration=120, color='gray', event_color={1: 'r'})

if os.path.isfile(res_epochs_file):
    epochs = mne.read_epochs(res_epochs_file)

else:
    print('PIPELINE: Epochs not found, creating a new one...')
    epochs = mne.Epochs(raw, events, tmin=epochs_tmin, tmax=epochs_tmax,
                        preload=True).resample(200, npad='auto')
    path = res_epochs_folder

    mkdir(path)

    epochs.save(res_epochs_file)


if os.path.isfile(res_evoked_file):
    evoked = mne.read_evokeds(res_evoked_file)
else:
    print('PIPELINE: Evokeds not found, creating a new one...')
    evoked = epochs.average()
    path = res_evoked_folder

    mkdir(path)

    mne.write_evokeds(res_evoked_file, evoked)

    evoked = [evoked]


if os.path.isfile(res_cov_file):
    noise_cov = mne.read_cov(res_cov_file)

else:
    print('PIPELINE: Noise covariance not found, creating a new one...')
    noise_cov = mne.compute_covariance(epochs.copy().pick_types(meg=True, eeg=False, eog=False), tmin=epochs_tmin, tmax=0,
                                       method='empirical')
    path = res_cov_folder

    mkdir(path)

    mne.write_cov(res_cov_file, noise_cov)

if os.path.isfile(res_inv_file):
    inv = mne.minimum_norm.read_inverse_operator(res_inv_file)

else:
    print('PIPELINE: Inverse operator not found, creating a new one...')
    inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov)
    path = res_inv_folder

    mkdir(path)

    mne.minimum_norm.write_inverse_operator(res_inv_file, inv)


if os.path.isfile(res_sLORETA_file_lh) and os.path.isfile(res_sLORETA_file_rh):
    stc = mne.read_source_estimate(res_sLORETA_file_lh)
    stc.__add__(mne.read_source_estimate(res_sLORETA_file_rh))

else:
    print('PIPELINE: sLORETA not found, creating a new one...')
    stc = mne.minimum_norm.apply_inverse(evoked[0], inv, lambda2, 'sLORETA')
    path = res_sLORETA_folder

    mkdir(path)

    stc.save(res_sLORETA_file)


labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)

label_tc = stc.extract_label_time_course(labels, src=inv['src'], mode='mean_flip')


if os.path.isfile(res_nodes_strength_file):
    n_strength = np.fromfile(res_nodes_strength_file,
                      dtype=float)

else:
    print('PIPELINE: Node strength not found, compute a new one')
    n_strength = nodes_strength(label_tc)
    path = res_nodes_folder

    mkdir(path)

    n_strength.tofile(res_nodes_strength_file)

vertexes = [mne.vertex_to_mni(
    label.vertices,
    hemis=0 if label.hemi == 'lh' else 1,
    subject=subject, subjects_dir=subjects_dir
)for label in labels]

# test spectral

out = mne.connectivity.spectral_connectivity(
    mne.minimum_norm.apply_inverse_epochs(
        epochs,
        inv,
        lambda2,
        method,
        pick_ori="normal",
        return_generator=True
    ),
    method='plv',
    sfreq=200,
    mode='fourier'
)
print(out[0].shape)
print(out[0][:, 0, 0])

del stc, src, raw, fwd, \
    n_strength, bem, labels, label_tc, inv, noise_cov

