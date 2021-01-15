import mne
import nibabel
import pickle


# def write_raw(path, raw):
#     raw.save(path)
#
#
# def write_src(path, src):
#     src.save(path)
#
#
# def write_epochs(path, epochs):
#     epochs.save(path)
#
#
# def write_stc(path, stc):
#     pickle.dump(stc, open(path, 'wb'))
#
#
# def read_stc(path):
#     return pickle.load(open(path, 'rb'))
#
#
# def write_resec(path, resec):
#     pickle.dump(resec, open(path, 'wb'))
#
#
# def read_resec(path):
#     return pickle.load(open(path, 'rb'))


save = {
    'raw': lambda path, raw: raw.save(path),
    'src': lambda path, src: src.save(path),
    'bem': mne.write_bem_solution,
    'trans': mne.write_trans,
    'cov': mne.write_cov,
    'fwd': mne.write_forward_solution,
    'inv': mne.minimum_norm.write_inverse_operator,
    'eve': mne.write_events,
    'epo': lambda path, epochs: epochs.save(path),
    'ave': mne.write_evokeds,
    'stc': lambda path, stc: pickle.dump(stc, open(path, 'wb')),
    'resec': nibabel.save,
    'resec-mni': lambda path, resec: pickle.dump(resec, open(path, 'wb'))
}

save_format = {
    'raw': 'fif',
    'src': 'fif',
    'bem': 'fif',
    'trans': 'fif',
    'cov': 'fif',
    'fwd': 'fif',
    'inv': 'fif',
    'eve': 'fif',
    'epo': 'fif',
    'ave': 'fif',
    'stc': 'pkl',
    'resec': 'nii',
    'resec-mni': 'pkl'
}

read = {
    'raw': mne.io.read_raw_fif,
    'src': mne.read_source_spaces,
    'bem': mne.read_bem_solution,
    'trans': mne.write_trans,
    'cov': mne.write_cov,
    'fwd': mne.read_forward_solution,
    'inv': mne.minimum_norm.read_inverse_operator,
    'eve': mne.read_events,
    'epo': mne.read_epochs,
    'ave': mne.read_evokeds,
    'stc': lambda path: pickle.load(open(path, 'rb')),
    'resec': nibabel.load,
    'resec-mni': lambda path: pickle.load(open(path, 'rb'))
}