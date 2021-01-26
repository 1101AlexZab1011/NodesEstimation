import mne
import nibabel
import pickle

save = {
    'raw': lambda path, raw: raw.save(path),
    'bem': mne.write_bem_solution,
    'src': lambda path, src: src.save(path),
    'trans': mne.write_trans,
    'fwd': mne.write_forward_solution,
    'eve': mne.write_events,
    'epo': lambda path, epochs: epochs.save(path),
    'cov': mne.write_cov,
    'ave': mne.write_evokeds,
    'inv': mne.minimum_norm.write_inverse_operator,
    'stc': lambda path, stc: pickle.dump(stc, open(path, 'wb')),
    'coords': lambda path, coord: pickle.dump(coord, open(path, 'wb')),
    'resec': nibabel.save,
    'resec_mni': lambda path, resec: pickle.dump(resec, open(path, 'wb')),
    'parc': lambda path, parc: pickle.dump(parc, open(path, 'wb')),
    'feat': lambda path, feat: pickle.dump(feat, open(path, 'wb')),
    'nodes': lambda path, nodes: pickle.dump(nodes, open(path, 'wb'))
}

read = {
    'raw': mne.io.read_raw_fif,
    'bem': mne.read_bem_solution,
    'src': mne.read_source_spaces,
    'trans': mne.read_trans,
    'fwd': mne.read_forward_solution,
    'eve': mne.read_events,
    'epo': mne.read_epochs,
    'cov': mne.read_cov,
    'ave': mne.read_evokeds,
    'inv': mne.minimum_norm.read_inverse_operator,
    'stc': lambda path: pickle.load(open(path, 'rb')),
    'coords': lambda path: pickle.load(open(path, 'rb')),
    'resec': nibabel.load,
    'resec_mni': lambda path: pickle.load(open(path, 'rb')),
    'parc': lambda path: pickle.load(open(path, 'rb')),
    'feat': lambda path: pickle.load(open(path, 'rb')),
    'nodes': lambda path: pickle.load(open(path, 'rb'))
}
