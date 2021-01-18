file_save_format = {
    'raw': 'fif',
    'bem': 'fif',
    'src': 'fif',
    'trans': 'fif',
    'fwd': 'fif',
    'eve': 'fif',
    'epo': 'fif',
    'cov': 'fif',
    'ave': 'fif',
    'inv': 'fif',
    'stc': 'pkl',
    'coords': 'pkl',
    'resec': 'nii',
    'resec_mni': 'pkl',
    'parc': 'pkl',
    'con': 'pkl',
    'psd': 'pkl',
    'nodes': 'pkl'
}

file_search_regexps = {
    'raw': r'.*raw.*\.fif',
    'bem': r'.*bem.*\.fif',
    'src': [r'.*src.*\.fif', r'.*source_space.*\.fif', r'.*source-space.*\.fif'],
    'trans': r'.*trans.*\.fif',
    'fwd': [r'.*fwd.*\.fif', r'.*forward.*\.fif'],
    'eve': r'.*eve.*',
    'epo': r'.*epo.*',
    'cov': r'.*cov.*\.fif',
    'ave': [r'.*ave.*\.fif', r'.*evoked.*\.fif'],
    'inv': r'.*inv.*\.fif',
    'stc': [r'.*stc.*\.fif', r'.*stc.*\.pkl'],
    'coords': [r'.*coord.*\.pkl', r'.*coordinate.*\.pkl'],
    'resec': r'.*resec.*\.nii.*',
    'resec_mni': r'.*resec.*\.pkl.*',
    'parc': r'.*parc.*\.pkl.*',
    'con': r'.*con.*\.pkl.*',
    'psd': [r'.*psd.*\.pkl.*', r'.*power_spectral_destiny.*\.pkl', r'.*power-spectral-destiny.*\.pkl'],
    'nodes': r'.*nodes.*\.pkl.*'
}

data_types = (
    'raw',
    'bem',
    'src',
    'trans',
    'fwd',
    'eve',
    'epo',
    'cov',
    'ave',
    'inv',
    'stc',
    'coords',
    'resec',
    'resec_mni',
    'parc',
    'con',
    'psd',
    'nodes'
)

connectivity_computation_output_features = (
    'con',
    'freqs',
    'times',
    'n_epochs',
    'n_tapers'
)

ml_features = (
    'coh',
    'imcoh',
    'plv',
    'ppc',
    'pli',
    'psd'
)