import mne
from nodestimation.project.path import read_or_write

def notchfir(raw, lfreq, nfreq, hfreq):

    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False)
    raw_filtered = raw \
        .load_data() \
        .notch_filter(nfreq, meg_picks) \
        .filter(l_freq=lfreq, h_freq=hfreq)

    return raw_filtered


def artifacts_clean(raw):

    ica = mne.preprocessing.ICA(n_components=15, random_state=97)
    ica.fit(raw)
    ica.exclude = ica.find_bads_eog(raw)[0] + \
                  ica.find_bads_ecg(raw, method='correlation', threshold=3.0)[0]

    ica.apply(raw)

    del ica

    return raw

@read_or_write('raw', target='original', write_file=False)
def read_original_raw(path, _subject_tree=None):
    return mne.io.read_raw_fif(path)

@read_or_write('raw', target='nepf')
def first_processing(raw, lfreq, nfreq, hfreq,
                     rfreq=None,
                     crop=None,
                     reconstruct=False,
                     meg=True,
                     eeg=True,
                     _subject_tree=None):

    out = raw.copy()

    if crop:
        if not isinstance(crop, list):
            out.crop(tmax=crop)
        elif isinstance(crop, list) \
                and len(crop) == 2:
            out.crop(tmin=crop[0], tmax=crop[1])
        else:
            raise ValueError('Crop range is incorrect: {}'. format(crop))

    if rfreq:
        out.resample(rfreq, npad='auto')

    out = notchfir(out, lfreq, nfreq, hfreq)
    if reconstruct:
        out = artifacts_clean(out)

    out.pick_types(meg=meg, eeg=eeg)

    return out
