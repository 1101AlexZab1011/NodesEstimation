import mne
import os
import pickle
import hashlib
import nodestimation.project.path as path
from nodestimation.project.subject import Subject
from nodestimation.project.structures import subject_data_types
from nodestimation.mlearning.features import \
    prepare_features, \
    prepare_data
from nodestimation.mneraw import \
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
    parcellation_creating, \
    read_original_resec, \
    resection_area_computation, \
    features_computation, \
    nodes_creation

write_subjects = lambda path, subjects: pickle.dump(subjects, open(path, 'wb'))

read_subjects = lambda path: pickle.load(open(path, 'rb'))


def conditions_unique_code(*args):
    out = ''
    for arg in args:
        out += str(arg)
    return hashlib.md5(bytes(out, 'utf-8')).hexdigest()


def pipeline(
        crop_time=120,
        snr=0.5,
        epochs_tmin=-1,
        epochs_tmax=1,
        conductivity=(0.3,),
        se_method="sLORETA",
        methods='plv',
        rfreq=200,
        nfreq=50,
        lfreq=1,
        hfreq=70,
        freq_bands=(0.5, 4),
        priority=0
):
    def get_ith(list_, i):
        if isinstance(list_, list):
            return list_[i]
        else:
            return list_

    if not isinstance(methods, list):
        methods = [methods]
    if not isinstance(freq_bands, list):
        freq_bands = [freq_bands]
    if any([not isinstance(freq_band, tuple) for freq_band in freq_bands]) \
            or any([len(freq_band) % 2 != 0 for freq_band in freq_bands]):
        raise ValueError('freq_bands must contain a list of frequency bands with [minimum_frequency, maximum_frequency]'
                         ' or list of lists with frequency bands, however it contains: {}'.format(freq_bands))

    conditions_code = conditions_unique_code(
        crop_time,
        snr,
        epochs_tmin,
        epochs_tmax,
        conductivity,
        se_method,
        methods,
        rfreq,
        nfreq,
        lfreq,
        hfreq,
        freq_bands
    )
    lambda2 = 1.0 / snr ** 2
    subjects_dir, subjects_ = path.find_subject_dir()
    subjects_file = os.path.join(subjects_dir, 'subjects_information_for_' + conditions_code + '.pkl')
    if os.path.exists(subjects_file):
        print('All computation has been already done, loading of the existing file with the solution...')
        return read_subjects(subjects_file)
    else:
        print('Building of the resources files tree...')
        tree = path.build_resources_tree(subjects_)
        subjects = list()
        print('Preparing data...')
        for subject in tree:
            raw, raw_path = get_ith(read_original_raw('./', _subject_tree=tree[subject], _conditions=None), priority)
            fp_raw, fp_raw_path = get_ith(first_processing(raw,
                                                           lfreq,
                                                           nfreq,
                                                           hfreq,
                                                           rfreq=rfreq,
                                                           crop=crop_time,
                                                           _subject_tree=tree[subject],
                                                           _conditions=conditions_code), priority)
            bem, bem_path = get_ith(bem_computation(subject,
                                                    subjects_dir,
                                                    conductivity,
                                                    _subject_tree=tree[subject],
                                                    _conditions=None), priority)
            src, src_path = get_ith(src_computation(subject,
                                                    subjects_dir,
                                                    bem, _subject_tree=tree[subject],
                                                    _conditions=None), priority)
            trans, trans_path = get_ith(read_original_trans('./', _subject_tree=tree[subject], _conditions=None))
            fwd, fwd_path = get_ith(forward_computation(fp_raw.info,
                                                        trans,
                                                        src,
                                                        bem,
                                                        _subject_tree=tree[subject],
                                                        _conditions=None), priority)
            eve, eve_path = get_ith(events_computation(fp_raw,
                                               range(1, 59),
                                               [1 for i in range(58)],
                                               _subject_tree=tree[subject],
                                               _conditions=conditions_code), priority)
            epo, epo_path = get_ith(epochs_computation(fp_raw,
                                               eve,
                                               epochs_tmin,
                                               epochs_tmax,
                                               _subject_tree=tree[subject],
                                               _conditions=conditions_code), priority)
            cov, cov_path = get_ith(noise_covariance_computation(epo,
                                                         epochs_tmin,
                                                         0,
                                                         _subject_tree=tree[subject],
                                                         _conditions=conditions_code), priority)
            ave, ave_path = get_ith(evokeds_computation(epo,
                                                        _subject_tree=tree[subject],
                                                        _conditions=conditions_code), priority)
            inv, inv_path = get_ith(inverse_computation(epo.info,
                                                fwd,
                                                cov,
                                                _subject_tree=tree[subject],
                                                _conditions=conditions_code), priority)
            stc, stc_path = get_ith(source_estimation(epo,
                                              inv,
                                              lambda2,
                                              se_method,
                                              _subject_tree=tree[subject],
                                              _conditions=conditions_code), priority)
            resec, resec_path = get_ith(read_original_resec('./',
                                                            _subject_tree=tree[subject],
                                                            _conditions=None), priority)
            resec_mni, resec_mni_path = get_ith(resection_area_computation(resec,
                                                                            _subject_tree=tree[subject],
                                                                            _conditions=conditions_code), priority)
            labels_parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
            labels_aseg = mne.get_volume_labels_from_src(src[0], subject,
                                                         subjects_dir)  # choosed the first src. It should be fixed in future release
            labels = labels_parc + labels_aseg  # where is label_aseg?
            parc, parc_path = get_ith(parcellation_creating(subject,
                                                    subjects_dir,
                                                    labels,
                                                    _subject_tree=tree[subject],
                                                    _conditions=conditions_code), priority)
            coords, coords_path = get_ith(coordinates_computation(subject,
                                                          subjects_dir,
                                                          labels,
                                                          _subject_tree=tree[subject],
                                                          _conditions=conditions_code), priority)
            label_names = [label.name for label in labels]
            label_ts = mne.extract_label_time_course(stc, labels, src[0], mode='mean_flip')
            feat, feat_path = get_ith(features_computation(
                epo,
                inv,
                lambda2,
                4,
                labels,
                label_ts,
                fp_raw.info['sfreq'],
                freq_bands,
                methods,
                se_method,
                _subject_tree=tree[subject],
                _conditions=conditions_code
            ), priority)
            nodes, nodes_path = get_ith(nodes_creation(
                labels,
                prepare_features(label_names, feat),
                coords,
                resec_mni,
                _subject_tree=tree[subject],
                _conditions=conditions_code
            ), priority)
            dataset, dataset_path = get_ith(prepare_data(
                nodes,
                _subject_tree=tree[subject],
                _conditions=conditions_code
            ), priority)
            subjects.append(
                Subject(
                    subject,
                    {
                        data_type: data_path
                        for data_type, data_path
                        in zip(subject_data_types, [
                        raw_path,
                        fp_raw_path,
                        bem_path,
                        src_path,
                        trans_path,
                        fwd_path,
                        eve_path,
                        epo_path,
                        cov_path,
                        ave_path,
                        inv_path,
                        stc_path,
                        resec_path,
                        resec_mni_path,
                        parc_path,
                        coords_path,
                        feat_path,
                        nodes_path,
                        dataset_path
                    ])
                    },
                    nodes,
                    subjects_[subject],
                    dataset
                )
            )

        print('All the data has been prepared. Saving the result...')

        write_subjects(subjects_file, subjects)

        print('Successfully saved')

        return subjects
