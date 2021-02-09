import mne
import os
import pickle
import nodestimation.project as proj
from nodestimation.project.subject import Subject
from nodestimation.project.structures import subject_data_types
from nodestimation.processing.features import \
    prepare_features, \
    prepare_data
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
    parcellation_creating, \
    read_original_resec, \
    resection_area_computation, \
    features_computation, \
    nodes_creation, read_original_resec_txt


def write_subjects(path, subjects):
    pickle.dump(subjects, open(path, 'wb'))


def read_subjects(path):
    return pickle.load(open(path, 'rb'))


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
):
    if not isinstance(methods, list):
        methods = [methods]
    if not isinstance(freq_bands, list):
        freq_bands = [freq_bands]
    if any([not isinstance(freq_band, tuple) for freq_band in freq_bands]) \
            or any([len(freq_band) % 2 != 0 for freq_band in freq_bands]):
        raise ValueError('freq_bands must contain a list of frequency bands with [minimum_frequency, maximum_frequency]'
                         ' or list of lists with frequency bands, however it contains: {}'.format(freq_bands))

    conditions_code = proj.conditions_unique_code(
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
    subjects_dir, subjects_ = proj.find_subject_dir()
    subjects_file = os.path.join(subjects_dir, 'subjects_information_for_' + conditions_code + '.pkl')
    if os.path.exists(subjects_file):
        print('All computation has been already done, loading of the existing file with the solution...')
        return read_subjects(subjects_file)
    else:
        print('Building of the resources files tree...')
        tree = proj.build_resources_tree(subjects_)
        subjects = list()
        print('Preparing data...')
        for subject in tree:
            raw, raw_path = read_original_raw(None, _subject_tree=tree[subject], _conditions=None)
            fp_raw, fp_raw_path = first_processing(raw,
                                                   lfreq,
                                                   nfreq,
                                                   hfreq,
                                                   rfreq=rfreq,
                                                   crop=crop_time,
                                                   _subject_tree=tree[subject],
                                                   _conditions=conditions_code)
            bem, bem_path = bem_computation(subject,
                                            subjects_dir,
                                            conductivity,
                                            _subject_tree=tree[subject],
                                            _conditions=None)
            src, src_path = src_computation(subject,
                                            subjects_dir,
                                            bem, _subject_tree=tree[subject],
                                            _conditions=None)
            trans, trans_path = read_original_trans(None, _subject_tree=tree[subject], _conditions=None)
            fwd, fwd_path = forward_computation(fp_raw.info,
                                                trans,
                                                src,
                                                bem,
                                                _subject_tree=tree[subject],
                                                _conditions=None)
            eve, eve_path = events_computation(fp_raw,
                                               range(1, 59),
                                               [1 for i in range(58)],
                                               _subject_tree=tree[subject],
                                               _conditions=conditions_code)
            epo, epo_path = epochs_computation(fp_raw,
                                               eve,
                                               epochs_tmin,
                                               epochs_tmax,
                                               _subject_tree=tree[subject],
                                               _conditions=conditions_code)
            cov, cov_path = noise_covariance_computation(epo,
                                                         epochs_tmin,
                                                         0,
                                                         _subject_tree=tree[subject],
                                                         _conditions=conditions_code)
            ave, ave_path = evokeds_computation(epo,
                                                _subject_tree=tree[subject],
                                                _conditions=conditions_code)
            inv, inv_path = inverse_computation(epo.info,
                                                fwd,
                                                cov,
                                                _subject_tree=tree[subject],
                                                _conditions=conditions_code)
            stc, stc_path = source_estimation(epo,
                                              inv,
                                              lambda2,
                                              se_method,
                                              _subject_tree=tree[subject],
                                              _conditions=conditions_code)
            resec, resec_path = read_original_resec(None,
                                                    _subject_tree=tree[subject],
                                                    _conditions=None)
            resec_txt, resec_txt_path = read_original_resec_txt(None,
                                                                _subject_tree=tree[subject],
                                                                _conditions=None)
            resec_mni, resec_mni_path = resection_area_computation(resec,
                                                                   _subject_tree=tree[subject],
                                                                   _conditions=conditions_code)
            labels_parc = mne.read_labels_from_annot(subject, parc='aparc.a2009s', subjects_dir=subjects_dir)
            labels_aseg = mne.get_volume_labels_from_src(src, subject,
                                                         subjects_dir)
            labels = labels_parc + labels_aseg
            parc, parc_path = parcellation_creating(subject,
                                                    subjects_dir,
                                                    labels,
                                                    _subject_tree=tree[subject],
                                                    _conditions=conditions_code)
            coords, coords_path = coordinates_computation(subject,
                                                          subjects_dir,
                                                          labels,
                                                          _subject_tree=tree[subject],
                                                          _conditions=conditions_code)
            label_names = [label.name for label in labels]
            label_ts = mne.extract_label_time_course(stc, labels, src, mode='pca_flip')
            feat, feat_path = features_computation(
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
            )
            nodes, nodes_path = nodes_creation(
                labels,
                prepare_features(label_names, feat),
                coords,
                resec_mni,
                resec_txt,
                _subject_tree=tree[subject],
                _conditions=conditions_code
            )
            dataset, dataset_path = prepare_data(
                nodes,
                _subject_tree=tree[subject],
                _conditions=conditions_code
            )
            subjects.append(
                Subject(
                    subject,
                    {
                        data_type: data_path
                        for data_type, data_path
                        in zip(
                        subject_data_types, [
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
                            resec_txt_path,
                            resec_mni_path,
                            parc_path,
                            coords_path,
                            feat_path,
                            nodes_path,
                            dataset_path
                        ]
                    )
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