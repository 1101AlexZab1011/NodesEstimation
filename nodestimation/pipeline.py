import mne
import os
import pickle
import hashlib
import nodestimation.project.path as path
from nodestimation.project.subject import Subject
from nodestimation.project.structures import subject_data_types
from nodestimation.mlearning.features import \
    prepare_connectivity, \
    prepare_psd
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
    connectivity_computation, \
    power_spectral_destiny_computation, \
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
        con_method='plv',
        rfreq=200,
        nfreq=50,
        lfreq=1,
        hfreq=70,
        freq_diaps=(0.5, 4)
):
    conditions_code = conditions_unique_code(
        crop_time,
        snr,
        epochs_tmin,
        epochs_tmax,
        conductivity,
        se_method,
        con_method,
        rfreq,
        nfreq,
        lfreq,
        hfreq,
        freq_diaps
    )
    lambda2 = 1.0 / snr ** 2
    subjects_dir, subjects = path.find_subject_dir()
    subjects_file = os.path.join(subjects_dir, 'subjects_information_for_' + conditions_code + '.pkl')
    if os.path.exists(subjects_file):
        print('All computation has been already done, loading of the existing file with the solution...')
        return read_subjects(subjects_file)
    else:
        print('Building of the resources files tree...')
        tree = path.build_resources_tree(subjects)
        subjects = list()
        print('Preparing data...')
        for subject in tree:
            raw, raw_path = read_original_raw('./', _subject_tree=tree[subject], _conditions=None)
            fp_raw, fp_raw_path = first_processing(raw,
                                                   lfreq,
                                                   nfreq,
                                                   hfreq,
                                                   rfreq=rfreq,
                                                   crop=crop_time,
                                                   _subject_tree=tree[subject],
                                                   _conditions=conditions_code)
            bem, bem_path = bem_computation(subject, subjects_dir, conductivity, _subject_tree=tree[subject],
                                            _conditions=None)
            src, src_path = src_computation(subject, subjects_dir, bem, _subject_tree=tree[subject], _conditions=None)
            trans, trans_path = read_original_trans('./', _subject_tree=tree[subject], _conditions=None)
            fwd, fwd_path = forward_computation(fp_raw, trans, src, bem, _subject_tree=tree[subject], _conditions=None)
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
            ave, ave_path = evokeds_computation(epo, _subject_tree=tree[subject], _conditions=conditions_code)
            inv, inv_path = inverse_computation(epo.info,
                                                fwd[0],
                                                cov,
                                                _subject_tree=tree[subject],
                                                _conditions=conditions_code)  # choosed the first fwd. It should be fixed in future release
            stc, stc_path = source_estimation(epo,
                                              inv,
                                              lambda2,
                                              se_method,
                                              _subject_tree=tree[subject],
                                              _conditions=conditions_code)
            resec, resec_path = read_original_resec('./', _subject_tree=tree[subject], _conditions=None)
            resec_mni, resec_mni_path = resection_area_computation(resec, _subject_tree=tree[subject],
                                                                   _conditions=conditions_code)
            labels_parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
            labels_aseg = mne.get_volume_labels_from_src(src[0], subject,
                                                         subjects_dir)  # choosed the first src. It should be fixed in future release
            labels = labels_parc + labels_aseg  # where is label_aseg?
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
            label_ts = mne.extract_label_time_course(stc, labels, src[0], mode='mean_flip')
            con, con_path = connectivity_computation(label_ts,
                                                     fp_raw.info['sfreq'],
                                                     freq_diaps,
                                                     con_method,
                                                     _subject_tree=tree[subject],
                                                     _conditions=conditions_code)
            psd, psd_path = power_spectral_destiny_computation(epo,
                                                               inv,
                                                               lambda2,
                                                               freq_diaps,
                                                               se_method,
                                                               4,
                                                               labels,
                                                               _subject_tree=tree[subject],
                                                               _conditions=conditions_code)
            nodes, nodes_path = nodes_creation(
                labels,
                prepare_connectivity(label_names, con),
                con_method,
                prepare_psd(label_names, psd),
                coords,
                resec_mni,
                _subject_tree=tree[subject],
                _conditions=conditions_code
            )
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
                            con_path,
                            psd_path,
                            nodes_path
                        ])
                    },
                    nodes
                )
            )

        print('All the data has been prepared. Saving the result...')

        write_subjects(subjects_file, subjects)

        print('Successfully saved')

        return subjects
