## resection areas

# for subject in subjects:
#     print(subject.name)
#     fig, ax = plt.subplots(figsize=(15, 15))
#     display = nplt.plot_glass_brain(None, display_mode='lyrz', figure=fig, axes=ax, title=subject.name)
#     spared = [node.center_coordinates for node in subject.nodes if node.type == 'spared']
#     resected = [node.center_coordinates for node in subject.nodes if node.type == 'resected']
#
#     if subject.data['resec_mni']:
#         resection = read['resec_mni'](subject.data['resec_mni'])
#         display.add_markers(resection, marker_color="violet", marker_size=1)
#     display.add_markers(np.array(spared), marker_color="yellow", marker_size=100)
#     display.add_markers(np.array(resected), marker_color="red", marker_size=250)
#     # plt.show()


## label time cources comparison

# plt.plot(np.arange(401), label_ts[0][10].T, np.arange(401), label_ts[0][36].T)
# plt.show()
# plt.plot(np.arange(401), label_ts[1][10].T, np.arange(401), label_ts[1][36].T)
# plt.show()
# plt.plot(np.arange(401), label_ts[2][10].T, np.arange(401), label_ts[2][36].T)
# plt.show()
# plt.plot(np.arange(401), label_ts[3][10].T, np.arange(401), label_ts[3][36].T)
# plt.show()


## connectivity representation

# for i in range(67):
#     print((conmat[:, i]))


## nodes strength

# plt.plot(n_strength, 'o')
# plt.title('Node Strength')
# plt.xlabel('node: number')
# plt.ylabel('node: strength')
# plt.show()


## compute roc curve

# resected_nodes = 15
#
# label_ind = np.zeros(len(n_strength))
# label_ind[0:resected_nodes] = True
# label_ind[resected_nodes+1:] = False
# Drs = roc_auc_score(label_ind, n_strength)


## example how to get freesurf_dict

# vertexes = [mne.vertex_to_mni(
#     label.vertices,
#     hemis=0 if label.hemi == 'lh' else 1,
#     subject=subject, subjects_dir=subjects_dir
# )for label in labels]
# freesurf_dict_sample = {l[0].name: np.mean(l[1], axis=0) for l in zip(labels, vertexes)}


## show one label

# nplt.plot_markers(np.zeros(vertexes[0].shape[0]), vertexes[0])
# nplt.show()


## show one node

# nplt.plot_markers(np.array([0, 0]), np.array([
#     np.mean(vertexes[0], axis=0),
#     np.array([1000, 1000, 1000]) ## plot markers does not work with one node
# ]))
# nplt.show()


## show resection + resected nodes + spared nodes

# fig, ax = plt.subplots(figsize=(15,15))
#
# display = nplt.plot_glass_brain(None, display_mode='lyrz', figure=fig, axes=ax)
#
# display.add_markers(resec_coordinates, marker_color="violet", marker_size=1)
#
# display.add_markers(node_coordinates, marker_color="yellow", marker_size=30)

# spared = list()
# resected = list()
#
# for node_coordinate in node_coordinates:
#     for resec_coordinate in resec_coordinates:
#         diff = node_coordinate - resec_coordinate
#         dist = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
#         if dist <= 1 and not node_coordinate in np.array(resected):
#             resected.append(node_coordinate)
#         else:
#             spared.append(node_coordinate)
#
# fig, ax = plt.subplots(figsize=(15,15))
#
#
# display = nplt.plot_glass_brain(
#     None, display_mode='lyrz', figure=fig, axes=ax)
# display.add_markers(resec_coordinates, marker_color="violet", marker_size=1)
# display.add_markers(np.array(spared), marker_color="yellow", marker_size=100)
# display.add_markers(np.array(resected), marker_color="red", marker_size=250)


## pearson and plv nodes

# %%

# if os.path.isfile(res_pearson_nodes_file):
#     print('Reading nodes...')
#     nodes = pickle.load(open(res_pearson_nodes_file, 'rb'))
#
# else:
#     print('PIPELINE: Pearson\'s Nodes file not found, create a new one')
#
#     if not os.path.exists(res_nodes_folder):
#         mkdir(res_nodes_folder)
#
#     nodes = []
#     n_strength, pearson_connectome = nodes_strength(label_ts, 'pearson')
#
#     for i in range(len(n_strength)):
#         nodes.append(Node(label_ts[i, :], n_strength[i], labels[i], 'Pearson', pearson_connectome[i, :]))
#
#     pickle.dump(nodes, open(res_pearson_nodes_file, 'wb'))
#
# coordinates = []
# n_strength = []
# for node in nodes:
#     coordinates.append(node.center_coordinates)
#     n_strength.append(node.strength)
#
# nplt.plot_markers(n_strength, coordinates, node_cmap='black_red_r')
# nplt.show()
#

# %%

# if os.path.isfile(res_plv_nodes_file):
#     print('Reading nodes...')
#     nodes = pickle.load(open(res_plv_nodes_file, 'rb'))
#
# else:
#     print('PIPELINE: PLV Nodes file not found, create a new one')
#
#     if not os.path.exists(res_nodes_folder):
#         mkdir(res_nodes_folder)
#
#     nodes = []
#     n_strength, plv_connectome = nodes_strength(label_ts, 'plv')
#
#     for i in range(len(n_strength)):
#         nodes.append(Node(label_ts[i, :], n_strength[i], labels[i], 'PLV', plv_connectome[i, :, :]))
#
#     pickle.dump(nodes, open(res_plv_nodes_file, 'wb'))
#
# coordinates = []
# n_strength = []
# for node in nodes:
#     coordinates.append(node.center_coordinates)
#     n_strength.append(node.strength)
#
# nplt.plot_markers(n_strength, coordinates, node_cmap='black_red_r')
# nplt.show()


## files tree errors

# if not 'raw' in subject_tree:
#     raise OSError("No one of raw files are found. Raw file must have extension "
#                   "\'.fif\' and contain \'raw\' in its name")
# if not 'resec' in subject_tree and not 'mni-resec' in subject_tree:
#     raise OSError("No one of resection files are found. Resection file must have extension "
#                   "\'.nii\' or  \'.pkl\' and contain \'resec\' in its name")


## pipeline in non-functional state

# conductivity = (0.3,)  # for single layer
# # conductivity = (0.3, 0.006, 0.3)  # for three layers
# epochs_tmin, epochs_tmax = -15, 15
# crop_time = 120
# snr = 0.5  # use SNR smaller than 1 for raw data
# lambda2 = 1.0 / snr ** 2
# method = "sLORETA"
# rfreq = 200
# nfreq = 50
# lfreq = 1
# hfreq = 70
# delta = (0.5, 4)
# theta = (4, 7)
# alpha = (7, 14)
# beta = (14, 30)
# subjects_dir, subjects = path.found_subject_dir()
# tree = path.build_resources_tree(subjects)
# subjects = []
# for subject in tree:
#     raw = read_original_raw('./', _subject_tree=tree[subject])
#     fp_raw = first_processing(raw, lfreq, nfreq, hfreq, rfreq=rfreq, crop=crop_time, _subject_tree=tree[subject])
#     bem = bem_computation(subject, subjects_dir, conductivity, _subject_tree=tree[subject])
#     src = src_computation(subject, subjects_dir, bem, _subject_tree=tree[subject])
#     trans = read_original_trans('./', _subject_tree=tree[subject])
#     fwd = forward_computation(fp_raw, trans, src, bem, _subject_tree=tree[subject])
#     eve = events_computation(fp_raw, range(1, 59), [1 for i in range(58)], _subject_tree=tree[subject])
#     epo = epochs_computation(fp_raw, eve, -1, +1, _subject_tree=tree[subject])
#     cov = noise_covariance_computation(epo, -1, 0, _subject_tree=tree[subject])
#     ave = evokeds_computation(epo, _subject_tree=tree[subject])
#     inv = inverse_computation(epo.info, fwd[0], cov, _subject_tree=tree[subject]) # choosed the first fwd. It should be fixed in future release
#     stc = source_estimation(epo, inv, lambda2, method, _subject_tree=tree[subject])
#     resec = read_original_resec('./', _subject_tree=tree[subject])
#     resec_mni = resection_area_computation(resec, _subject_tree=tree[subject])
#     labels_parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
#     labels_aseg = mne.get_volume_labels_from_src(src[0], subject, subjects_dir) # choosed the first src. It should be fixed in future release
#     labels = labels_parc + labels_aseg # where is label_aseg?
#     parc = parcellation_creating(subject, subjects_dir, labels, _subject_tree=tree[subject])
#     coords = coordinates_computation(subject, subjects_dir, labels,  _subject_tree=tree[subject])
#     label_names = [label.name for label in labels]
#     label_ts = mne.extract_label_time_course(stc, labels, src[0], mode='mean_flip')
#     con = connectivity_computation(label_ts, fp_raw.info['sfreq'], [delta, theta, alpha, beta], ['coh', 'imcoh', 'plv', 'ppc', 'pli'], _subject_tree=tree[subject])
#     psd = power_spectral_destiny_computation(epo, inv, lambda2, [delta, theta, alpha, beta], method, 4, labels, _subject_tree=tree[subject])
#     nodes = nodes_creation(
#         labels,
#         prepare_connectivity(label_names, con),
#         prepare_psd(label_names, psd),
#         coords,
#         resec_mni,
#         _subject_tree=tree[subject]
#     )
#
#     subjects.append(
#         Subject(
#             subject,
#             {
#                 data_type: data
#                 for data_type, data
#                 in zip(subject_data_types, (
#                 raw,
#                 bem,
#                 src,
#                 trans,
#                 fwd,
#                 eve,
#                 epo,
#                 cov,
#                 ave,
#                 inv,
#                 stc,
#                 coords,
#                 resec_mni,
#                 parc,
#                 labels,
#                 con,
#                 psd
#                 ))
#             },
#             nodes
#         )
#     )
