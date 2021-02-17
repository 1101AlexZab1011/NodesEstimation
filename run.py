import numpy as np
from nodestimation.pipeline import pipeline
import nilearn.plotting as nplt
import matplotlib.pyplot as plt
from nodestimation.project.actions import read

subjects = pipeline(methods=['wpli', 'psd'],
                    freq_bands=[(0.5, 4), (4, 7), (7, 14), (14, 30), (30, 70)])

for subject in subjects:
    print(subject.name)
    fig, ax = plt.subplots(figsize=(15, 15))
    display = nplt.plot_glass_brain(None, display_mode='lyrz', figure=fig, axes=ax, title=subject.name)
    spared = [node.center_coordinates for node in subject.nodes if node.type == 'spared']
    resected = [node.center_coordinates for node in subject.nodes if node.type == 'resected']
    display.add_markers(np.array(spared), marker_color="yellow", marker_size=100)
    display.add_markers(np.array(resected), marker_color="red", marker_size=250)

    if subject.data['resec_mni']:
        resection = read['resec_mni'](subject.data['resec_mni'])
        display.add_markers(resection, marker_color="violet", marker_size=1)
