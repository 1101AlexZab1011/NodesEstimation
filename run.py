from nodestimation.pipeline import pipeline


subjects = pipeline(methods=['wpli', 'psd'],
                    freq_bands=[(0.5, 4), (4, 7), (7, 14), (14, 30), (30, 70)])
