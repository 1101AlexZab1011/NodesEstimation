import os
import re
from nodestimation import project

if __name__ == "__main__":

    subjects_dir, subjects_ = project.find_subject_dir()

    count = 0

    for root, dirs, files in os.walk('./'):
        for file in files:
            if any(re.search(reg, file) for reg in [r'.*subjects_information_for_.*\.pkl', r'.*subject_information_for_.*\.pkl']):
                os.remove(file)
                count += 1

    print(f'Cleanup completed, {count} files were deleted in total.')
