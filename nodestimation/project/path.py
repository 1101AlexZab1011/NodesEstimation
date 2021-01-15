import re
import os
import itertools
from nodestimation.project.actions import save, read, save_format


def found_subject_dir(root='./'):
    subjects_dir = None
    subjects_found = False
    for walk in os.walk(os.path.join(root, 'Source')):
        for subdir in walk[1]:
            if subdir == 'Subjects' or subdir == 'subjects' and not subjects_found:
                subjects_found = True
                subjects_dir = os.path.join(walk[0], subdir)
            elif subdir == 'Subjects' or subdir == 'subjects' and subjects_found:
                raise OSError("There are two subjects directories: {}, {}; Only one must be".format(
                    subjects_dir, os.path.join(walk[0], subdir)
                ))
            return subjects_dir, \
                   {
                       subject: os.path.join(subjects_dir, subject) for subject in next(os.walk(subjects_dir))[1]
                   }
    if not subjects_found:
        raise OSError("Subjects directory not found!")


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def add_file_to_tree(regular, file, subject_tree, type, walk):
    if isinstance(regular, list):
        found = any(re.search(reg, file) for reg in regular)
    else:
        found = re.search(regular, file)
    if found:
        if type in subject_tree.keys():
            if isinstance(subject_tree[type], list):
                subject_tree[type].append(os.path.join(walk[0], file))
                subject_tree[type] = sorted(subject_tree[type])
            else:
                subject_tree.update({type: sorted([subject_tree[type], os.path.join(walk[0], file)])})
        else:
            subject_tree.update({type: os.path.join(walk[0], file)})
            print('\t\t' + '{} file: ok'.format(type).capitalize())


def build_resources_tree(subject_paths):
    print('Analysing project structure...')
    tree = dict()
    for subject in subject_paths:
        subject_tree = dict()
        path = subject_paths[subject]
        size = 0
        print('\tSubject: ', subject)
        subdirs = list()
        files = list()
        for walk in os.walk(path):
            subdirs = itertools.chain(subdirs, [os.path.join(walk[0], subdir) for subdir in walk[1]])
            files = itertools.chain(files, [os.path.join(walk[0], sfile) for sfile in walk[2]])
            for file in walk[2]:
                add_file_to_tree(r'.*raw.*\.fif', file, subject_tree, 'raw', walk)
                add_file_to_tree([r'.*src.*\.fif', r'.*source_space.*\.fif'], file, subject_tree, 'src', walk)
                add_file_to_tree(r'.*bem.*\.fif', file, subject_tree, 'bem', walk)
                add_file_to_tree(r'.*trans.*\.fif', file, subject_tree, 'trans', walk)
                add_file_to_tree(r'.*cov.*\.fif', file, subject_tree, 'cov', walk)
                add_file_to_tree([r'.*fwd.*\.fif', r'.*forward.*\.fif'], file, subject_tree, 'fwd', walk)
                add_file_to_tree([r'.*inv.*\.fif', r'.*inverse.*\.fif'], file, subject_tree, 'inv', walk)
                add_file_to_tree([r'.*eve.*', r'.*events.*'], file, subject_tree, 'eve', walk)
                add_file_to_tree(r'.*epo.*\.fif', file, subject_tree, 'epo', walk)
                add_file_to_tree(r'.*ave.*\.fif', file, subject_tree, 'ave', walk)
                add_file_to_tree([r'.*stc.*\.fif', r'.*stc.*\.pkl'], file, subject_tree, 'stc', walk)
                add_file_to_tree(r'.*resec.*\.nii.*', file, subject_tree, 'resec', walk)
                add_file_to_tree(r'.*resec.*\.pkl.*', file, subject_tree, 'resec-mni', walk)

        meta = {'subject': subject,
                'path': path,
                'directories': list(subdirs),
                'files': list(files),
                'size': get_size(path)
                }
        tree.update({subject: (meta, subject_tree)})
        print('\tFiles structure for {} has been analysed. Files tree is built'.format(subject))

    return tree


def is_target(file, target):
    if isinstance(target, int):
        num = target
    else:
        num = 0
    try:
        if isinstance(file, list):
            out = {
                'any': True,
                'nepf': any(['node_estimation_pipeline_file' in f for f in file]),
                'original': any(['node_estimation_pipeline_file' not in f for f in file]),
                num: bool(file[num])
            }[target]
        else:
            out = {
                'any': True,
                'nepf': 'node_estimation_pipeline_file' in file,
                'original': 'node_estimation_pipeline_file' not in file,
                num: False
            }[target]

    except KeyError:
        raise ValueError('Unexpected target: {}'.format(target))

    return out


def read_or_write(type, target='any', read_file=True, write_file=True):
    def decorator(func):

        def wrapper(*args, **kwargs):

            out = None

            if '_subject_tree' in kwargs:
                meta, tree = kwargs['_subject_tree']
            else:
                raise OSError('{}: subject_tree parameter is lost'.format(func.__name__))
            if read_file:
                print('Looking for {} {} file in files tree...'
                      .format(target, type))
            else:
                print('Skipping a reading step')
            if type in tree \
                    and is_target(tree[type], target) \
                    and read_file:
                print('{} {} file has been found. Trying to read...'
                      .format(target, type).capitalize())
                try:
                    if isinstance(tree[type], list) \
                            and isinstance(target, int):
                        print('{} 4 files found: {}. Only {}th file is going to be read'
                              .format(len(tree[type]), tree[type], target))
                        out = read[type](tree[type][target])
                    elif isinstance(tree[type], list) \
                            and target == 'any':
                        print('{} files found: {}. All the files are going to be read'
                              .format(len(tree[type]), tree[type]))
                        out = [
                            read[type](tree[type][i])
                            for i in range(len(tree[type]))
                        ]
                        if len(out) == 1:
                            out = out[0]
                    elif isinstance(tree[type], list) \
                            and target == 'nepf':
                        print('{} files found: {}. Only the native files are going to be read'
                              .format(len(tree[type]), tree[type]))
                        out = [
                            read[type](tree[type][i])
                            for i in range(len(tree[type]))
                            if 'node_estimation_pipeline_file' in tree[type][i]
                        ]
                        if len(out) == 1:
                            out = out[0]
                    elif isinstance(tree[type], list) \
                            and target == 'original':
                        print('{} files found: {}. Only the original files are going to be read'
                              .format(len(tree[type]), tree[type]))
                        out = [
                            read[type](tree[type][i])
                            for i in range(len(tree[type]))
                            if 'node_estimation_pipeline_file' not in tree[type][i]
                        ]
                        if len(out) == 1:
                            out = out[0]
                    elif not isinstance(tree[type], list) \
                            and (
                            (target == 'nepf' and 'node_estimation_pipeline_file' in tree[type])
                            or (target == 'original' and 'node_estimation_pipeline_file' not in tree[type])
                    ):
                        out = read[type](tree[type])

                except OSError:
                    print('Incorrect reading conditions: '
                          '\n\tReading type: {}, '
                          '\n\tReading target: {}, '
                          '\n\tReading function: {}'
                          .format(type, target, func.__name__))
                print('Successfully read')
            else:
                if read_file:
                    print('The {} {} file has not been found'
                          .format(target, type))
                if not isinstance(target, int):
                    print('Creating a new {} {} file'.format(target, type))
                else:
                    raise OSError('Incorrect writing conditions: '
                                  '\n\tReading type: {}, '
                                  '\n\tReading target: {}, '
                                  '\n\tReading function: {}'
                                  .format(type, target, func.__name__))
                out = func(*args, **kwargs)
                if write_file:
                    path_to_file = os.path.join(meta['path'],
                                                meta['subject'] +
                                                '_node_estimation_pipeline_file_' +
                                                func.__name__ +
                                                '_output_' +
                                                type +
                                                '.' +
                                                save_format[type]
                                                )
                    save[type](path_to_file, out)
                    print('Done. Path to new file: {}'
                          .format(path_to_file))
                    del path_to_file

            return out

        return wrapper

    return decorator
