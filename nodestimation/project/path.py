import re
import os
import itertools
from nodestimation.project.actions import save, read
from nodestimation.project.structures import file_save_format, file_search_regexps, data_types


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
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def add_file_to_tree(regexp, file, subject_tree, type, walk):
    if isinstance(regexp, list):
        found = any(re.search(reg, file) for reg in regexp)
    else:
        found = re.search(regexp, file)
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
                for type in data_types:
                    add_file_to_tree(file_search_regexps[type], file, subject_tree, type, walk)

        meta = {'subject': subject,
                'path': path,
                'directories': list(subdirs),
                'files': list(files),
                'size': get_size(path)
                }
        tree.update({subject: (meta, subject_tree)})
        print('\tFiles structure for {} has been analysed. Files tree is built'.format(subject))

    return tree


def is_target(files, target):
    if isinstance(target, int):
        num = target
    else:
        num = 0
    if not isinstance(files, list):
        files = [files]
    try:
        out = {
            'any': True,
            'nepf': any(['node_estimation_pipeline_file' in file for file in files]),
            'original': any(['node_estimation_pipeline_file' not in file for file in files]),
            num: num < len(files) or num == 0
        }[target]

    except KeyError:
        raise ValueError('Unexpected target: {}'.format(target))

    return out


def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def read_or_write(type, target='any', read_file=True, write_file=True):
    def decorator(func):

        def wrapper(*args, **kwargs):

            out = None

            types_found = False

            if '_subject_tree' in kwargs:
                meta, tree = kwargs['_subject_tree']
            else:
                raise OSError('{}: subject_tree parameter is lost'.format(func.__name__))

            if kwargs['_conditions'] is not None:
                conditions = kwargs['_conditions']
                check_path(
                    os.path.join(meta['path'], conditions)
                )
            else:
                conditions = ''

            if read_file:
                print('Looking for {} {} file in files tree...'
                      .format(target, type))
            else:
                print('Skipping a reading step')

            if type in tree \
                    and is_target(tree[type], target) \
                    and read_file:
                print('{} {} file has been found; trying to read...'
                      .format(target, type).capitalize())
                types_found = True
                try:
                    if isinstance(tree[type], list) \
                            and isinstance(target, int) \
                            and any([conditions in sample for sample in tree[type]]):
                        print('{} files found: {}. Only {}th file in the {} folder is going to be read'
                              .format(len(tree[type]), tree[type], target, conditions))
                        to_read = [sample for sample in tree[type] if conditions in sample]
                        out = (read[type](to_read[target]), to_read[target])
                        print('Successfully read')
                    elif isinstance(tree[type], list) \
                            and target == 'any'\
                            and any([conditions in sample for sample in tree[type]]):
                        print('{} files found: {}. All the files in {} are going to be read'
                              .format(len(tree[type]), tree[type], conditions))
                        out = ([
                            read[type](tree[type][i])
                            for i in range(len(tree[type]))
                            if conditions in tree[type][i]
                        ], [
                            tree[type][i]
                            for i in range(len(tree[type]))
                            if conditions in tree[type][i]
                        ])
                        if len(out[0]) == 1:
                            out = (out[0][0], out[1][0])
                        print('Successfully read')
                    elif isinstance(tree[type], list) \
                            and target == 'nepf' \
                            and any([conditions in sample for sample in tree[type]]):
                        print('{} files found: {}. Only the native files in {} are going to be read'
                              .format(len(tree[type]), tree[type], conditions))
                        out = ([
                            read[type](tree[type][i])
                            for i in range(len(tree[type]))
                            if 'node_estimation_pipeline_file' in tree[type][i]
                               and conditions in tree[type][i]
                        ], [
                            tree[type][i]
                            for i in range(len(tree[type]))
                            if conditions in tree[type][i]
                        ])
                        if len(out[0]) == 1:
                            out = (out[0][0], out[1][0])
                        print('Successfully read')
                    elif isinstance(tree[type], list) \
                            and target == 'original' \
                            and any([conditions in sample for sample in tree[type]]):
                        print('{} files found: {}. Only the original files are going to be read'
                              .format(len(tree[type]), tree[type]))
                        out = ([
                            read[type](tree[type][i])
                            for i in range(len(tree[type]))
                            if 'node_estimation_pipeline_file' not in tree[type][i]
                               and conditions in tree[type][i]
                        ], [
                            tree[type][i]
                            for i in range(len(tree[type]))
                            if conditions in tree[type][i]
                        ])
                        if len(out[0]) == 1:
                            out = (out[0][0], out[1][0])
                        print('Successfully read')
                    elif not isinstance(tree[type], list) \
                            and conditions in tree[type] \
                            and (
                            (target == 'nepf' and 'node_estimation_pipeline_file' in tree[type])
                            or (target == 'original' and 'node_estimation_pipeline_file' not in tree[type])
                            or target == 'any'
                    ):
                        out = (read[type](tree[type]), tree[type])
                        print('Successfully read')
                    else:
                        print('All found files do not meet the specified conditions')
                        types_found = False

                except OSError:
                    print('Incorrect reading conditions: '
                          '\n\tReading type: {}, '
                          '\n\tReading target: {}, '
                          '\n\tReading function: {}'
                          '\n\t Reading conditions: {}'
                          .format(type, target, func.__name__, kwargs['_conditions']))
            if not types_found:
                if read_file:
                    print('The {} {} file has not been found'
                          .format(target, type))
                if not isinstance(target, int):
                    print('Creating a new {} {} file'.format(target, type))
                else:
                    raise OSError('Incorrect writing conditions: '
                                  '\n\tWriting type: {}, '
                                  '\n\tWriting target: {}, '
                                  '\n\tWriting function: {}'
                                  '\n\t Writing conditions: {}'
                                  .format(type, target, func.__name__, kwargs['_conditions']))
                out = func(*args, **kwargs)
                if write_file:
                    path_to_file = os.path.join(
                        meta['path'],
                        conditions,
                        meta['subject'] +
                        '_node_estimation_pipeline_file_' +
                        func.__name__ +
                        '_output_' +
                        type +
                        '.' +
                        file_save_format[type]
                    )
                    save[type](path_to_file, out)
                    print('Done. Path to new file: {}'
                          .format(path_to_file))
                    out = (out, path_to_file)

            return out

        return wrapper

    return decorator
