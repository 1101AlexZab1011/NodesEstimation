import re
import os
import itertools
from nodestimation.project.actions import save, read
from nodestimation.project.structures import file_save_format, file_search_regexps, data_types
import hashlib


def conditions_unique_code(*args):
    # creates code which lets to identify given conditions (whether this condition appears at the first time or computations are already done)

    out = ''
    for arg in args:
        out += str(arg)
    return hashlib.md5(bytes(out, 'utf-8')).hexdigest()


def find_subject_dir(root='./'):
    # Analyses project file structure trying to find a directory containing subjects subdirectories

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
    # computes the size (in bytes) of all contained files

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def add_file_to_tree(regexp, file, subject_tree, type, walk):
    # adds a file matching the search conditions to the project tree

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
    # builds a project tree describing all the required files

    print('Analysing project structure...')
    tree = dict()
    for subject in subject_paths:
        subject_tree = dict()
        path = subject_paths[subject]
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


def check_path(path):
    # creates a directory at the specified path if it does not exist

    if not os.path.isdir(path):
        os.mkdir(path)


def read_file(type, paths, priority):
    if not isinstance(paths, list):
        print('There is only one suitable {} file, trying to read...'.format(type))
        return read[type](paths), paths
    elif priority is None:
        print('There are {} suitable {} files. Reading all...'.format(len(paths), type))
        return [
            (read[type](path), path)
            for path in paths
        ]
    elif isinstance(priority, int) and priority < len(paths):
        print('There are {} suitable {} files. Reading the {}th...'.format(len(paths), type, priority))
        return read[type](paths[priority]), paths[priority]
    elif (isinstance(priority, tuple) or isinstance(priority, list)) and not any([p > len(paths) for p in priority]):
        print('There are {} suitable {} files. Reading {}...'.format(len(paths), type, priority))
        return [
            (read[type](paths[p]), paths[p])
            for p in priority
        ]
    else:
        raise ValueError('Incorrect conditions; type of read files: {}, found {} files of this type, paths to these files: {} and are going to be read: {}'.format(type, len(path), path, priority))


def target_exists(paths, target):
    # determines if a file suitable to the target of a search exists

    if not isinstance(paths, list):
        paths = [paths]
    try:
        out = {
            'any': True,
            'nepf': any(['node_estimation_pipeline_file' in file for file in paths]),
            'original': any(['node_estimation_pipeline_file' not in file for file in paths])
        }[target]

    except KeyError:
        raise ValueError('Unexpected target: {}'.format(target))

    return out


def is_target(path, target):
    # determines if a file is the target of a search
    return {
        'any': True,
        'nepf': 'node_estimation_pipeline_file' in path,
        'original': 'node_estimation_pipeline_file' not in path
    }[target]


def select_suitable_paths(type, paths, target):
    print('Choosing {} {} files...'.format(target, type))
    if not isinstance(paths, list) and is_target(paths, target):
        return paths
    else:
        out = [
            path
            for path in paths
            if is_target(path, target)
        ]
        if len(out) == 1:
            return out[0]
        elif len(out) > 1:
            return out
        else:
            raise ValueError('There are not {} files of type {}'.format(target, type))


def read_target_file(type, paths, target, priority):
    suitable_paths = select_suitable_paths(type, paths, target)
    print('Required files found')
    return read_file(type, suitable_paths, priority)


def read_or_write(type, target='any', read_file=True, write_file=True):
    # if the file with the result of the function exists, then it reads the file, otherwise it executes the function and writes the result to the file
    # possible types given in nodestimation/project/structures.py in data_types list
    # possible targets: 'any' - any found file, 'nepf' - NodeEstimationPipeline File, native program files, 'original' - given source files

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

            if '_priority' in kwargs:
                priority = kwargs['_priority']
            else:
                priority = None

            if read_file:
                print('Looking for {} {} file in files tree...'
                      .format(target, type))
            else:
                print('Skipping a reading step')

            if type in tree \
                    and target_exists(tree[type], target) \
                    and read_file:
                print('{} {} file has been found; trying to read...'
                      .format(target, type).capitalize())
                types_found = True
                try:
                    out = (read_target_file(type, tree[type], target, priority), tree[type])
                    print('Successfully read')

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


def get_ith(func):
    # if wrapped function returns list, it ignores all except priority_ value
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)

        if not isinstance(out, list):
            return out

        elif 'priority_' in kwargs:
            return out[kwargs['priority_']]

        else:
            return out[0]

    return wrapper
