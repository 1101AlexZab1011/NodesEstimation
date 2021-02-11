import re
import os
import itertools
from typing import *

from nodestimation.project.actions import save, read
from nodestimation.project.structures import file_save_format, file_search_regexps, tree_data_types
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
    if subjects_found:
        return subjects_dir, \
               {
                   subject: os.path.join(subjects_dir, subject) for subject in next(os.walk(subjects_dir))[1]
               }
    else:
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
                for type in tree_data_types:
                    add_file_to_tree(file_search_regexps[type], file, subject_tree, type, walk)

        meta = {
                'subject': subject,
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


def read_files(type, paths, priority):
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
        raise ValueError('Incorrect conditions; type of read files: {}, found {} files of this type, paths to these files: {} and are going to be read: {}'.format(type, len(paths), paths, priority))


def target_exists(paths, target):
    # determines if a file suitable to the search_target of a search exists

    if not isinstance(paths, list):
        paths = [paths]
    try:
        out = {
            'any': True,
            'nepf': any(['node_estimation_pipeline_file' in file for file in paths]),
            'original': any(['node_estimation_pipeline_file' not in file for file in paths])
        }[target]

    except KeyError:
        raise ValueError('Unexpected search_target: {}'.format(target))

    return out


def is_target(path, target):
    # determines if a file is the search_target of a search
    return {
        'any': True,
        'nepf': 'node_estimation_pipeline_file' in path,
        'original': 'node_estimation_pipeline_file' not in path
    }[target]


def select_suitable_paths(type, paths, target):
    print('Choosing {} {} files...'.format(target, type))
    if not isinstance(paths, list) and is_target(paths, target):
        print('Required files found')
        return paths
    else:
        out = [
            path
            for path in paths
            if is_target(path, target)
        ]
        if len(out) == 1:
            print('Required files found')
            return out[0]
        elif len(out) > 1:
            print('Required files found')
            return out
        else:
            raise ValueError('There are not {} files of type {}'.format(target, type))


def read_or_write(type, search_target='any', read_file=True, write_file=True, main_arg_indexes=0):
    # if the file with the result of the function exists, then it reads the file, otherwise it executes the function and writes the result to the file
    # possible types given in nodestimation/project/structures.py in data_types list
    # possible targets: 'any' - any found file, 'nepf' - NodeEstimationPipeline File, native program files, 'original' - given source files

    if not isinstance(main_arg_indexes, list):
        main_arg_indexes = [main_arg_indexes]

    def decorator(func):

        def wrapper(*args, **kwargs):

            main_args = [args[i] for i in main_arg_indexes]

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
                      .format(search_target, type))
            else:
                print('Skipping the reading step')

            if type in tree \
                    and target_exists(tree[type], search_target) \
                    and read_file:
                print('The {} {} file has been found; trying to read...'
                      .format(search_target, type))
                types_found = True
                try:
                    out = read_files(
                        type,
                        select_suitable_paths(type, tree[type], search_target),
                        priority
                    )
                    print('Successfully read')

                except OSError:
                    print('Incorrect reading conditions: '
                          '\n\tReading type: {}, '
                          '\n\tReading search_target: {}, '
                          '\n\tReading function: {}'
                          '\n\t Reading conditions: {}'
                          .format(type, search_target, func.__name__, kwargs['_conditions']))

            if not types_found:
                if read_file:
                    print('The {} {} file has not been found'
                          .format(search_target, type))
                if isinstance(search_target, str) and write_file:
                    print('Creating a new {} {} file'.format(search_target, type))

                if write_file and all([arg is not None for arg in main_args]):
                    out = func(*args, **kwargs)
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
                else:
                    out = (None, None)

            return out

        return wrapper

    return decorator

