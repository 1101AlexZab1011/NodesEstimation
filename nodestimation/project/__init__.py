import re
import os
import itertools
from typing import *
from functools import wraps
from nodestimation.project.actions import save, read
from nodestimation.project.annotations import SubjectTree, ResourcesTree, SubjectTreeData
from nodestimation.project.structures import file_save_format, file_search_regexps, tree_data_types
import hashlib


def conditions_unique_code(*args, **kwargs) -> str:
    """creates code which lets to identify given conditions (whether this condition appears at the first time or computations are already done)

        :param args: any arguments of function to create code
        :type args: Any
        :param kwargs: any key arguments of function to create code
        :type kwargs: Any
        :return: unique code related to given arguments
        :rtype: str
    """

    out = ''
    for arg in args:
        out += str(arg)

    if kwargs:
        for kwarg in kwargs:
            out += str(kwargs[kwarg])

    return hashlib.md5(bytes(out, 'utf-8')).hexdigest()


def find_subject_dir(root: str = './') -> Tuple[str, Dict[str, str]]:
    """Analyses project file structure trying to find a directory containing subjects subdirectories

        :param root: directory where to start searching, default "./"
        :type root: str
        :return: directory containing subjects subdirectories and dictionary with patient`s id as key and path to directory with patient`s information as value
        :rtype: tuple_ of str_ and dict_ of str_ to str_
        :raise OSError: if subject directory not found

        .. _tuple: https://docs.python.org/3/library/stdtypes.html#tuple
        .. _str: https://docs.python.org/3/library/stdtypes.html#str
    """

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


def get_size(start_path: str = './') -> float:
    """computes the size (in bytes) of all contained files

        :param start_path: directory where to start computation, default "./"
        :return: size in bytes
        :rtype: float
    """

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def add_file_to_tree(regexp: str, file: Union[str, List[str]], subject_tree: SubjectTreeData, type: str, walk: tuple) -> None:
    """adds a file matching the search conditions to the project tree

        :param regexp: regular expression to find a file path in a list of paths
        :type regexp: str
        :param file: single path or list of path to build a tree
        :type file: |istr|_ *or* |ilist|_ *of* |istr|_
        :param subject_tree: files tree related to subject to which the new file will be added
        :type subject_tree: |idict|_ *of* |istr|_ *to* |istr|_ *or* |idict|_ *of* |istr|_ *to* |ilist|_ *of* |istr|_
        :param type: `file type`_ to add in the tree
        :type type: str
        :param walk: output of `os.walk <https://docs.python.org/3/library/os.html#os.walk>`_ for current file structure layer
        :type walk: tuple
        :rtype: None

        .. _ifloat: https://docs.python.org/3/library/functions.html#float
        .. _ituple: https://docs.python.org/3/library/stdtypes.html#tuple

        .. |ifloat| replace:: *float*
        .. |ituple| replace:: *tuple*

        .. _`type`:
        .. _`file type`:
        .. _`file types`:
        .. note:: File types:

            :raw: path to file with `Raw <https://mne.tools/stable/generated/mne.io.Raw.html>`_ object, format: `".fif"`_
            :bem: path to file with `ConductorModel <https://mne.tools/stable/generated/mne.bem.ConductorModel.html#mne.bem.ConductorModel>`_ object, format: `".fif"`_
            :src: path to file with `SourceSpaces <https://mne.tools/stable/generated/mne.SourceSpaces.html>`_, format: `".fif"`_
            :trans: path to file with `Transformation <https://mne.tools/stable/generated/mne.transforms.Transform.html#mne.transforms.Transform>`_, format: `".fif"`_
            :fwd: path to file with `Forward Model <https://mne.tools/stable/generated/mne.Forward.html#mne.Forward>`_, format: `".fif"`_
            :eve: path to file with `Data Events <https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray>`_, format: `".pkl"`_
            :epo: path to file with `Epoched Data <https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs>`_, format: `".fif"`_
            :cov: path to file with `Covariance Matrix <https://mne.tools/stable/generated/mne.Covariance.html?highlight=covariance#mne.Covariance>`_, format: `".pkl"`_
            :ave: path to file with `Evoked Data <https://mne.tools/stable/generated/mne.Evoked.html?highlight=evoked#mne.Evoked>`_, format: `".pkl"`_
            :inv: path to file with `InverseOperator <https://mne.tools/stable/generated/mne.minimum_norm.InverseOperator.html#mne.minimum_norm.InverseOperator>`_, format: `".pkl"`_
            :stc: path to file with `SourceEstimate <https://mne.tools/stable/generated/mne.SourceEstimate.html#mne.SourceEstimate>`_ objects, format: `".pkl"`_ (because epoched SourceEstimate stored)
            :resec: path to file with resection in `".nii" <https://nipy.org/nibabel/nibabel_images.html#the-image-object>`_ format
            :resec_txt: path to file with resection in `".txt" <https://en.wikipedia.org/wiki/Text_file>`_ format
            :resec_mni: path to file with resection in mni_ coordinates, format: `".pkl"`_
            :coords: path to file with centers coordinates of `cortical parcellation <https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation>`_ in mni_ coordinates, format: `".pkl"`_
            :feat: path to file with dictionary for all metrics values to all frequency bands to all methods, format: `".pkl"`_
            :nodes: path to file with list of :class:`nodestimation.Node` objects, format: `".pkl"`_
            :dataset: path to file with dataset for all features and frequencies to all nodes, format: `".csv"`_

        .. _".fif": https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#neuromagelektamegin
        .. _".pkl": https://docs.python.org/3/library/pickle.html
        .. _".csv": https://en.wikipedia.org/wiki/Comma-separated_values
        .. _mni: https://brainmap.org/training/BrettTransform.html
    """

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


def build_resources_tree(subject_paths: Dict[str, str]) -> ResourcesTree:
    """builds a project tree describing all the required files

        :param subject_paths: dictionary with subject id as a key and path to its home directory a value
        :type subject_paths: |idict|_ *of* |istr|_ *to* |istr|_
        :return: resources tree with paths to all necessary files (look `file types`_) and with `subject tree metadata`_
        :rtype: dict_ of str_ to str_ or dict_ of str_ to list_ of str_

        .. _dict: https://docs.python.org/3/library/stdtypes.html#dict
        .. _list: https://docs.python.org/3/library/stdtypes.html#list

        .. _`subject tree metadata`:
        .. note:: Subject tree metadata:

            :subject: patient id
            :path: path to patient`s home directory
            :directories: paths to all directories inside patient`s home directory
            :files: paths to all files inside patient`s home directory
            :size: size of patient`s home directory in bytes
    """

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


def check_path(path: str) -> None:
    """creates a directory at the specified path if it does not exist

        :param path: path to check
        :type path: str
        :rtype: None
    """

    if not os.path.isdir(path):
        os.mkdir(path)


def read_files(type: str, paths: Union[str, List[str]], priority: Optional[int] = None) -> Any:
    """reads given file

    :param type: which type_ given file has
    :type type: str
    :param paths: paths to read
    :type paths: |istr|_ *or* |ilist|_ *of* |istr|_
    :param priority: if list of path is given, which path to read. If None, reads all of them, default None
    :type priority: int or None, optional
    :return: file content
    :rtype: any
    :raise ValueError: if given reading conditions are wrong (file not found or not readable)
    """

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


def is_allowed_target(path: str, conditions: str) -> bool:
    """checks if the type is allowed

        :param path: path to checked file
        :type path: str
        :param conditions: conditions_ code for current :func:`nodestimation.pipeline.pipeline` parameters
        :type conditions: str
        :return: True or False for allowed type or not respectively
        :rtype: bool

        .. _`allowed`:
        .. _`allowed type`:
        .. note:: Allowed type means that this file is not generated by this program,
            or created by this program but contains code for the current conditions_ in its path

        .. _conditions:
        .. note:: Pipeline conditions

            This is a code by which there is possible to determine whether required data already exists or not.
            Code created by :func:`nodestimation.project.conditions_unique_code` for arguments given to :func:`nodestimation.pipeline.pipeline`
    """

    if 'nodes_estimation_pipeline_file' in path and conditions not in path:
        return False
    else:
        return True


def target_exists(paths: Union[str, List[str]], target: str, conditions: str) -> bool:
    """determines if a file suitable to the `search target`_ and allowed_ by the conditions_ of a search exists

    :param paths: paths to check
    :type paths: |istr|_ *or* |ilist|_ *of* |istr|_
    :param target: `target`_ to search
    :type target: str
    :param conditions: conditions_ code for current :func:`nodestimation.pipeline.pipeline` parameters
    :type conditions: str
    :return: True if at least one file for allowed_ `search target`_ exists, otherwise False
    :rtype: bool
    :raise ValueError: if `search target` unknown
    """

    if not isinstance(paths, list):
        paths = [paths]
    try:
        out = {
            'any': any([is_allowed_target(file, conditions) for file in paths]),
            'nepf': any(['nodes_estimation_pipeline_file' in file and is_allowed_target(file, conditions) for file in paths]),
            'original': any(['nodes_estimation_pipeline_file' not in file for file in paths])
        }[target]

    except KeyError:
        raise ValueError('Unexpected search_target: {}'.format(target))

    return out


def is_target(path: str, target: str, conditions: str) -> bool:
    """determines if a file is a target_ of a search

        :param path: path to checked file
        :type path: str
        :param target: target_ of a search
        :type target: str
        :param conditions: conditions code for current :func:`nodestimation.pipeline.pipeline` parameters
        :type conditions: str
        :return: True or False whether current file is a target of search or not respectively
        :rtype: bool

        .. _`search target`:
        .. _target:
        .. note:: Target related to a way file was created

            :any: all found files
            :nepf: only files created in context of :func:`nodestimation.pipeline.pipeline` function (literally means: 'NodesEstimation Pipeline File')
            :original: all files created outside :func:`nodestimation.pipeline.pipeline` function supposed to be original
    """
    return {
        'any': is_allowed_target(path, conditions),
        'nepf': 'nodes_estimation_pipeline_file' in path and is_allowed_target(path, conditions),
        'original': 'nodes_estimation_pipeline_file' not in path
    }[target]


def select_suitable_paths(type: str, paths: Union[str, List[str]], target: str, conditions: str) -> Union[str, List[str]]:
    """selects files for allowed_ `search target`_

        :param type: `file type`_ to read
        :type type: str
        :param paths: paths to check
        :type paths: |istr|_ *or* |ilist|_ *of* |istr|_
        :param target: target_ of a search
        :type target: str
        :param conditions: conditions code for current :func:`nodestimation.pipeline.pipeline` parameters
        :type conditions: str
        :return: required paths
        :rtype: str_ or list_ of str_
        :raise ValueError: if files matching to allowed_ `search target`_ not found
    """

    print('Choosing {} {} files...'.format(target, type))
    if not isinstance(paths, list) and is_target(paths, target, conditions):
        print('Required file found')
        return paths
    else:
        out = [
            path
            for path in paths
            if is_target(path, target, conditions)
        ]
        if len(out) == 1:
            print('Required file found')
            return out[0]
        elif len(out) > 1:
            print('Required files found')
            return out
        else:
            raise ValueError('There are not {} files of type {}'.format(target, type))


def read_or_write(type: str, search_target: str = 'any', read_file: bool = True, write_file: bool = True, main_arg_indexes: Union[int, List[int]] = 0):
    """decorator function, if the file with the result of wrapped function exists,
        then it reads this file, otherwise it executes wrapped function and writes the result to the file

        :param type: supposed type_ of wrapped function output
        :type type: str
        :param search_target: target_ of a search
        :type search_target: str
        :param read_file: if False, it does not try to find and read existing file with wrapped function output, default True
        :type read_file: bool
        :param write_file:  if False, it does not try to store wrapped function output, default True
        :type write_file: bool
        :param main_arg_indexes: indexes of necessary arguments for wrapped function without which it does not work.
            If some are not specified it does not call wrapped function and returns None, default 0
        :type main_arg_indexes: |iint|_ or |ilist|_ of |iint|_
        :return: wrapped function result and where it was read/written
        :rtype: tuple_ of str_ or None_ and any or None_
        :raise OSError: if subject tree not found, or given conditions to read file incorrect

        .. _iint: https://docs.python.org/3/library/functions.html#int
        .. _None: https://docs.python.org/3/library/constants.html#None

        .. |iint| replace:: *int*
    """

    if not isinstance(main_arg_indexes, list):
        main_arg_indexes = [main_arg_indexes]

    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args, **kwargs):

            main_args = [args[i] for i in main_arg_indexes]

            out = None
            types_found = False
            home = 'NodesEstimationFiles'

            if '_subject_tree' in kwargs:
                meta, tree = kwargs['_subject_tree']
            else:
                raise OSError('{}: subject_tree parameter is lost'.format(func.__name__))

            check_path(
                os.path.join(meta['path'], home)
            )
            conditions = conditions_unique_code(*args, **kwargs)

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
                    and target_exists(tree[type], search_target, conditions) \
                    and read_file:
                print('The {} {} file has been found; trying to read...'
                      .format(search_target, type))
                types_found = True
                try:
                    out = read_files(
                        type,
                        select_suitable_paths(type, tree[type], search_target, conditions),
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
                    if search_target == 'original':
                        path_to_file = os.path.join(
                            meta['path'],
                            home,
                            meta['subject'] +
                            '_' +
                            func.__name__ +
                            '_' +
                            type +
                            '.' +
                            file_save_format[type]
                        )
                    else:
                        path_to_file = os.path.join(
                            meta['path'],
                            home,
                            meta['subject'] +
                            '_nodes_estimation_pipeline_file_' +
                            func.__name__ +
                            '_' +
                            type +
                            '_' +
                            conditions +
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
