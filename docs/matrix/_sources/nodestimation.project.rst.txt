nodestimation.project package
=============================

Package containing the structural units of the project and providing functionality for their creation and modification

Submodules
----------

nodestimation.project.actions module
------------------------------------

Functions for storing and reading required files

.. list-table:: Actions
   :widths: 15 25 25
   :header-rows: 1

   * - File type
     - Reader
     - Writer
   * - raw
     - `mne.io.read_raw_fif <https://mne.tools/stable/generated/mne.io.read_raw_fif.html>`_
     - lambda_ path, raw: `raw.save(path) <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.save>`_
   * - bem
     - `mne.read_bem_solution <https://mne.tools/stable/generated/mne.read_bem_solution.html?highlight=read_bem#mne.read_bem_solution>`_
     - `mne.write_bem_solution <https://mne.tools/stable/generated/mne.write_bem_solution.html?highlight=write_bem#mne.write_bem_solution>`_
   * - src
     - `mne.read_source_spaces <https://mne.tools/stable/generated/mne.read_source_spaces.html?highlight=read_source#mne.read_source_spaces>`_
     - lambda_ path, src: `src.save(path) <https://mne.tools/stable/generated/mne.SourceSpaces.html#mne.SourceSpaces.save>`_
   * - trans
     - `mne.read_trans <https://mne.tools/stable/generated/mne.read_trans.html?highlight=read_trans#mne.read_trans>`_
     - `mne.write_trans <https://mne.tools/stable/generated/mne.write_trans.html?highlight=write_trans#mne.write_trans>`_
   * - fwd
     - `mne.read_forward_solution <https://mne.tools/stable/generated/mne.read_forward_solution.html?highlight=read_forward#mne.read_forward_solution>`_
     - `mne.write_forward_solution <https://mne.tools/stable/generated/mne.write_forward_solution.html?highlight=write_forward#mne.write_forward_solution>`_
   * - eve
     - `mne.read_events <https://mne.tools/stable/generated/mne.read_events.html?highlight=read_events>`_
     - `mne.write_events <https://mne.tools/stable/generated/mne.write_events.html?highlight=write_events>`_
   * - epo
     - `mne.read_epochs <https://mne.tools/stable/generated/mne.read_epochs.html?highlight=read_epochs#mne.read_epochs>`_
     - lambda_ path, epochs: `epochs.save(path) <https://mne.tools/stable/generated/mne.Epochs.html?highlight=epochs#mne.Epochs.save>`_
   * - cov
     - `mne.read_cov <https://mne.tools/stable/generated/mne.read_cov.html?highlight=read_cov#mne.read_cov>`_
     - `mne.write_cov <https://mne.tools/stable/generated/mne.write_cov.html?highlight=write_cov#mne.write_cov>`_
   * - ave
     - `mne.read_evokeds <https://mne.tools/stable/generated/mne.read_evokeds.html?highlight=read_evokeds#mne.read_evokeds>`_
     - `mne.write_evokeds <https://mne.tools/stable/generated/mne.write_evokeds.html?highlight=write_evokeds#mne.write_evokeds>`_
   * - inv
     - `mne.minimum_norm.read_inverse_operator <https://mne.tools/stable/generated/mne.minimum_norm.read_inverse_operator.html?highlight=mne%20minimum_norm%20read_inverse_operator#mne.minimum_norm.read_inverse_operator>`_
     - `mne.minimum_norm.write_inverse_operator <https://mne.tools/stable/generated/mne.minimum_norm.write_inverse_operator.html?highlight=mne%20minimum_norm%20write_inverse_operator#mne.minimum_norm.write_inverse_operator>`_
   * - stc
     - lambda_ path: `pickle.load(open(path, 'rb')) <https://docs.python.org/3/library/pickle.html#pickle.load>`_
     - lambda_ path, stc: `pickle.dump(stc, open(path, 'wb')) <https://docs.python.org/3/library/pickle.html#pickle.dump>`_
   * - coords
     - lambda_ path: `pickle.load(open(path, 'rb')) <https://docs.python.org/3/library/pickle.html#pickle.load>`_
     - lambda_ path, coord: `pickle.dump(coord, open(path, 'wb')) <https://docs.python.org/3/library/pickle.html#pickle.dump>`_
   * - resec
     - `nibabel.load <https://nipy.org/nibabel/reference/nibabel.loadsave.html#load>`_
     - `nibabel.save <https://nipy.org/nibabel/reference/nibabel.loadsave.html#save>`_
   * - resec_mni
     - lambda_ path: `pickle.load(open(path, 'rb')) <https://docs.python.org/3/library/pickle.html#pickle.load>`_
     - lambda_ path, resec: `pickle.dump(resec, open(path, 'wb')) <https://docs.python.org/3/library/pickle.html#pickle.dump>`_
   * - resec_txt
     - lambda_ path: `open(path, 'r').read() <https://docs.python.org/3/library/functions.html#open>`_
     - lambda_ path, resec: `open(path, 'w').write(resec) <https://docs.python.org/3/library/functions.html#open>`_
   * - feat
     - lambda_ path: `pickle.load(open(path, 'rb')) <https://docs.python.org/3/library/pickle.html#pickle.load>`_
     - lambda_ path, feat: `pickle.dump(feat, open(path, 'wb')) <https://docs.python.org/3/library/pickle.html#pickle.dump>`_
   * - nodes
     - lambda_ path: `pickle.load(open(path, 'rb')) <https://docs.python.org/3/library/pickle.html#pickle.load>`_
     - lambda_ path, nodes: `pickle.dump(nodes, open(path, 'wb')) <https://docs.python.org/3/library/pickle.html#pickle.dump>`_
   * - dataset
     - lambda_ path: `pickle.load(open(path, 'rb')) <https://docs.python.org/3/library/pickle.html#pickle.load>`_
     - lambda_ path, data: `pickle.dump(data, open(path, 'wb')) <https://docs.python.org/3/library/pickle.html#pickle.dump>`_

.. _lambda: https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions

.. automodule:: nodestimation.project.actions
    :members:
    :undoc-members:
    :show-inheritance:

nodestimation.project.annotations module
----------------------------------------

This module provides `type aliases <https://docs.python.org/3/library/typing.html#type-aliases>`_ used in the NodesEstimation project

Annotations used in this package::

        SubjectTreeData = Dict[
        str, Union[
            str, List[str]
        ]
    ]

    SubjectTreeMetaData = Dict[
        str, Union[float, str, List[str]]
    ]

    SubjectTree = Tuple[
        SubjectTreeMetaData,
        SubjectTreeData
    ]

    ResourcesTree = Dict[
        str, SubjectTree
    ]

    Features = Dict[
        str, Dict[
            str, np.ndarray
        ]
    ]

    LabelsFeatures = Dict[
        str, Dict[
            str, Dict[
                str, Dict[
                    str, float
                ]
            ]
        ]
    ]

    NodeFeatures = Dict[
                str, Dict[
                    str, Dict[
                        str, float
                    ]
                ]
            ]

.. note:: Annotations module uses type hints

    `Type hinting <https://www.python.org/dev/peps/pep-0484/>`_ is a formal solution to statically indicate the type of a value within python code.

    Used type hints:

    * |thtuple|_
    * |thlist|_
    * |thdict|_
    * `Union <https://docs.python.org/3/library/typing.html#typing.Union>`_

.. _thtuple: https://docs.python.org/3/library/typing.html#typing.Tuple
.. _thlist: https://docs.python.org/3/library/typing.html#typing.List
.. _thdict: https://docs.python.org/3/library/typing.html#typing.Dict

.. |thtuple| replace:: Tuple
.. |thlist| replace:: List
.. |thdict| replace:: Dict


.. automodule:: nodestimation.project.annotations
    :members:
    :undoc-members:
    :show-inheritance:

nodestimation.project.structures module
---------------------------------------

This module provides structures to help understand what types of files exist, how to search, distinguish, and in what format to store them

.. list-table:: Structures
   :widths: 15 25 25
   :header-rows: 1

   * - Data type
     - `RegExp <https://docs.python.org/3/library/re.html>`_
     - File format
   * - raw
     - ``r'.*raw.*\.fif'``
     - `".fif" <https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#neuromagelektamegin>`_
   * - bem
     - ``r'.*bem.*\.fif'``
     - `".fif" <https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#neuromagelektamegin>`_
   * - src
     - ``[r'.*src.*\.fif', r'.*source_space.*\.fif', r'.*source-space.*\.fif']``
     - `".fif" <https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#neuromagelektamegin>`_
   * - trans
     - ``r'.*trans.*\.fif'``
     - `".fif" <https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#neuromagelektamegin>`_
   * - fwd
     - ``[r'.*fwd.*\.fif', r'.*forward.*\.fif']``
     - `".fif" <https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#neuromagelektamegin>`_
   * - eve
     - ``r'.*eve.*'``
     - `".fif" <https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#neuromagelektamegin>`_
   * - epo
     - ``r'.*epo.*'``
     - `".fif" <https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#neuromagelektamegin>`_
   * - cov
     - ``r'.*cov.*\.fif'``
     - `".fif" <https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#neuromagelektamegin>`_
   * - ave
     - ``[r'.*ave.*\.fif', r'.*evoked.*\.fif']``
     - `".fif" <https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#neuromagelektamegin>`_
   * - inv
     - ``r'.*inv.*\.fif'``
     - `".fif" <https://bids-specification.readthedocs.io/en/stable/99-appendices/06-meg-file-formats.html#neuromagelektamegin>`_
   * - stc
     - ``[r'.*stc.*\.fif', r'.*stc.*\.pkl']``
     - `".pkl" <https://docs.python.org/3/library/pickle.html>`_
   * - coords
     - ``[r'.*coord.*\.pkl', r'.*coordinate.*\.pkl']``
     - `".pkl" <https://docs.python.org/3/library/pickle.html>`_
   * - resec
     - ``r'.*resec.*\.nii.*'``
     - `".nii" <https://nipy.org/nibabel/nibabel_images.html#the-image-object>`_
   * - resec_mni
     - ``r'.*resec.*\.pkl.*'``
     - `".pkl" <https://docs.python.org/3/library/pickle.html>`_
   * - resec_txt
     - ``r'.*resec.*\.txt.*'``
     - `".txt" <https://en.wikipedia.org/wiki/Text_file>`_
   * - feat
     - ``r'.*feat.*\.pkl.*'``
     - `".pkl" <https://docs.python.org/3/library/pickle.html>`_
   * - nodes
     - ``r'.*nodes.*\.pkl.*'``
     - `".pkl" <https://docs.python.org/3/library/pickle.html>`_
   * - dataset
     - ``r'.*dataset.*\.csv.*'``
     - `".csv" <https://en.wikipedia.org/wiki/Comma-separated_values>`_

.. automodule:: nodestimation.project.structures
    :members:
    :undoc-members:
    :show-inheritance:

nodestimation.project.subject module
------------------------------------

This module provides the main structural unit of NodesEstimation project. This unit represents a patient

.. automodule:: nodestimation.project.subject
    :members:
    :undoc-members:
    :show-inheritance:


Module contents
---------------

.. automodule:: nodestimation.project
    :members:
    :undoc-members:
    :show-inheritance:
