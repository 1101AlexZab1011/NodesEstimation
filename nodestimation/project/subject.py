from typing import *

import pandas as pd

from nodestimation import Node
from pandas import DataFrame

from nodestimation.project.annotations import Connectomes


class Subject(object):
    """Class containing all the necessary information about the patient

    :param name: patient`s ID
    :type name: str
    :param data: a dictionary containing paths to patient data files used in calculations.
        Keys are `data types`_, values are paths to files with this data (or None if they do not exist)
    :type data: |idict|_ *of* |istr|_ *to* |istr|_
    :param nodes: list of :class:`nodestimation.Node` objects related to this patient
    :type nodes: |ilist|_ *of* |Node|
    :param directory: patient home directory (must start with ``'./Source/Subjects/'``)
    :type directory: str
    :param datasets: dataset with all features and frequencies to all nodes
    :type datasets: |idict|_ *of* |istr|_ *to* |pandas.DataFrame|_
    :param connectomes: matrix representations for all features and frequencies to all nodes
    :type connectomes: |idict|_ *of* |istr|_ *to* |idict|_ *of* |istr|_ *to* |pandas.DataFrame|_

    .. _ilist: https://docs.python.org/3/library/stdtypes.html#list
    .. _idict: https://docs.python.org/3/library/stdtypes.html#dict
    .. _istr: https://docs.python.org/3/library/stdtypes.html#str
    .. _pandas.DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    .. _inx.Graph: https://networkx.org/documentation/stable/reference/classes/graph.html#networkx.Graph

    .. |ilist| replace:: *list*
    .. |idict| replace:: *dict*
    .. |istr| replace:: *str*
    .. |Node| replace:: :py:class:`nodestimation.Node`
    .. |pandas.DataFrame| replace:: *pandas.DataFrame*
    .. |inx.Graph| replace:: *inx.Graph*

    .. _`data types`:
    .. note:: Used types:

        :raw: `Raw <https://mne.tools/stable/generated/mne.io.Raw.html>`_ object
        :raw_fp: `Raw <https://mne.tools/stable/generated/mne.io.Raw.html>`_ object after processing (cropping, filtering and so on)
        :bem: `ConductorModel <https://mne.tools/stable/generated/mne.bem.ConductorModel.html#mne.bem.ConductorModel>`_ object
        :src: `SourceSpaces <https://mne.tools/stable/generated/mne.SourceSpaces.html>`_
        :trans: `Transform <https://mne.tools/stable/generated/mne.transforms.Transform.html#mne.transforms.Transform>`_
        :fwd: `Forward <https://mne.tools/stable/generated/mne.Forward.html#mne.Forward>`_
        :eve: `Events <https://numpy.org/devdocs/reference/generated/numpy.ndarray.html#numpy.ndarray>`_
        :epo: `Epochs <https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs>`_
        :cov: `Covariance <https://mne.tools/stable/generated/mne.Covariance.html?highlight=covariance#mne.Covariance>`_
        :ave: `Evoked <https://mne.tools/stable/generated/mne.Evoked.html?highlight=evoked#mne.Evoked>`_
        :inv: `InverseOperator <https://mne.tools/stable/generated/mne.minimum_norm.InverseOperator.html#mne.minimum_norm.InverseOperator>`_,
        :stc: `SourceEstimate <https://mne.tools/stable/generated/mne.SourceEstimate.html#mne.SourceEstimate>`_
        :resec: Resection in `".nii" <https://nipy.org/nibabel/gettingstarted.html>`_ format
        :resec_txt: Resection in `".txt" <https://en.wikipedia.org/wiki/Text_file>`_ format
        :resec_mni: Resection in `mni <https://brainmap.org/training/BrettTransform.html>`_ coordinates
        :coords: centers coordinates of `cortical parcellation <https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation>`_ in `mni <https://brainmap.org/training/BrettTransform.html>`_ coordinates
        :feat: dictionary for all metrics values to all frequency bands to all methods,
        :nodes: list of :class:`nodestimation.Node` objects
        :dataset: dataset for all features and frequencies to all nodes
    """

    def __init__(self, name: str, data: Dict[str, str], nodes: List[Node], directory: str, datasets: Dict[str, pd.DataFrame], connectomes: Connectomes):
        self.name = name
        self.data = data
        self.nodes = nodes
        self.dir = directory
        self.datasets = datasets
        self.connectomes = connectomes

    def __str__(self):
        return 'Subject {} at {} \n'.format(self.name, self.dir)

    @property
    def data(self):

        return self

    @property
    def nodes(self):

        return self

    @property
    def dir(self):

        return self

    @property
    def datasets(self):

        return self

    @property
    def connectomes(self):

        return self

    @data.setter
    def data(self, value):

        self._data = value

    @data.getter
    def data(self) -> Dict[str, str]:
        """a dictionary containing paths to patient data files used in calculations.
            Keys are `data types`_, values are paths to files with this data (or None if they do not exist).
        """

        return self._data

    @nodes.setter
    def nodes(self, value):

        self._nodes = value

    @nodes.getter
    def nodes(self) -> List[Node]:
        """list of :class:`nodestimation.Node` objects related to this patient
        """

        return self._nodes

    @dir.setter
    def dir(self, value):

        self._dir = value

    @dir.getter
    def dir(self) -> str:
        """patient home directory (must start with ``'./Source/Subjects/'``)
        """

        return self._dir

    @datasets.setter
    def datasets(self, value):

        self._dataset = value

    @datasets.getter
    def datasets(self):
        """datasets with all features and frequencies to all nodes
        """

        return self._dataset

    @connectomes.setter
    def connectomes(self, value):

        self._connectomes = value

    @connectomes.getter
    def connectomes(self):
        """graphs for all features to all frequencies
        """

        return self._connectomes

    def data_description(self) -> str:
        """makes description of subject data structure

        :rtype: str
        """
        description = '{\n'
        for data_type, path in self.data.items():
            description += '{}: {},\n'.format(data_type, path)
        return description + '}'
