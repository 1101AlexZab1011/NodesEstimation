from typing import *

from nodestimation import Node
from pandas import DataFrame


class Subject(object):

    def __init__(self, name: str, data: Dict[str, str], nodes: List[Node], directory: str, dataset: DataFrame):
        self.name = name
        self.data = data
        self.nodes = nodes
        self.dir = directory
        self.dataset = dataset

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
    def dataset(self):
        return self

    @data.setter
    def data(self, value):
        self._data = value

    @data.getter
    def data(self):
        return self._data

    @nodes.setter
    def nodes(self, value):
        self._nodes = value

    @nodes.getter
    def nodes(self):
        return self._nodes

    @dir.setter
    def dir(self, value):
        self._dir = value

    @dir.getter
    def dir(self):
        return self._dir

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    @dataset.getter
    def dataset(self):
        return self._dataset

    def data_description(self):
        description = '{\n'
        for data_type, path in self.data.items():
            description += '{}: {},\n'.format(data_type, path)
        return description + '}'
