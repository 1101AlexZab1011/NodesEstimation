import time
from typing import *
from abc import *
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
from nodestimation.learning.modification import append_series, choose_items, choose_indices, rm, binarize, suppress, promote, clusterize
from nodestimation.project.subject import Subject


class Test(ABC):
    """Abstract test providing `statistical tests <https://en.wikipedia.org/wiki/Statistical_hypothesis_testing>`_ for given `true data`_ and `false data`_

        :param class1: `false data`_
        :type class1: |ipd.DataFrame|_
        :param class2: `true data`_
        :type class2: |ipd.DataFrame|_
        :param name: some information to describe analyzed datasets
        :type name: str

    """

    def __init__(self, class1, class2, name):
        self.result = class1, class2
        self.name = name

    def __str__(self):
        return '{} test object for {} data'.format(self.type, self.name)

    @property
    def result(self):
        return self

    @result.setter
    def result(self, classes: Tuple[pd.DataFrame, pd.DataFrame]):
        self._result = self.run_test(classes[0], classes[1])

    @result.getter
    def result(self):
        """dictionary with features as keys and with statistics and p_values as values"""
        return self._result

    @property
    def name(self):
        return self

    @name.setter
    def name(self, value):
        self._name = value

    @name.getter
    def name(self):
        """some information to describe analyzed datasets"""
        return self._name

    @property
    @abstractmethod
    def type(self):
        """Key word to describe statistical test"""
        return 'Abstract'

    @staticmethod
    @abstractmethod
    def run_test(class1: pd.DataFrame, class2: pd.DataFrame) -> Dict[Union[str, int], Tuple[float, float]]:
        """computes statistic and p-value
            for given datasets

            :param class1: `false data`_
            :type class1: |ipd.DataFrame|_
            :param class2: `true data`_
            :type class2: |ipd.DataFrame|_
            :return: dictionary with features as keys and with statistics and p_values as values
            :rtype: dict_ of str_ or int_ to tuple_ of float_

        """
        pass

    def show_result(self):
        """prints test results for each feature

        """

        for feature in self.result:
            print(
                '{} test, {}:\n'
                '\tFeature: {}\n'
                '\tStatistic: {}\n'
                '\tP-value: {}\n'
                .format(
                    self.type,
                    self.name,
                    feature,
                    self.result[feature][0],
                    self.result[feature][1]
                )
            )


class Wilcoxon(Test):
    """`Wilcoxon <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html>`_ test for given `true data`_ and `false data`_,
        inherits from :class:`nodestimation.learning.selection.Test`

        :param class1: `false data`_
        :type class1: |ipd.DataFrame|_
        :param class2: `true data`_
        :type class2: |ipd.DataFrame|_
        :param name: some information to describe analyzed datasets
        :type name: str

    """

    def __init__(self, class1, class2, name):
        super().__init__(class1, class2, name)

    @property
    def type(self):
        return 'Wilcoxon'

    @staticmethod
    def run_test(class1: pd.DataFrame, class2: pd.DataFrame) -> Dict[Union[str, int], Tuple[float, float]]:
        """computes statistic and p-value
            for given datasets

            :param class1: `false data`_
            :type class1: |ipd.DataFrame|_
            :param class2: `true data`_
            :type class2: |ipd.DataFrame|_
            :return: dictionary with features as keys and with statistics and p_values as values
            :rtype: dict_ of str_ or int_ to tuple_ of float_

        """

        columns = class1.columns.tolist()
        false_dataset = class1.to_numpy()
        true_dataset = class2.to_numpy()

        out = dict()

        for i in range(len(columns)):
            w, p = wilcoxon(false_dataset[:, i], true_dataset[:, i])
            out.update(
                {
                    columns[i]:
                        (
                            w,
                            p
                        )
                }
            )

        return out


class Mannwhitneyu(Test):
    """`Mannwhitneyu <https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test>`_ test for given `true data`_ and `false data`_,
        inherits from :class:`nodestimation.learning.selection.Test`

        :param class1: `false data`_
        :type class1: |ipd.DataFrame|_
        :param class2: `true data`_
        :type class2: |ipd.DataFrame|_
        :param name: some information to describe analyzed datasets
        :type name: str

    """

    def __init__(self, class1, class2, name):
        super().__init__(class1, class2, name)

    @property
    def type(self):
        return 'Mannwhitneyu'

    @staticmethod
    def run_test(class1: pd.DataFrame, class2: pd.DataFrame) -> Dict[Union[str, int], Tuple[float, float]]:
        """computes statistic and p-value
            for given datasets

            :param class1: `false data`_
            :type class1: |ipd.DataFrame|_
            :param class2: `true data`_
            :type class2: |ipd.DataFrame|_
            :return: dictionary with features as keys and with statistics and p_values as values
            :rtype: dict_ of str_ or int_ to tuple_ of float_

        """

        columns = class1.columns.tolist()
        false_dataset = class1.to_numpy()
        true_dataset = class2.to_numpy()

        out = dict()

        for i in range(len(columns)):
            u, p = mannwhitneyu(false_dataset[:, i], true_dataset[:, i])
            out.update(
                {
                    columns[i]:
                        (
                            u,
                            p
                        )
                }
            )

        return out


class SubjectsStatistic(object):
    """Provides statistical information about a list of subjects

        :param subjects: list of subjects
        :type subjects: |ilist|_ *of* :class:`nodestimation.project.subject.Subject`
        :param target: `target feature`_
        :type target: str
        :param centrality_metric: `centrality metric <nodestimation.html#centrality-metrics>`_ to compute, default "eigen"
        :type centrality_metric: str, optional
        :param convert: what type of conversion_ to apply to the given data, if None, does not convert, default None
        :type convert: str, oprional

        .. _conversion:
        .. note:: Available data conversions

            :None: does not convert given data
            :binarize: binarizes given data, provided by :func:`nodestimation.learning.modification.binarize`
            :suppress: suppresses given data with optimal value equated to minimum value, provided by :func:`nodestimation.learning.modification.suppress`
            :promote: promotes given data with optimal value equated to maximum value, provided by :func:`nodestimation.learning.modification.promote`
            :clusterize: clusterizes given data into 5 clusters with optimal value equated to the cluster ordinal number, provided by: :func:`nodestimation.learning.modification.clusterize`
    """

    def __init__(self, subjects: List[Subject], target: str, centrality_metric: str = 'eigen', convert: Optional[str] = None):
        self.__convert = convert
        self.__centrality_metric = centrality_metric
        self.subjects = subjects
        self.datasets = subjects
        self.target = target

    def __str__(self):
        return 'SubjectsStatistic for: {}\n\t target feature: {}\n\t metric: {}\n\t data transformation: {}'.format(
            [subject.name for subject in self.subjects],
            self.target,
            self.__centrality_metric,
            self.__convert
        )

    @property
    def centrality_metric(self):
        return self.__centrality_metric

    @centrality_metric.setter
    def centrality_metric(self, value):
        raise AttributeError('Can not set centrality_metric')

    @property
    def subjects(self):
        return self

    @subjects.getter
    def subjects(self) -> List[Subject]:
        """list of :class:`nodestimation.project.subject.Subject` objects
        """

        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        self._subjects = subjects

    @property
    def datasets(self):
        return self

    @datasets.setter
    def datasets(self, subjects: List[Subject]):

        full = self.subjects_to_dataframe(subjects, self.__centrality_metric)
        full = {
            None: lambda df, **kwargs: df,
            'binarize': binarize,
            'suppress': suppress,
            'promote': promote,
            'clusterize': clusterize
        }[self.__convert](full.drop(['resected'], axis=1), axis=1).assign(resected=full['resected'])
        true, false = self.true_false_division(full)
        false_mirror = self.reflect_true(true, false, subjects)
        self._datasets = {
            'full': full,
            'true': true,
            'false': false,
            'false_mirror': false_mirror,
            'false_res': self.resample_false(true, false)
        }

    @datasets.getter
    def datasets(self):
        """dictionary of `required datasets`_

            .. _`provided datasets`:
            .. _`required dataset`:
            .. _`required datasets`:
            .. note:: Provided datasets:

                :full: concatenated datasets from each :class:`nodestimation.project.subject.Subject` object
                :true: all `true data`_ from full dataset
                :false: all `false data`_ from full dataset
                :false_mirror: dataset from data symmetric_ to `true data`_
                :false_res: dataset with `false data`_ resampled to the shape of dataset with `true data`_

            .. _symmetric:
            .. note:: Each data sample_ represents :class:`nodestimation.Node` that refers to one of brain regions. Brain regions are
                `named <http://www.cis.jhu.edu/~parky/MRN/Desikan%20Region%20Labels%20and%20Descriptions.pdf>`_ according
                to `parcellation type <https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation>`_.
                If two brain regions have the same name and different hemisphere markers ("-lh" and "-rh"), they are symmetrical

        """
        return self._datasets

    @property
    def target(self):
        return self

    @target.setter
    def target(self, new_target):
        self._target = new_target

    @target.getter
    def target(self):
        """a `target feature`_
        """

        return self._target

    @staticmethod
    def subjects_to_dataframe(subjects: List[Subject], centrality_metric: str) -> pd.DataFrame:
        """Concatenates all subjects datasets in one dataset and adds subject name to its sample_ names

        :param subjects: list of subjects
        :type subjects: |ilist|_ *of* :class:`nodestimation.project.subject.Subject`
        :param centrality_metric: `centrality metric <nodestimation.html#centrality-metrics>`_ to use
        :return: concatenated dataset
        :rtype: pd.DataFrame_
        """

        dataset = pd.concat(
            [subject.datasets[centrality_metric]
             for subject in subjects],
            axis=0,
        )
        if dataset.isnull().values.any():
            raise ValueError('Nan values appear. Check if column names are equal through subjects datasets')

        new_indexes = list()

        for subject in subjects:
            for index in subject.datasets[centrality_metric].index:
                new_indexes.append('_'.join([subject.name, index]))

        return pd.DataFrame(dataset.to_numpy(), columns=dataset.columns, index=new_indexes)

    @staticmethod
    def true_false_division(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """divides dataset into `true data`_ and `false data`_

        :param dataset: dataset to divide
        :type dataset: |ipd.DataFrame|_
        :return: dataset with `true data`_ and `false data`_
        :rtype: tuple_ of pd.DataFrame_

        """

        true_cases, false_cases = pd.DataFrame(), pd.DataFrame()

        for index in dataset.index:

            if dataset.loc[index]['resected']:
                true_cases = append_series(true_cases, dataset.loc[index], index=[index])

            else:

                false_cases = append_series(false_cases, dataset.loc[index], index=[index])

        return true_cases.drop(['resected'], axis=1), false_cases.drop(['resected'], axis=1)

    @staticmethod
    def reflect_true(true_dataset: pd.DataFrame, false_dataset: pd.DataFrame, subjects: List[Subject]) -> pd.DataFrame:
        """finds samples_ symmetric_ to given dataset with `true data`_

        :param true_dataset: dataset with `true data`_
        :type true_dataset: |ipd.DataFrame|_
        :param false_dataset: dataset with `false data`_
        :type false_dataset: |ipd.DataFrame|_
        :param subjects: list of subjects
        :type subjects: |ilist|_ *of* :class:`nodestimation.project.subject.Subject`
        :return: symmetric_ dataset
        :rtype: pd.DataFrame_
        :raise ValueError: if required sample_ is not found or hemisphere marker different from "rh" or "lh"
        """

        def postfix(name: str) -> str:
            if name[-2:] == 'rh':
                return 'lh'
            elif name[-2:] == 'lh':
                return 'rh'
            else:
                raise ValueError(name[-2:])

        def prefix(name: str, subjects: List[Subject]) -> str:
            label_name = name[5:-2] + postfix(name)
            for subject in subjects:
                if subject.datasets.loc[label_name]['resected']:
                    continue
                else:
                    return subject.name
            raise ValueError('label' + name[4:] + ' not found')

        false_dataset_symmetric = pd.DataFrame()

        for index in true_dataset.index:
            try:
                false_dataset_symmetric = pd.concat(
                    [
                        false_dataset_symmetric,
                        false_dataset.loc[index[0:-2] + postfix(index)]
                    ],
                    axis=1
                )
            except KeyError:
                false_dataset_symmetric = pd.concat(
                    [
                        false_dataset_symmetric,
                        false_dataset.loc[prefix(index, subjects) + index[4:-2] + postfix(index)]
                    ],
                    axis=1
                )

        return false_dataset_symmetric.T

    @staticmethod
    def resample_false(true_dataset: pd.DataFrame, false_dataset: pd.DataFrame) -> pd.DataFrame:
        """resample `false data`_ to the same shape as `true data`_

        :param true_dataset: dataset with `true data`_
        :type true_dataset: |ipd.DataFrame|_
        :param false_dataset: dataset with `false data`_
        :type false_dataset: |ipd.DataFrame|_
        :return: resampled dataset with `false data`_
        :rtype: |ipd.DataFrame|_
        """

        false_data_np, \
        true_data_np \
            = \
            false_dataset.to_numpy().T, \
            true_dataset.to_numpy().T

        data_for_resampling = false_data_np.copy()

        bands = list()

        for i in range(false_data_np.shape[1] // true_data_np.shape[1]):
            indices = choose_indices(data_for_resampling, number=true_data_np.shape[1], axis=1)
            bands.append(choose_items(data_for_resampling, indices=indices, axis=1))
            data_for_resampling = rm(data_for_resampling, indices=indices, axis=1)

        return pd.DataFrame(np.mean(np.array(bands), axis=0).T, columns=true_dataset.columns)

    def random_samples(self, data: str = 'false', number: int = None):
        """creates dataset with randomly chosen `samples`_

            :param data: data to use, can be any of `provided datasets`_ , default "false"
            :type data: str, optional
            :param number: number of samples to choose, if None, number equated to number of samples for `true data`_ will be chosen, default None
            :type number: int, optional
            :return: dataset with randomly chosen `samples`_
            :rtype: pd.DataFrame_
        """

        data_np = {
            'true': self.datasets['true'].to_numpy(),
            'false': self.datasets['false'].to_numpy(),
            'full': self.datasets['full'].to_numpy(),
            'false_mirror': self.datasets['false_mirror'].to_numpy(),
            'false_res': self.datasets['false_res'].to_numpy()
        }[data]

        if number is None:
            number = self.datasets['true'].shape[0]

        samples = choose_items(data_np, number)

        return pd.DataFrame(samples, columns=self.datasets['true'].columns)

    def test(self, state: str = 'resampled', test: str = 'wilcoxon') -> Test:
        """computes required `statistical tests <https://en.wikipedia.org/wiki/Statistical_hypothesis_testing>`_
            for `required dataset`_: either "false_res" or "false_mirror"

            :param state: dataset to analyze: either "resampled" for "false_res" or "reflected" for "false_mirror", default "resampled"
            :type state: str, optional
            :param test: what `test`_ to compute, default "wilcoxon"
            :type test: str, optional
            :return: computed test
            :rtype: :class:`nodestimation.learning.selection.Test`
            :raise ValueError: if specified state incorrect

            .. _dict: https://docs.python.org/3/library/stdtypes.html#dict
            .. _str: https://docs.python.org/3/library/stdtypes.html#str
            .. _float: https://docs.python.org/3/library/functions.html#float

            .. _`test`:
            .. note:: Available tests:

                :mannwhitneyu: `Mann–Whitney U test <https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test>`_
                :wilcoxon: `Wilcoxon signed-rank test <https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test>`_
        """

        if state == 'resampled':
            true_data = self.datasets['true']
            false_data = self.datasets['false_res']

        elif state == 'reflected':
            true_data = self.datasets['true']
            false_data = self.datasets['false_mirror']
        else:
            raise ValueError('parameter state can be "resampled" or "reflected"')

        return {
            'wilcoxon': Wilcoxon,
            'mannwhitneyu': Mannwhitneyu
        }[test](true_data, false_data, state + '/true')
