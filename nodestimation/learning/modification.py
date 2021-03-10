from typing import *
import pandas as pd
import numpy as np


def append_series(df: pd.DataFrame, series: Union[pd.Series, List[pd.Series]], index: Optional[Union[int, str, List[Union[int, str]]]] = None) -> pd.DataFrame:
    """ Add series_ to the end
        of given pd.DataFrame_

        :param df: pd.DataFrame_ to change
        :type df: |ipd.DataFrame|_
        :param series: pd.Series_ to add
        :type series: |ipd.Series|_ *or* |ilist|_ *of* |ipd.Series|_
        :param index: index_ to new pd.DataFrame_ row
        :type index: |istr|_ *or* |iint|_ *or* |ilist|_ *of* |istr|_ *or* |iint|_
        :return: modified pd.DataFrame_
        :rtype: pd.DataFrame_

        .. _ipd.Series:
        .. _pd.Series:
        .. _series: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
        .. _ipd.dataframe:
        .. _pd.DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
        .. _ilist: https://docs.python.org/3/library/stdtypes.html#list
        .. _index: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.html
        .. _istr: https://docs.python.org/3/library/stdtypes.html#str
        .. _iint: https://docs.python.org/3/library/functions.html#int

        .. |ipd.DataFrame| replace:: *pd.DataFrame*
        .. |ipd.Series| replace:: *pd.Series*
        .. |ilist| replace:: *list*
        .. |istr| replace:: *str*
        .. |iint| replace:: *int*
    """

    if not isinstance(series, list):
        if len(df.columns.tolist()) != 0:
            columns = df.columns.tolist()
        else:
            columns = series.index.tolist()

        series = [series.to_numpy()]
    else:
        if len(df.columns.tolist()) != 0:
            columns = df.columns.tolist()
        else:
            columns = series[0].index.tolist()

        series = [
            s.to_numpy()
            for s in series
        ]

    if index is not None:

        if not isinstance(index, list):
            index = [str(index)]

        else:

            if len(index) > len(series):
                index = index[:len(series)]

            elif len(index) < len(series):

                for i in range(len(series) - len(index)):
                    index.append('Unnamed {}'.format(i))

            for i in index:
                if not isinstance(i, str):
                    str(i)

            index = np.array(index)

    other = pd.DataFrame(series, index=index, columns=columns)

    return df.append(other)


def appstart_series(df: pd.DataFrame, series: Union[pd.Series, List[pd.Series]], index: Optional[Union[int, str, List[Union[int, str]]]] = None) -> pd.DataFrame:
    """ Add series_ to the start
        of given pd.DataFrame_

        :param df: pd.DataFrame_ to change
        :type df: |ipd.DataFrame|_
        :param series: pd.Series_ to add
        :type series: |ipd.Series|_ *or* |ilist|_ *of* |ipd.Series|_
        :param index: index_ to new pd.DataFrame_ row, default None
        :type index: |istr|_ *or* |iint|_ *or* |ilist|_ *of* |istr|_ *or* |iint|_ *, optional*
        :return: modified pd.DataFrame_
        :rtype: pd.DataFrame_
    """

    # TODO does not optimized function!

    if not isinstance(series, list):
        series = [series]

    if index is not None:

        if not isinstance(index, list):
            index = [str(index)]

        else:

            if len(index) > len(series):
                index = index[:len(series)]

            elif len(index) < len(series):

                for i in range(len(series) - len(index)):
                    index.append('Unnamed {}'.format(i))

            for i in index:
                if not isinstance(i, str):
                    str(i)

        index = np.array(index + df.index.tolist())

    df_series = [df.iloc[i] for i in range(df.shape[0])]

    for s in df_series:
        series.append(s)

    return pd.DataFrame(series, index=index)


def rm(array: Union[list, np.ndarray], indices: Union[int, List[int]], axis: int = 0) -> any:
    """removes elements with specified indexes from array

        :param array: array-like object to change
        :type array: |inp.ndarray|_ *or* |ilist|_
        :param indices: indices related to the elements to be removed
        :type indices: |iint|_ *or* |ilist|_ *of* |iint|_
        :param axis: axis along which to choose, default 0
        :type axis: int, optional
        :return: modified np.ndarray_
        :rtype: np.ndarray_

        .. _np.ndarray:
        .. _inp.ndarray: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html

        .. |inp.ndarray| replace:: *np.ndarray*
    """

    if not isinstance(indices, list):
        indices = [indices]

    if not isinstance(array, np.ndarray):
        array = np.array(array)

    return np.delete(array, indices, axis=axis)


def choose_indices(array: np.ndarray, number: int, axis: int = 0) -> List[int]:
    """chooses specified number of random indices from given np.ndarray_

        :param array: array to choose indices
        :type array: |inp.ndarray|_
        :param number: number of indices to choose
        :type number: int
        :param axis: axis along which to choose, default 0
        :type axis: int, optional
        :return: list of indices
        :rtype: list_ of int_

        .. _list: https://docs.python.org/3/library/stdtypes.html#list
        .. _int: https://docs.python.org/3/library/functions.html#int
    """

    if len(array.shape) < axis:
        raise ValueError('This axis does not exist')

    if number > array.shape[axis]:
        raise ValueError('Cannot choose the number of indices greater than the considered axis of the given array')

    return sorted([np.random.randint(array.shape[axis]) for i in range(number)])


def choose_items(array: np.ndarray, number: Optional[int] = None, indices: Optional[List[int]] = None, axis=0) -> np.ndarray:
    """chooses specified number of random elements from given np.ndarray_

    :param array: array to take elements
    :type array: |inp.ndarray|_
    :param number: number of random elements to take. If None, considers required indices, default None
    :type number: int, optional
    :param indices: If number is None, takes specified indices, default None
    :type indices: |ilist|_ *of* |iint|_ *,optional*
    :param axis: axis along which to choose, default 0
    :type axis: int, optional
    :return: array of chosen elements
    :rtype: np.ndarray_
    :raise ValueError: if neither number nor indices specified
    """
    if number is not None:
        indices = choose_indices(array, number, axis)
    elif indices is None:
        raise ValueError('Either number or indices must be specified')

    return np.take(array, indices, axis=axis)


# FIXME: duplicated code below!
# motivated by the fact that I want to "supress", "promote" and "binarize" to be different functions

def suppress(dataset: pd.DataFrame, trigger: Optional[int] = None, optimal: Optional[Union[int, str]] = 'mean', axis: int = 0) -> pd.DataFrame:
    """Suppress given dataset to optimal value

        :param dataset: dataset to suppress
        :type dataset: |ipd.DataFrame|_
        :param trigger: trigger value, all values less than the trigger will be optimized, if None takes mean, default None
        :type trigger: int, optional
        :param optimal: value to suppress, if "mean" takes mean of suppressed values, if "min" takes min of suppressed values, if "max" takes max of suppressed values, default "mean"
        :type optimal: |iint|_ *or* |istr|_ *, optional*
        :param axis: axis along which to suppress, default 0
        :type axis: int, optional
        :return: suppressed dataset
        :rtype: pd.DataFrame_
        :raise ValueError: if rigger or optimal conditions unreached or optimal value is wrong
    """

    def create_optimal_mask(dataset, trigger_mask, for_):

        optimal_mask = list()

        for i in range(dataset.shape[0]):

            suppressed = list()
            row = np.take(dataset, [i], axis=0)[0]

            for elem in row:

                if elem < trigger_mask[i]:
                    suppressed.append(elem)

            if for_ == 'mean':
                optimal_mask.append(np.mean(np.array(suppressed)))

            elif for_ == 'max':
                optimal_mask.append(np.max(np.array(suppressed)))

            elif for_ == 'min':
                optimal_mask.append(np.min(np.array(suppressed)))

        return optimal_mask

    columns = dataset.columns
    indices = dataset.index
    dataset = dataset.copy()

    if axis == 0:
        dataset = dataset.to_numpy()
    else:
        dataset = dataset.to_numpy().T

    if trigger is None:
        trigger_mask = [np.mean(row) for row in np.take(dataset, range(dataset.shape[0]), axis=0)]
    else:
        trigger_mask = [trigger for i in range(dataset.shape[0])]

    if len(trigger_mask) == 0:
        raise ValueError('trigger is corrupted')

    if optimal == 'max' or optimal == 'min' or optimal == 'mean':
        optimal_mask = create_optimal_mask(dataset, trigger_mask, for_=optimal)

    elif isinstance(optimal, int):
        optimal_mask = [optimal for i in range(dataset.shape[0])]

    else:
        raise ValueError('Wrong optimal value: {}'.format(optimal))

    if len(optimal_mask) == 0:
        raise ValueError('optimal is corrupted')

    processed = list()

    for i in range(dataset.shape[0]):
        row = np.take(dataset, [i], axis=0)[0]

        for j in range(len(row)):
            if row[j] < trigger_mask[i]:
                row[j] = optimal_mask[i]
        processed.append(row)

    if axis == 0:
        return pd.DataFrame(
            np.array(processed),
            index=indices,
            columns=columns
        )
    else:
        return pd.DataFrame(
            np.array(processed).T,
            index=indices,
            columns=columns
        )


def promote(dataset: pd.DataFrame, trigger: Optional[int] = None, optimal: Optional[Union[int, str]] = 'mean', axis: int = 0) -> pd.DataFrame:
    """Promotes given dataset to optimal value

        :param dataset: dataset to promote
        :type dataset: |ipd.DataFrame|_
        :param trigger: trigger value, all values greater than the trigger will be optimized, if None takes mean, default None
        :type trigger: int, optional
        :param optimal: value to promote, if "mean" takes mean of promoted values, if "min" takes min of promoted values, if "max" takes max of promoted values, default "mean"
        :type optimal: |iint|_ *or* |istr|_ *, optional*
        :param axis: axis along which to promote, default 0
        :type axis: int, optional
        :return: promoted dataset
        :rtype: pd.DataFrame_
        :raise ValueError: if rigger or optimal conditions unreached or optimal value is wrong
        """

    def create_optimal_mask(dataset, trigger_mask, for_):

        optimal_mask = list()

        for i in range(dataset.shape[0]):

            promoted = list()
            row = np.take(dataset, [i], axis=0)[0]

            for elem in row:

                if elem > trigger_mask[i]:
                    promoted.append(elem)

            if for_ == 'mean':
                optimal_mask.append(np.mean(np.array(promoted)))

            elif for_ == 'max':
                optimal_mask.append(np.max(np.array(promoted)))

            elif for_ == 'min':
                optimal_mask.append(np.min(np.array(promoted)))

        return optimal_mask

    columns = dataset.columns
    indices = dataset.index
    dataset = dataset.copy()

    if axis == 0:
        dataset = dataset.to_numpy()
    else:
        dataset = dataset.to_numpy().T

    if trigger is None:
        trigger_mask = [np.mean(row) for row in np.take(dataset, range(dataset.shape[0]), axis=0)]
    else:
        trigger_mask = [trigger for i in range(dataset.shape[0])]

    if len(trigger_mask) == 0:
        raise ValueError('trigger is corrupted')

    if optimal == 'max' or optimal == 'min' or optimal == 'mean':
        optimal_mask = create_optimal_mask(dataset, trigger_mask, for_=optimal)

    elif isinstance(optimal, int):
        optimal_mask = [optimal for i in range(dataset.shape[0])]

    else:
        raise ValueError('Wrong optimal value: {}'.format(optimal))

    if len(optimal_mask) == 0:
        raise ValueError('optimal is corrupted')

    processed = list()

    for i in range(dataset.shape[0]):
        row = np.take(dataset, [i], axis=0)[0]

        for j in range(len(row)):
            if row[j] > trigger_mask[i]:
                row[j] = optimal_mask[i]
        processed.append(row)

    if axis == 0:
        return pd.DataFrame(
            np.array(processed),
            index=indices,
            columns=columns
        )
    else:
        return pd.DataFrame(
            np.array(processed).T,
            index=indices,
            columns=columns
        )


def binarize(dataset: pd.DataFrame, trigger: Optional[int] = None, axis: int = 0) -> pd.DataFrame:
    """binarizes given dataset

    :param dataset: dataset to binarize
    :type dataset: |ipd.DataFrame|_
    :param trigger: trigger value, all values less than the trigger will be nullified, the rest - equated to 1, if None takes mean, default None
    :type trigger: int, optional
    :param axis: axis along which to binarize, default 0
    :type axis: int, optional
    :return: binarized dataset
    :rtype: pd.DataFrame_
    :raise ValueError: if trigger value unreached
    """
    columns = dataset.columns
    indices = dataset.index
    dataset = dataset.copy()

    if axis == 0:
        dataset = dataset.to_numpy()
    else:
        dataset = dataset.to_numpy().T

    if trigger is None:
        trigger_mask = [np.mean(row) for row in np.take(dataset, range(dataset.shape[0]), axis=0)]
    else:
        trigger_mask = [trigger for i in range(dataset.shape[0])]

    if len(trigger_mask) == 0:
        raise ValueError('trigger is corrupted')

    processed = list()

    for i in range(dataset.shape[0]):
        row = np.take(dataset, [i], axis=0)[0]

        for j in range(len(row)):
            if row[j] > trigger_mask[i]:
                row[j] = 1
            else:
                row[j] = 0
        processed.append(row)

    if axis == 0:
        return pd.DataFrame(
            np.array(processed),
            index=indices,
            columns=columns
        )
    else:
        return pd.DataFrame(
            np.array(processed).T,
            index=indices,
            columns=columns
        )


def clusterize(dataset: pd.DataFrame, n_clusters: int = 5, optimal: str = 'num', axis: int = 0) -> pd.DataFrame:
    """groups given data into set of clusters

        :param dataset: dataset to binarize
        :type dataset: |ipd.DataFrame|_
        :param n_clusters: number of clusters to group, default 5
        :type n_clusters: int, optional
        :param optimal: rule_ to cclusterize, default "num"
        :type optimal: str, optional
        :param axis: axis along which to clusterize, default 0
        :type axis: int, optional
        :return: clusterized dataset
        :rtype: pd.DataFrame_

        .. _rule:
        .. note:: Clusterization rules

            :num: the value for each element in the cluster is equated to the ordinal number of this cluster
            :mean: the value for each element in the cluster is equated to the average of cluster elements
            :min: the value for each element in the cluster is equated to the minimum of cluster elements
            :max: the value for each element in the cluster is equated to the maximum of cluster elements
            :symfar: the value for each element in the cluster is equated to the maximum of cluster elements if the ordinal number of this cluster less than half of clusters number, otherwise to minimum
            :symclose: the value for each element in the cluster is equated to the minimum of cluster elements if the ordinal number of this cluster less than half of clusters number, otherwise to maximum
    """

    def group(dataset: np.ndarray, n_clusters: int, indices: Iterable, sort: bool = False) -> Dict[Any, List[List[np.ndarray]]]:
        dataset = dataset.copy()
        cluster_size = dataset.shape[1]//n_clusters

        out = dict()

        for row, index in zip(dataset, indices):
            out.update({index: [list() for i in range(n_clusters)]})
            if sort:
                row.sort()

            group_i = 0
            cluster_i = 0
            cluster_count = 0

            for i in range(row.shape[0]):

                if not group_i % cluster_size:
                    group_i = 0
                    cluster_i += 1

                    if cluster_count + 1 <= n_clusters:
                        cluster_count += 1

                group_i += 1
                out[index][cluster_count - 1].append(row[group_i + cluster_size*(cluster_i - 1) - 1])

        return out

    def symfar(cluster: List[np.ndarray], num: int, n_clusters: int) -> float:
        if num > n_clusters/2:
            return round(np.max(cluster), 4) + 0.0001
        else:
            return round(np.min(cluster), 4) - 0.0001

    def symclose(cluster: List[np.ndarray], num: int, n_clusters: int) -> float:
        if num > n_clusters/2:
            return round(np.min(cluster), 4) - 0.0001
        else:
            return round(np.max(cluster), 4) + 0.0001

    def cluster_statistic(groups, n_clusters):
        statistic = dict()

        for group_ in groups:
            statistic.update({
                group_: [
                    {
                        'num': num,
                        'mean': np.mean(cluster),
                        'stdev': np.std(cluster),
                        'min': round(np.min(cluster), 4) - 0.0001,
                        'max': round(np.max(cluster), 4) + 0.0001,
                        'symfar': symfar(cluster, num, n_clusters),
                        'symclose': symclose(cluster, num, n_clusters)
                    }
                    for cluster, num in zip(groups[group_], range(len(groups[group_])))
                ]
            })

        return statistic

    columns = dataset.columns
    indices = dataset.index
    labels = None

    if axis == 0:
        labels = indices
    elif axis == 1:
        labels = columns

    dataset = dataset.copy()

    if axis == 0:
        dataset = dataset.to_numpy()
    else:
        dataset = dataset.to_numpy().T

    groups_sorted = group(dataset, n_clusters, labels, sort=True)
    statistic = cluster_statistic(groups_sorted, n_clusters)

    processed = dataset.copy()

    for row, index in zip(processed, labels):
        for i in range(row.shape[0]):
            row[i] = [
                statistic[index][j][optimal]
                for j in range(n_clusters) if statistic[index][j]['min'] < row[i] < statistic[index][j]['max']
            ][0]

    if axis == 0:
        return pd.DataFrame(
                processed,
            index=indices,
            columns=columns
        )
    else:
        return pd.DataFrame(
            processed.T,
            index=indices,
            columns=columns
        )
