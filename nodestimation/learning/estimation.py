from typing import *
from nodestimation.learning.modification import append_series, appstart_series
import pandas as pd
import numpy as np


# FIXME collect_statistic have to be an object!

def collect_statistic(data: pd.DataFrame) -> pd.DataFrame:
    """collects `statistical information`_ about given pd.DataFrame_

        :param data: pd.DataFrame_ to analyze
        :type data: |ipd.DataFrame|_
        :return: pd.DataFrame_ with collected information for each column of given one
        :rtype: pd.DataFrame_

        .. _`statistical information`:
        .. note:: Collected information:

            :mean: mean_ value for feature_
            :stdev: `standard deviation`_ for feature_
            :m+std: mean_ value for feature_ + `standard deviation`_ for feature_
            :m-std: mean_ value for feature_ - `standard deviation`_ for feature_


        .. _mean: https://en.wikipedia.org/wiki/Mean
        .. _`standard deviation`: https://en.wikipedia.org/wiki/Standard_deviation
    """

    means = pd.Series([data[feat].mean() for feat in data.columns], index=data.columns)
    stds = pd.Series([data[feat].std() for feat in data.columns], index=data.columns)
    upper_bound = pd.Series([mean + std for mean, std in zip(means, stds)], index=data.columns)
    lower_bound = pd.Series([mean - std for mean, std in zip(means, stds)], index=data.columns)

    return pd.DataFrame([means, stds, upper_bound, lower_bound],
                        index=['mean', 'stdev', 'm+std', 'm-std'])


def compute_importance(data: pd.DataFrame, statistic: pd.DataFrame) -> pd.Series:
    """Analyze importance_ for each feature_

        :param data: pd.DataFrame_ to analyze
        :type data: |ipd.DataFrame|_
        :param statistic: output from :func:`nodestimation.learning.selection.collect_statistic`
        :type statistic: |ipd.DataFrame|_
        :return: pd.Series_ with computed importance for each feature_
        :rtype: pd.Series_

        .. _sample:
        .. _samples:
        .. _features:
        .. _feature:
        .. note:: Feature in terms of spreadsheet with data means one column of data. Otherwise, one row of data called "sample"

        .. _`important features`:
        .. _importance:
        .. note:: The important feature_ for this algorithm means the feature_ which is more deviant for the `true data`_ than for the `false data`_ and visa versa

        .. _`true data`:
        .. _`false data`:
        .. note:: True data here from the point of view of the classification problem for two classes means samples_ belonging to the first class,
            and false data, respectively, means samples_ belonging to the second class
    """

    sample_statistics = list()

    for i in range(data.shape[0]):
        sample_data = data.iloc[i]
        local_statistic = statistic
        sample_statistics.append(appstart_series(local_statistic, sample_data))

    feature_imp_statistics = pd.DataFrame()

    for sample_statistic in sample_statistics:
        outbounds = pd.Series([
            {
                True: sample_statistic.iloc[0][feat] - sample_statistic.loc['m+std'][feat],
                False: sample_statistic.loc['m-std'][feat] - sample_statistic.iloc[0][feat]
            }[sample_statistic.iloc[0][feat] > sample_statistic.loc['mean'][feat]]
            for feat in sample_statistic.columns
        ], index=sample_statistic.columns)

        importances = pd.Series([(sample_statistic.loc['stdev'][feat] + outbounds[feat])
                                 / sample_statistic.loc['stdev'][feat]
                                 for feat in sample_statistic.columns], index=sample_statistic.columns)

        feature_imp_statistics = append_series(feature_imp_statistics, importances)

    return pd.Series([
        feature_imp_statistics[feat].mean()
        for feat in feature_imp_statistics.columns
    ], index=feature_imp_statistics.columns)


def separate_datasets(datasets: Union[pd.DataFrame, List[pd.DataFrame]], target: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """Removes `target feature`_ from given dataset and divide it into true dataset (with `true data`_ only) and false dataset (with `false data`_ only)

        :param datasets: datasets to change
        :type datasets: |ipd.DataFrame|_ *or* |ilist|_ *of* |ipd.DataFrame|_
        :param target: `target feature`_ name
        :type target: str
        :return: full dataset without target feature, true dataset and false dataset
        :rtype: tuple_ of pd.DataFrame_

        .. _tuple: https://docs.python.org/3/library/stdtypes.html#tuple

        .. _`target feature`:
        .. note:: Target feature means a feature_ that marks samples_ as true and false. Used in the case of `supervised learning <https://en.wikipedia.org/wiki/Supervised_learning>`_
    """

    if not isinstance(datasets, list):
        datasets = [datasets]
    data = [dataset.drop([dataset.columns[0], target], axis=1) for dataset in datasets]
    true_cases = [
        pd.DataFrame([
            sample.iloc[i]
            for i in range(sample.shape[0])
            if dataset.iloc[i][target]
        ]) for sample, dataset in zip(data, datasets)
    ]
    false_cases = [
        pd.DataFrame([
            sample.iloc[i]
            for i in range(sample.shape[0])
            if not dataset.iloc[i][target]
        ]) for sample, dataset in zip(data, datasets)
    ]
    return data, true_cases, false_cases


def collect_cross_statistic(data: List[pd.DataFrame], true_cases: List[pd.DataFrame], false_cases: List[pd.DataFrame]) -> pd.DataFrame:
    """collects `cross statistical information`_ through several given datasets

        :param data: list of datasets
        :type data: |ilist|_ *of* |ipd.DataFrame|_
        :param true_cases: list of datasets with `true data`_ only
        :type true_cases: |ilist|_ *of* |ipd.DataFrame|_
        :param false_cases: list of datasets with `false data`_ only
        :type false_cases: |ilist|_ *of* |ipd.DataFrame|_
        :return: pd.DataFrame_ with collected `cross statistical information`_
        :rtype: pd.DataFrame_

        .. _`cross statistical information`:
        .. note:: Statistical information analyzed through datasets (for this algorithm suppose to analyze several datasets with similar features_ and samples_)

            :mstd/mave: `standard deviation`_ of mean_ values of the feature_ through datasets divided to averaged mean_ values of the feature through datasets
            :timpave: averaged importance_ of the feature_ for `true data`_
            :timpstd: `standard deviation`_ of importance_ of the feature_ for all `true data`_
            :fimpave: averaged importance_ of the feature_ for `false data`_
            :fimpstd: `standard deviation`_ of importance_ of the feature_ for all `false data`_

        .. _cross_statistic:
        .. note:: cross_statistic is :func:`nodestimation.learning.selection.collect_cross_statistic` output. Technically speaking, this is usual pd.DataFrame_, but
            it must contain `statistical information`_ for features

    """

    statistics = [collect_statistic(sample) for sample in data]
    mean_values = pd.DataFrame([statistic.loc['mean'] for statistic in statistics])
    true_importance = pd.DataFrame([compute_importance(true_case, statistic) for true_case, statistic in zip(true_cases, statistics)])
    false_importance = pd.DataFrame([compute_importance(false_case, statistic) for false_case, statistic in zip(false_cases, statistics)])

    return pd.DataFrame([
        pd.Series([mean_values[feat].std() / mean_values[feat].mean() for feat in mean_values.columns], index=mean_values.columns),
        pd.Series([true_importance[feat].mean() for feat in true_importance.columns], index=true_importance.columns),
        pd.Series([true_importance[feat].std() for feat in true_importance.columns], index=true_importance.columns),
        pd.Series([false_importance[feat].mean() for feat in false_importance.columns], index=false_importance.columns),
        pd.Series([false_importance[feat].std() for feat in false_importance.columns], index=false_importance.columns)
    ], index=[
        'mstd/mave', 'timpave', 'timpstd', 'fimpave', 'fimpstd'
    ])


# FIXME make_selection_map have to be an object!


def make_selection_map(cross_statistic: pd.DataFrame) -> pd.DataFrame:
    """computes average for all kind of samples_ of `cross statistical information`_ and compares if with `cross statistical information`_ for each feature_,
        then creates boolean map with "True" for `good feature`_ and "False" otherwise

        :param cross_statistic: `cross statistical information`_ for features_
        :type cross_statistic: |ipd.DataFrame|_
        :return: bool values for features ("True" for `good`_ otherwise "False")
        :rtype: pd.DataFrame_

        .. _good:
        .. _`good feature`:
        .. _`good features`:
        .. note:: The feature is better, the lesser **mstd/mave**, the lesser **timpstd** for `true data`_ (or **fimpstd** for `false data`_),
            and the greater **timpave** for `true data`_ (or **fimpave** for `false data`_) it has
            (look `cross statistical information`_ for meaning **mstd/mave**, **timpstd**, **fimpstd**, **timpave** and **fimpave**)

        .. _selection_map:
        .. note:: selection_map is :func:`nodestimation.learning.selection.make_selection_map` output. Technically speaking, this is usual pd.DataFrame_, but
            its values must be boolean and it must have only two rows: the first one used for `true data`_ selection and the second - for `false data`_
    """

    criteria = pd.Series([
        cross_statistic.loc['mstd/mave'].mean(),
        cross_statistic.loc['timpave'].mean(),
        cross_statistic.loc['timpstd'].mean(),
        cross_statistic.loc['fimpave'].mean(),
        cross_statistic.loc['fimpstd'].mean()
    ], index=[
        'criterion universal',
        'criterion true 1',
        'criterion true 2',
        'criterion false 1',
        'criterion false 2'
    ])

    return pd.DataFrame([
        pd.Series([
            cross_statistic.iloc[0][feat] < criteria['criterion universal'] and
            cross_statistic.iloc[1][feat] > criteria['criterion true 1'] and
            cross_statistic.iloc[2][feat] < criteria['criterion true 2']
            for feat in cross_statistic.columns
        ], index=cross_statistic.columns),
        pd.Series([
            cross_statistic.iloc[0][feat] < criteria['criterion universal'] and
            cross_statistic.iloc[3][feat] > criteria['criterion false 1'] and
            cross_statistic.iloc[4][feat] < criteria['criterion false 2']
            for feat in cross_statistic.columns
        ], index=cross_statistic.columns)
    ], index=['for true cases', 'for false cases'])


def select(data: Union[pd.DataFrame, List[pd.DataFrame]], droplist: List[str]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """Removes all specified features_ from dataset

    :param data: datasets to change
    :type data: |ipd.DataFrame|_ *or* |ilist|_ *of* |ipd.DataFrame|_
    :param droplist: features_ to remove
    :type droplist: |ilist|_ *of* |istr|_
    :return: modified datasets
    :rtype: pd.DataFrame_ or list_ of pd.DataFrame_
    """

    if isinstance(data, list):
        return [
            sample.drop(droplist, axis=1) for sample in data
        ]
    else:
        return data.drop(droplist, axis=1)


def selected_data(data: Union[pd.DataFrame, List[pd.DataFrame]], selection_map: pd.DataFrame) -> Tuple[Union[pd.DataFrame, List[pd.DataFrame]], Union[pd.DataFrame, List[pd.DataFrame]]]:
    """Select required features from dataset using `selection_map`_

        :param data: datasets to change
        :type data: |ipd.DataFrame|_ *or* |ilist|_ *of* |ipd.DataFrame|_
        :param selection_map: boolean map to distinguish `good features`_ from bad ones
        :type selection_map: |selection_map|_
        :return: modified dataset
        :rtype: pd.DataFrame_ or list_ of ipd.DataFrame_

        .. |selection_map| replace:: *selection_map*
    """

    droplist_true = [feat for feat in data[0].columns if not selection_map.iloc[0][feat]]
    droplist_false = [feat for feat in data[0].columns if not selection_map.iloc[1][feat]]

    return (
        select(data, droplist_true),
        select(data, droplist_false)
    )


def selected_statistic(cross_statistic: pd.DataFrame, selection_map: pd.DataFrame) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """removes all unimportant features from `cross statistical information`_ leaving it for `good features`_ only

    :param cross_statistic: `cross statistical information`_ to change
    :type cross_statistic: |cross_statistic|_
    :param selection_map: boolean map to distinguish `good features`_ from bad ones
    :type selection_map: |selection_map|_
    :return: `cross statistical information`_ for `good features`_
    :rtype: cross_statistic_

    .. |cross_statistic| replace:: *cross_statistic*
    """

    droplist = [feat for feat in cross_statistic.columns if not selection_map.loc['for true cases'][feat] and not selection_map.loc['for false cases'][feat]]

    return select(cross_statistic, droplist)


def choose_best(data: List[pd.DataFrame], cross_statistic: pd.DataFrame, for_: str, corr_thresholds: float = 0.9) -> pd.Index:
    """Compares features_ in given list of datasets and returns indices of more good_ and less correlated features_

        :param data: list of datasets to analyze
        :type data: |ilist|_ *of* |ipd.DataFrame|_
        :param cross_statistic: `cross statistical information`_ to change
        :type cross_statistic: |cross_statistic|_
        :param for_: for which data make comparison: "true" - for `true data`_, "false" - for `false data`_, "both" - for all data
        :type for_: str
        :param corr_thresholds: which features_ consider as correlated (if absolute correlation ratio less than given value, features_ considered as not correlated), default 0.9
        :type corr_thresholds: float, optional
        :return: indices of better features_
        :rtype: pd.Index_

        .. _pd.Index: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.html

        .. _best:
        .. note:: Best feature is a feature which satisfy to some relations in `cross statistical information`_:
            the lesser **mstd/mave**, the lesser **timpstd** for `true data`_ (or **fimpstd** for `false data`_)
            and the greater **timpave** for `true data`_ (or **fimpave** for `false data`_)
    """

    def worst_feature(feat1: str, feat2: str, for_: str, cross_statistic: pd.DataFrame) -> str:

        if for_ == 'both':
            if (cross_statistic.loc['timpave'][feat1] /
                (cross_statistic.loc['mstd/mave'][feat1] *
                 cross_statistic.loc['timpstd'][feat1])) * \
                    (cross_statistic.loc['fimpave'][feat1] /
                     cross_statistic.loc['fimpstd'][feat1]) > \
                    (cross_statistic.loc['timpave'][feat2] /
                     (cross_statistic.loc['mstd/mave'][feat2] *
                      cross_statistic.loc['timpstd'][feat2])) * \
                    (cross_statistic.loc['fimpave'][feat2] /
                     cross_statistic.loc['fimpstd'][feat2]):
                return feat2
            else:
                return feat1
        else:
            if cross_statistic.loc[for_[0] + 'impave'][feat1] / \
                    (cross_statistic.loc['mstd/mave'][feat1] *
                     cross_statistic.loc[for_[0] + 'impstd'][feat1]) > \
                    cross_statistic.loc[for_[0] + 'impave'][feat2] / \
                    (cross_statistic.loc['mstd/mave'][feat2] *
                     cross_statistic.loc[for_[0] + 'impstd'][feat2]):
                return feat2
            else:
                return feat1

    def choose(corrmap: pd.DataFrame, from_: Optional[int] = None) -> pd.Index:

        if from_ is None:
            from_ = corrmap.columns.tolist()[0]

        for column in corrmap.columns.tolist()[corrmap.columns.tolist().index(from_)::]:
            for row in corrmap.index.tolist()[corrmap.index.tolist().index(from_)::]:
                if row != column \
                        and (corrmap.loc[row][column] > corr_thresholds
                             or corrmap.loc[row][column] < -corr_thresholds):
                    worst = worst_feature(column, row, for_, cross_statistic)
                    from_ = corrmap.columns.tolist()[corrmap.columns.tolist().index(worst) + 1]
                    corrmap = corrmap.drop([worst], axis=0).drop([worst], axis=1)
                    return choose(corrmap, from_=from_)
        return corrmap.columns

    corrmap = pd.DataFrame(
        np.array([
            sample.corr().to_numpy() for sample in data
        ]).mean(axis=0),
        index=data[0].columns,
        columns=data[0].columns
    )

    return choose(corrmap)


def make_feature_selection(datasets: List[pd.DataFrame], target: str) -> List[pd.DataFrame]:
    """chooses the most `important features`_ from the list of datasets

    :param datasets: list of datasets to analyze
    :type datasets: |ilist|_ *of* |ipd.DataFrame|_
    :param target: name of the `target feature`_
    :type target: str
    :return: list of modified datasets
    :rtype: list_ of pd.DataFrame_
    """

    data, true_cases, false_cases = separate_datasets(datasets, target)
    cross_statistic = collect_cross_statistic(data, true_cases, false_cases)
    selection_map = make_selection_map(cross_statistic)
    data_for_true, data_for_false = selected_data(data, selection_map)
    best_true = choose_best(data_for_true, cross_statistic, 'true')
    best_false = choose_best(data_for_false, cross_statistic, 'false')
    best_raw = best_true.append(best_false).drop_duplicates()
    drop = [column for column in data[0].columns if column not in best_raw]
    best_data_common = select(data, drop)
    best = choose_best(best_data_common, cross_statistic, 'both')
    drop_ = [column for column in best_data_common[0].columns if column not in best]
    best_data = select(best_data_common, drop_)
    best_data = [
        pd.DataFrame(best_sample.to_numpy(), columns=best_sample.columns, index=sample[
            sample.columns[0]
        ].tolist())
        for best_sample, sample in zip(best_data, datasets)
    ]
    for best_sample, sample in zip(best_data, datasets):
        best_sample[target] = sample[target].tolist()

    return best_data
