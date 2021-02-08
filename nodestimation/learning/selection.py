import pandas as pd
import numpy as np


def collect_statistic(data):
    means = pd.Series([data[feat].mean() for feat in data.columns], index=data.columns)
    stds = pd.Series([data[feat].std() for feat in data.columns], index=data.columns)
    upper_bound = pd.Series([mean + std for mean, std in zip(means, stds)], index=data.columns)
    lower_bound = pd.Series([mean - std for mean, std in zip(means, stds)], index=data.columns)

    return pd.DataFrame([means, stds, upper_bound, lower_bound],
                        index=['mean', 'stdev', 'm+std', 'm-std'])


def compute_importance(data, statistic):

    def append_series(df, series, index=None):

        if not isinstance(series, list):
            series = [series]

        if index is not None:

            if not isinstance(index, list):
                index = [index]

            index = np.concatenate(df.columns, np.array(index), axis=0)

        df_series = [df.iloc[i] for i in range(df.shape[0])]

        for s in series:
            df_series.append(s)

        return pd.DataFrame(df_series, index=index)

    def appstart_series(df, series, index=None):

        if not isinstance(series, list):
            series = [series]

        if index is not None:

            if not isinstance(index, list):
                index = [index]

            index = np.concatenate(np.array(index), df.columns, axis=0)

        df_series = [df.iloc[i] for i in range(df.shape[0])]

        for s in df_series:
            series.append(s)

        return pd.DataFrame(series, index=index)

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


def separate_datasets(datasets, target):
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


def collect_cross_statistic(data, true_cases, false_cases):
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
        'mean, std / mean, averaged', 'true importance, averaged', 'true importance, std', 'false importance, averaged', 'false importance, std'
    ])


def make_selection_map(cross_statistic):
    criteria = pd.Series([
        cross_statistic.loc['mean, std / mean, averaged'].mean(),
        cross_statistic.loc['true importance, averaged'].mean(),
        cross_statistic.loc['true importance, std'].mean(),
        cross_statistic.loc['false importance, averaged'].mean(),
        cross_statistic.loc['false importance, std'].mean()
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


def select(data, droplist):
    if isinstance(data, list):
        return [
            sample.drop(droplist, axis=1) for sample in data
        ]
    else:
        return data.drop(droplist, axis=1)


def selected_data(data, selection_map):
    droplist_true = [feat for feat in data[0].columns if not selection_map.loc['for true cases'][feat]]
    droplist_false = [feat for feat in data[0].columns if not selection_map.loc['for false cases'][feat]]

    return (
        select(data, droplist_true),
        select(data, droplist_false)
    )


def selected_statistic(cross_statistic, selection_map):
    droplist = [feat for feat in cross_statistic.columns if not selection_map.loc['for true cases'][feat] and not selection_map.loc['for false cases'][feat]]

    return select(cross_statistic, droplist)


def choose_best(data, cross_statistic, for_, corr_thresholds=0.9):

    def worst_feature(feat1, feat2, for_, cross_statistic):

        if for_ == 'both':
            if (cross_statistic.loc['true importance, averaged'][feat1] /
                (cross_statistic.loc['mean, std / mean, averaged'][feat1] *
                 cross_statistic.loc['true importance, std'][feat1])) * \
                    (cross_statistic.loc['false importance, averaged'][feat1] /
                     cross_statistic.loc['false importance, std'][feat1]) > \
                    (cross_statistic.loc['true importance, averaged'][feat2] /
                     (cross_statistic.loc['mean, std / mean, averaged'][feat2] *
                      cross_statistic.loc['true importance, std'][feat2])) * \
                    (cross_statistic.loc['false importance, averaged'][feat2] /
                     cross_statistic.loc['false importance, std'][feat2]):
                return feat2
            else:
                return feat1
        else:
            if cross_statistic.loc[for_ + ' importance, averaged'][feat1] / \
                    (cross_statistic.loc['mean, std / mean, averaged'][feat1] *
                     cross_statistic.loc[for_ + ' importance, std'][feat1]) > \
                    cross_statistic.loc[for_ + ' importance, averaged'][feat2] / \
                    (cross_statistic.loc['mean, std / mean, averaged'][feat2] *
                     cross_statistic.loc[for_ + ' importance, std'][feat2]):
                return feat2
            else:
                return feat1

    def choose(corrmap, from_=None):

        if from_ is None:
            from_ = corrmap.columns.tolist()[0]

        for column in corrmap.columns.tolist()[corrmap.columns.tolist().index(from_)::]:
            for row in corrmap.index.tolist()[corrmap.index.tolist().index(from_)::]:
                if row != column \
                        and (corrmap.loc[row][column] > corr_thresholds \
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


def make_feature_selection(datasets, target):
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
