from abc import ABC
from copy import deepcopy
from typing import *

import numpy as np


class AbstractInformativeness(ABC):

    def __init__(self):
        self._informativeness = {
                'correct': {'true': dict(), 'false': dict()},
                'wrong': {'true': dict(), 'false': dict()}
            }

    @property
    def informativeness(self):
        return self

    def sorted(self, type: str = 'growth', reverse: bool = False) -> dict:

        def rating_sort(prediction: str, key: str) -> Callable:

            contrary = 'correct' if prediction == 'wrong' else 'wrong'

            def sort_tool(item: Tuple[str, int]) -> Union[int, float]:
                return item[1]/(self._informativeness[contrary][key][item[0]] + item[1])

            return sort_tool


        if type == 'growth':
            return {
                correctness: {
                    positiveness:
                        dict(
                            sorted(
                                self._informativeness[correctness][positiveness].items(),
                                key=lambda item: item[1],
                                reverse=reverse
                            )
                        )
                    for positiveness in self._informativeness[correctness]
                }
                for correctness in self._informativeness
            }

        elif type == 'alphabet':
            return {
                correctness: {
                    positiveness:
                        dict(
                            sorted(
                                self._informativeness[correctness][positiveness].items(),
                                key=lambda item: item[0],
                                reverse=reverse
                            )
                        )
                    for positiveness in self._informativeness[correctness]
                }
                for correctness in self._informativeness
            }

        elif type == 'rating':
            return {
                correctness: {
                    positiveness:
                        dict(
                            sorted(
                                self._informativeness[correctness][positiveness].items(),
                                key=rating_sort(correctness, positiveness),
                                reverse=reverse
                            )
                        )
                    for positiveness in self._informativeness[correctness]
                }
                for correctness in self._informativeness
            }

    def copy(self):
        return deepcopy(self)

class Informativeness(AbstractInformativeness):
    def __init__(self):
        super().__init__()

    @property
    def informativeness(self) -> dict:
        return self._informativeness

    @informativeness.setter
    def informativeness(self, sample: Tuple[str, bool, str]):
        sample_name, positive, group = sample
        container = self._informativeness[group][str(positive).lower()]

        if sample_name in container:
            container[sample_name] += 1
        else:
            container.update({sample_name: 1})

    def acc(self) -> dict:

        out = dict()

        keys = list(
            set(
                list(
                    self.informativeness['correct']['true'].keys()
                ) + list(
                    self.informativeness['correct']['false'].keys()
                ) + list(
                    self.informativeness['wrong']['true'].keys()
                ) + list(
                    self.informativeness['wrong']['false'].keys()
                )
            )
        )

        for key in keys:

            tp, tn, p, n = 0, 0, 0, 0

            if key in self.informativeness['correct']['true']:
                tp += self.informativeness['correct']['true'][key]
                p += self.informativeness['correct']['true'][key]
            if key in self.informativeness['correct']['false']:
                tn += self.informativeness['correct']['false'][key]
                p += self.informativeness['correct']['false'][key]
            if key in self.informativeness['wrong']['true']:
                n += self.informativeness['wrong']['true'][key]
            if key in self.informativeness['wrong']['false']:
                n += self.informativeness['wrong']['false'][key]

            if p+n != 0:
                out.update({
                    key: (tp + tn)/(p + n)
                })
            else:
                out.update({
                    key: None
                })

        return out


    def ppv(self) -> dict:
        out = dict()

        for sample in self.informativeness['correct']['true']:
            if sample in self.informativeness['wrong']['true']:
                out.update({
                    sample: self.informativeness['correct']['true'][sample]/
                            (self.informativeness['correct']['true'][sample] +
                             self.informativeness['wrong']['true'][sample])
                })
            else:
                out.update({sample: 1})

        for sample in self.informativeness['wrong']['true']:
            if sample not in self.informativeness['correct']['true']:
                out.update({sample: 0})

        return out


    def npv(self) -> dict:
        out = dict()

        for sample in self.informativeness['correct']['false']:
            if sample in self.informativeness['wrong']['false']:
                out.update({
                    sample: self.informativeness['correct']['false'][sample]/
                            (self.informativeness['correct']['false'][sample] +
                             self.informativeness['wrong']['false'][sample])
                })
            else:
                out.update({sample: 1})

        for sample in self.informativeness['wrong']['false']:
            if sample not in self.informativeness['correct']['false']:
                out.update({sample: 0})

        return out


class NodesInformativeness(Informativeness):
    def __init__(self):
        super().__init__()

    @property
    def informativeness(self) -> dict:
        return self._informativeness

    @informativeness.setter
    def informativeness(self, sample: Tuple[str, bool, str]):
        sample_name, positive, group = sample
        node_name = sample_name[5:]
        container = self._informativeness[group][str(positive).lower()]

        if node_name in container:
            container[node_name] += 1
        else:
            container.update({node_name: 1})


class SubjectsInformativeness(Informativeness):
    def __init__(self):
        super().__init__()

    @property
    def informativeness(self) -> dict:
        return self._informativeness

    @informativeness.setter
    def informativeness(self, sample: Tuple[str, bool, str]):
        sample_name, positive, group = sample
        subject_name = sample_name[:4]
        container = self._informativeness[group][str(positive).lower()]

        if subject_name in container:
            container[subject_name] += 1
        else:
            container.update({subject_name: 1})


class CrossInformativeness(AbstractInformativeness):
    def __init__(self):
        super().__init__()

    @property
    def informativeness(self) -> dict:
        return self._informativeness

    @informativeness.setter
    def informativeness(self, other: Informativeness):
        for correctness in other.informativeness:
            if correctness in self.informativeness:
                for positiveness in other.informativeness[correctness]:
                    if positiveness in self.informativeness[correctness]:
                        for name in other.informativeness[correctness][positiveness]:
                            if name in self.informativeness[correctness][positiveness]:
                                self.informativeness[correctness][positiveness][name] = np.append(
                                    self.informativeness[correctness][positiveness][name],
                                    other.informativeness[correctness][positiveness][name]
                                )
                            else:
                                self.informativeness[correctness][positiveness].update({
                                    name: np.array([
                                        other.informativeness[correctness][positiveness][name]
                                    ])
                                })
                    else:
                        self.informativeness[correctness].update({
                            positiveness: self.numeric_keys_to_numpy(
                                other.informativeness[correctness][positiveness]
                            )
                        })

    def numeric_keys_to_numpy(self, dictionary: dict) -> dict:
        for key in dictionary:
            if isinstance(dictionary[key], dict):
                self.numeric_keys_to_numpy(dictionary[key])
            elif isinstance(dictionary[key], int) or isinstance(dictionary[key], float):
                dictionary[key] = np.array([dictionary[key]])
            elif isinstance(dictionary[key], list):
                dictionary[key] = np.array(dictionary[key])
            elif isinstance(dictionary[key], np.ndarray):
                continue

        return dictionary

    def mean(self):
        def numpy_to_mean(dictionary: Dict[str, np.ndarray]):
            return {
                key: dictionary[key].mean()
                for key in dictionary
            }

        def dict_to_mean(dictionary: dict):
            return {
                correctness: {
                    positiveness: numpy_to_mean(dictionary[correctness][positiveness])
                    for positiveness in dictionary[correctness]
                }
                for correctness in dictionary
            }

        return dict_to_mean(self.informativeness)

    def std(self):
        def numpy_to_std(dictionary: Dict[str, np.ndarray]) -> dict[str, float]:
            return {
                key: dictionary[key].std()
                for key in dictionary
            }

        def dict_to_std(dictionary: dict) -> dict[str, dict[str, float]]:
            return {
                correctness: {
                    positiveness: numpy_to_std(dictionary[correctness][positiveness])
                    for positiveness in dictionary[correctness]
                }
                for correctness in dictionary
            }

        return dict_to_std(self.informativeness)

    def acc(self) -> dict:

        out = dict()

        keys = list(
            set(
                list(
                    self.informativeness['correct']['true'].keys()
                ) + list(
                    self.informativeness['correct']['false'].keys()
                ) + list(
                    self.informativeness['wrong']['true'].keys()
                ) + list(
                    self.informativeness['wrong']['false'].keys()
                )
            )
        )

        for key in keys:

            tp, tn, p, n = 0, 0, 0, 0

            if key in self.informativeness['correct']['true']:
                tp += self.informativeness['correct']['true'][key].mean()
                p += self.informativeness['correct']['true'][key].mean()
            if key in self.informativeness['correct']['false']:
                tn += self.informativeness['correct']['false'][key].mean()
                p += self.informativeness['correct']['false'][key].mean()
            if key in self.informativeness['wrong']['true']:
                n += self.informativeness['wrong']['true'][key].mean()
            if key in self.informativeness['wrong']['false']:
                n += self.informativeness['wrong']['false'][key].mean()

            if p+n != 0:
                out.update({
                    key: (tp + tn)/(p + n)
                })
            else:
                out.update({
                    key: None
                })

        return out

    def ppv(self) -> dict:
        out = dict()

        for sample in self.informativeness['correct']['true']:
            if sample in self.informativeness['wrong']['true']:
                out.update({
                    sample: self.informativeness['correct']['true'][sample].mean()/
                            (self.informativeness['correct']['true'][sample].mean() +
                             self.informativeness['wrong']['true'][sample].mean())
                })
            else:
                out.update({sample: 1})

        for sample in self.informativeness['wrong']['true']:
            if sample not in self.informativeness['correct']['true']:
                out.update({sample: 0})

        return out

    def npv(self) -> dict:
        out = dict()

        for sample in self.informativeness['correct']['false']:
            if sample in self.informativeness['wrong']['false']:
                out.update({
                    sample: self.informativeness['correct']['false'][sample].mean()/
                            (self.informativeness['correct']['false'][sample].mean() +
                             self.informativeness['wrong']['false'][sample].mean())
                })
            else:
                out.update({sample: 1})

        for sample in self.informativeness['wrong']['false']:
            if sample not in self.informativeness['correct']['false']:
                out.update({sample: 0})

        return out

    def sorted(self, type: str = 'growth', reverse: bool = False) -> dict:

        def rating_sort(prediction: str, key: str) -> Callable:

            contrary = 'correct' if prediction == 'wrong' else 'wrong'

            def sort_tool(item: Tuple[str, np.ndarray]) -> Union[int, float]:
                return item[1].mean()/\
                       (
                               self._informativeness[contrary][key][item[0]].mean() +
                               item[1].mean()
                       )

            return sort_tool

        if type == 'growth':
            return {
                correctness: {
                    positiveness:
                        dict(
                            sorted(
                                self._informativeness[correctness][positiveness].items(),
                                key=lambda item: item[1].mean(),
                                reverse=reverse
                            )
                        )
                    for positiveness in self._informativeness[correctness]
                }
                for correctness in self._informativeness
            }

        elif type == 'alphabet':
            return {
                correctness: {
                    positiveness:
                        dict(
                            sorted(
                                self._informativeness[correctness][positiveness].items(),
                                key=lambda item: item[0],
                                reverse=reverse
                            )
                        )
                    for positiveness in self._informativeness[correctness]
                }
                for correctness in self._informativeness
            }

        elif type == 'rating':
            return {
                correctness: {
                    positiveness:
                        dict(
                            sorted(
                                self._informativeness[correctness][positiveness].items(),
                                key=rating_sort(correctness, positiveness),
                                reverse=reverse
                            )
                        )
                    for positiveness in self._informativeness[correctness]
                }
                for correctness in self._informativeness
            }
