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
                return item[1] / (self._informativeness[contrary][key][item[0]] + item[1])

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

    def confusion_matrix(self) -> Dict[str, Tuple[int, int, int, int]]:

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

            tp, tn, fp, fn = 0, 0, 0, 0

            if key in self.informativeness['correct']['true']:
                tp += self.informativeness['correct']['true'][key]
            if key in self.informativeness['wrong']['true']:
                fp += self.informativeness['wrong']['true'][key]
            if key in self.informativeness['correct']['false']:
                tn += self.informativeness['correct']['false'][key]
            if key in self.informativeness['wrong']['false']:
                fn += self.informativeness['wrong']['false'][key]

            out.update({
                key: (tp, tn, fp, fn)
            })

        return out

    def acc(self) -> dict:

        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if tp + tn + fp + fn != 0:
                out.update({
                    subject_name: (tp + tn) / (tp + tn + fp + fn)
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def tnr(self) -> dict:

        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if tn + fp != 0:
                out.update({
                    subject_name: tn / (tn + fp)
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def tpr(self) -> dict:

        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if tp + fn != 0:
                out.update({
                    subject_name: tp / (tp + fn)
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def ppv(self) -> dict:

        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if tp + fp != 0:
                out.update({
                    subject_name: tp / (tp + fp)
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def npv(self) -> dict:

        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if tn + fn != 0:
                out.update({
                    subject_name: tn / (tn + fn)
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def fnr(self) -> dict:

        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if tp + fn != 0:
                out.update({
                    subject_name: fn / (tp + fn)
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def fpr(self) -> dict:

        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if fp + tn != 0:
                out.update({
                    subject_name: fp / (tp + fn)
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def fdr(self) -> dict:

        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if fp + tp != 0:
                out.update({
                    subject_name: fp / (tp + fp)
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def for_(self) -> dict:

        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if fn + tn != 0:
                out.update({
                    subject_name: fn / (tn + fn)
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def pt(self) -> dict:
        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if fn + tp != 0 and tn + fp != 0:
                tpr = tp / (tp + fn)
                tnr = tn / (tn + fp)
                out.update({
                    subject_name: (np.sqrt(tpr * (1 - tnr)) + tnr - 1) / (tpr + tnr - 1)
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def ba(self) -> dict:
        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if fn + tp != 0 and tn + fp != 0:
                tpr = tp / (tp + fn)
                tnr = tn / (tn + fp)
                out.update({
                    subject_name: (tpr + tnr) / 2
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def f1(self) -> dict:
        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if 2 * tp + fp + fn != 0:
                out.update({
                    subject_name: 2 * tp / (2 * tp + fp + fn)
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def bm(self) -> dict:
        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if fn + tp != 0 and tn + fp != 0:
                tpr = tp / (tp + fn)
                tnr = tn / (tn + fp)
                out.update({
                    subject_name: tpr + tnr - 1
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def fm(self) -> dict:
        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if fn + tp != 0 and tp + fp != 0:
                tpr = tp / (tp + fn)
                ppv = tp / (tp + fp)
                out.update({
                    subject_name: np.sqrt(tpr * ppv)
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def mk(self) -> dict:
        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]

            if fn + tn != 0 and tp + fp != 0:
                npv = tn / (tn + fn)
                ppv = tp / (tp + fp)
                out.update({
                    subject_name: npv + ppv - 1
                })
            else:
                out.update({
                    subject_name: None
                })

        return out

    def mcc(self) -> dict:
        out = dict()

        confusion = self.confusion_matrix()

        for subject_name in confusion:

            tp, tn, fp, fn = confusion[subject_name]
            den = np.sqrt(
                (tp + fp) *
                (tp + fn) *
                (tn + fp) *
                (tn + fn)
            )
            if den != 0:
                out.update({
                    subject_name: (tp * tn - fp * fn) / den
                })
            else:
                out.update({
                    subject_name: None
                })

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


class CrossInformativeness(Informativeness):
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

    def confusion_matrix(self) -> Dict[str, Tuple[int, int, int, int]]:

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

            tp, tn, fp, fn = 0, 0, 0, 0

            if key in self.informativeness['correct']['true']:
                tp += self.informativeness['correct']['true'][key].mean()
            if key in self.informativeness['wrong']['true']:
                fp += self.informativeness['wrong']['true'][key].mean()
            if key in self.informativeness['correct']['false']:
                tn += self.informativeness['correct']['false'][key].mean()
            if key in self.informativeness['wrong']['false']:
                fn += self.informativeness['wrong']['false'][key].mean()

            out.update({
                key: (tp, tn, fp, fn)
            })

        return out

    def sorted(self, type: str = 'growth', reverse: bool = False) -> dict:

        def rating_sort(prediction: str, key: str) -> Callable:

            contrary = 'correct' if prediction == 'wrong' else 'wrong'

            def sort_tool(item: Tuple[str, np.ndarray]) -> Union[int, float]:
                return item[1].mean() / \
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
