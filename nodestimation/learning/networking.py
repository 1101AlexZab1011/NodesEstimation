import time
from typing import *
import networkx as nx
import pandas as pd

import nodestimation.learning.modification as lmd
from nodestimation.project.subject import Subject


def sparse_graph(g: nx.Graph) -> nx.Graph:
    con = nx.to_numpy_matrix(g)
    out = nx.from_numpy_matrix(
        lmd.suppress(
            pd.DataFrame(
                con
            ),
            trigger=con.mean().mean(),
            optimal=0
        ).to_numpy()
    )
    mapping = {node: label_name for node, label_name in zip(out, g)}
    out = nx.relabel_nodes(out, mapping)
    return out


def graph_to_connectome(g: nx.Graph) -> pd.DataFrame:
    return pd.DataFrame(
        nx.to_numpy_matrix(g),
        index= g.nodes,
        columns = g.nodes
    )


def labels_for_hemispheres(g: nx.Graph) -> Tuple[List[str], List[str]]:
    labels_rh, labels_lh = list(), list()
    for node in g.nodes:
        if 'lh' in node:
            labels_lh.append(node)
        elif 'rh' in node:
            labels_rh.append(node)
        else:
            raise ValueError(f'Wrong node name: {node}')
    return labels_lh, labels_rh


def hemispheres_division_modularity(g: nx.Graph) -> float:
    labels_lh, labels_rh = labels_for_hemispheres(g)
    return nx.algorithms.community.quality.modularity(g, [labels_lh, labels_rh])


def hemispheres_division_performance(g: nx.Graph) -> float:
    labels_lh, labels_rh = labels_for_hemispheres(g)
    return nx.algorithms.community.quality.performance(g, [labels_lh, labels_rh])


def graph_to_hemispheres(g: nx.Graph) -> Tuple[nx.Graph, nx.Graph]:
    labels_lh, labels_rh = labels_for_hemispheres(g)
    return g.subgraph(labels_lh), g.subgraph(labels_rh)


def smallworldness(g: nx.Graph) -> Tuple[float, float]:
    return nx.algorithms.smallworld.sigma(g), nx.algorithms.smallworld.omega(g)


def metric_for_hemispheres(subjects: List[Subject], metric: Callable, show_info: bool = False, **kwargs) -> pd.DataFrame:
    dataset = pd.DataFrame()

    for subject in subjects:
        start = time.time()
        lupd, rupd = dict(), dict()

        for node in subject.nodes:
            if node.type == 'resected' and 'rh' in node.label.name:
                lupd.update({'resected': False})
                rupd.update({'resected': True})
                break

            elif node.type == 'resected' and 'lh' in node.label.name:
                lupd.update({'resected': True})
                rupd.update({'resected': False})
                break

        for freq in subject.connectomes:
            for method in subject.connectomes[freq]:
                label_names = list(subject.connectomes[freq][method].index)
                mapping = {
                    i: label_name
                    for i, label_name in zip(
                        range(len(label_names)),
                        label_names
                    )
                }
                G = sparse_graph(
                    nx.convert_matrix.from_numpy_matrix(
                        subject.connectomes[freq][method].to_numpy()
                    )
                )
                G = nx.relabel_nodes(G, mapping)
                lh, rh = graph_to_hemispheres(G)
                lupd.update({f'{metric.__name__}_for_{method}_{freq}': metric(lh, **kwargs)})
                rupd.update({f'{metric.__name__}_for_{method}_{freq}': metric(rh, **kwargs)})

        dataset = lmd.append_series(dataset, pd.Series(lupd), index=f'{subject.name}_lh')
        dataset = lmd.append_series(dataset, pd.Series(rupd), index=f'{subject.name}_rh')
        if show_info:
            print(f'{subject.name}: DONE, RUNTIME: {time.time() - start}')

    return dataset
