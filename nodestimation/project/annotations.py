from typing import *

import networkx as nx
import numpy as np
import pandas as pd

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

Graphs = Dict[
    str, Dict[
        str, nx.Graph
    ]
]

Connectomes = Dict[
    str, Dict[
        str, pd.DataFrame
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
