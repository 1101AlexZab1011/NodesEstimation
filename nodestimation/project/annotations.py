from typing import *

import numpy as np

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

LabelsFeatures = Dict[
    str, Dict[
        str, Dict[
            str, float
        ]
    ]
]

NodeFeatures = Dict[
                str, Dict[
                    str, float
                ]
            ]
