from typing import *

import numpy as np

SubjectTree = Dict[
    str, Union[
        str, List[str]
    ]
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
