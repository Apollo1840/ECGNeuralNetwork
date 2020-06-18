TEST_DATA_PATH = 'data/beats_img/test.txt'
TEST_DATA_PATH_CROSS = 'data/beats_img/test_DS2.txt'

DS_bank = {

    # ML-II
    "normal": {
        "DS1": [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
                122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                223, 230],
        "DS2": [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
                210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
                233, 234]
    },

    # ML-II + V1
    "reduced": {
        "DS1": [101, 106, 108, 109, 112, 115, 118, 119, 201, 203,
                205, 207, 208, 209, 215, 220, 223, 230],
        "DS2": [105, 111, 113, 121, 200, 202, 210, 212, 213, 214,
                219, 221, 222, 228, 231, 232, 233, 234]
    }
}


MITBIH_CLASSES = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', 'P', '/', 'f', 'u']
MITBIH_CLASSES_REDUCED = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']  # , 'P', '/', 'f', 'u']

AAMI2MITBIH_MAPPING = {
    "N": ['N', 'L', 'R'],
    "SVEB": ['A', 'a', 'J', 'S', 'e', 'j'],
    "VEB": ['V', 'E'],
    "F": ['F'],
    "Q": ['P', '/', 'f', 'u'],
}

AAMI2MITBIH_MAPPING_REDUCED = {
    "N": ['N', 'L', 'R'],
    "SVEB": ['A', 'a', 'J', 'S', 'e', 'j'],
    "VEB": ['V', 'E'],
    "F": ['F'],
    # "Q": ['P', '/', 'f', 'u'],
}

# AAMI_CLASSES_REDUCED = sorted(AAMI2MITBIH_MAPPING_REDUCED.keys())
AAMI_CLASSES_REDUCED = ['F', 'N', 'SVEB', 'VEB']

MITBIH2TYPE2_MAPPING = {".": "NOR",
                        "N": "NOR",
                        "V": "PVC",
                        "/": "PAB",
                        "L": "LBB",
                        "R": "RBB",
                        "A": "APC",  # APB
                        "a": "aPC",  # aAPB
                        "e": "AEB",
                        "J": "JPB",
                        "j": "JEB",
                        "S": "SVP",  # SVPB
                        "!": "VFW",
                        "F": "FUS",  # Fusion
                        "E": "VEB"}

# TYPE2_CLASSES_REDUCED = sorted(MITBIH2TYPE2_MAPPING.values())
TYPE2_CLASSES = ['AEB',
                 'APC',
                 'FUS',
                 'JEB',
                 'JPB',
                 'LBB',
                 'NOR',
                 'NOR',
                 'PAB',
                 'PVC',
                 'RBB',
                 'SVP',
                 'VEB',
                 'VFW',
                 'aPC']

MITBIH2TYPE2_MAPPING_REDUCED = {".": "NOR",
                                "N": "NOR",
                                "V": "PVC",
                                "/": "PAB",
                                "L": "LBB",
                                "R": "RBB",
                                "A": "APC",
                                "!": "VFW",
                                "E": "VEB"}

# TYPE2_CLASSES_REDUCED = sorted(MITBIH2TYPE2_MAPPING_REDUCED.values())
TYPE2_CLASSES_REDUCED = ['APC', 'LBB', 'NOR', 'NOR', 'PAB', 'PVC', 'RBB', 'VEB', 'VFW']

# mitclasses2type2classes = lambda mit_classes: [MITBIH2TYPE2_MAPPING[mit_cls] for mit_cls in mit_classes if mit_cls in MITBIH2TYPE2_MAPPING]
# mitclasses2type2classes_reduced = lambda mit_classes: [MITBIH2TYPE2_MAPPING_REDUCED[mit_cls] for mit_cls in mit_classes if mit_cls in MITBIH2TYPE2_MAPPING_REDUCED]
# AAMI2TYPE2_MAPPING_REDUCED = dict([(aami_label, mitclasses2type2classes(mitclasses)) for aami_label, mitclasses in AAMI2MITBIH_MAPPING_REDUCED.items()])
# AAMI2TYPE2_MAPPING_REDUCED2 = dict([(aami_label, mitclasses2type2classes_reduced(mitclasses)) for aami_label, mitclasses in AAMI2MITBIH_MAPPING_REDUCED.items()])
AAMI2TYPE2_MAPPING_REDUCED = {
    'N': ['NOR', 'LBB', 'RBB'],
    'SVEB': ['APC', 'aPC', 'JPB', 'SVP', 'AEB', 'JEB'],
    'VEB': ['PVC', 'VEB'],
    'F': ['FUS']}

AAMI2TYPE2_MAPPING_REDUCED2 = {
    "N": ["NOR", "LBB", "RBB"],
    "SVEB": ["APC"],
    "VEB": ["VEB", "PVC"],
    "F": []
}
