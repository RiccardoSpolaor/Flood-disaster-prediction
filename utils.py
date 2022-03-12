import pandas as pd
from io import StringIO
from pgmpy.factors.discrete.CPD import TabularCPD
from typing import Dict, List


def pick_dictionary_subset(dictionary: Dict, keys: List[str]):
    return dict((k, dictionary[k]) for k in keys if k in dictionary)


def pcd_to_pandas(pcd: TabularCPD):
    # String representing the pretty print format of the PCD table
    tabulate_string = str(pcd)

    # Pandas dataframe creation
    data = (pd.read_csv(StringIO(tabulate_string), sep=r'\|', comment='+', engine='python',
                        # skiprows=pcd.variable_card,
                        skipinitialspace=True,
                        header=[i for i in range(0, len(pcd.state_names) - 1)]).dropna(how='all', axis=1))

    # The first column is turned into the index of the Dataframe
    data.set_index(data.columns[0], inplace=True, drop=True)
    data.index.name = None

    return data
