import numpy as np
import pandas as pd
from io import StringIO
from pgmpy.factors.discrete.CPD import TabularCPD
from typing import Dict, List, Optional


def get_state_names(variable: str, state_names_dictionary: Dict[str, List[str]],
                    evidence_dictionary: Dict[str, List[str]]):
    evidence: Optional[List[str]] = evidence_dictionary[variable]
    if evidence is None:
        variable_and_evidence_list = [variable]
    else:
        variable_and_evidence_list = [variable] + evidence

    return dict((v, state_names_dictionary[v]) for v in variable_and_evidence_list if v in state_names_dictionary)


def cpd_to_pandas(pcd: TabularCPD):
    # String representing the pretty print format of the PCD table
    tabulate_string = str(pcd)

    # Set the header according to the number of parents of the variable
    header: Optional[List[int]] = None if len(pcd.state_names) == 1 else np.arange(0, len(pcd.state_names) - 1).tolist()

    # Create the Pandas Dataframe
    data = (pd.read_csv(StringIO(tabulate_string),
                        sep=r'\|',
                        comment='+',
                        engine='python',
                        skipinitialspace=True,
                        header=header).dropna(how='all', axis=1)
            )

    # Remove the column name if the variable has no parents
    if header is None:
        data.columns = [data.columns[0], '']

    # Turn the first column into the index of the Dataframe
    data.set_index(data.columns[0], inplace=True, drop=True)
    data.index.name = None

    return data


def get_evidence_card(variable: str, state_names_dictionary: Dict[str, List[str]],
                      evidence_dictionary: Dict[str, List[str]]):
    evidence: Optional[List[str]] = evidence_dictionary[variable]
    if evidence is None:
        return None
    else:
        return [len(state_names_dictionary[e]) for e in evidence]
