import pandas as pd
from io import StringIO


def pick_dictionary_subset(dictionary, keys):
    return dict((k, dictionary[k]) for k in keys if k in dictionary)


def pcd_to_pandas(pcd):
    tabulate_string = str(pcd)
    # header = (pd.read_csv(StringIO(tabulate_string), sep=r'\|' , comment='+', engine='python',
    #                     nrows=pcd.variable_card, header=None)
    #            .dropna(how='all', axis=1))
    data = (pd.read_csv(StringIO(tabulate_string), sep=r'\|', comment='+', engine='python',
                        # skiprows=pcd.variable_card,
                        skipinitialspace=True,
                        header=[i for i in range(0, pcd.variable_card - 1)]).dropna(how='all', axis=1))

    # print(header.T[0])
    # data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Replace data's RangeIndex with column labels.
    # data.columns = data.columns.map(header.T[0].str.strip().to_dict())
    data.set_index(data.columns[0], inplace=True, drop=True)
    data.index.name = None
    return data
