import pandas as pd
import os


def load_csv(file_name):
    dirname = os.path.dirname(__file__)
    df =pd.read_csv(os.path.join(dirname,file_name))
    return df

def get_number_of_rows(dataframe):
    return dataframe.shape[0]


def get_number_of_columns(dataframe):
    return len(dataframe.columns)


def get_rows_in_range(dataframe, first_row, last_row):
    df = dataframe.iloc[first_row:last_row]
    return df


def get_columns_in_range(dataframe, first_column, last_column):
    return dataframe.iloc[ :, first_column:last_column]


def select_rows_by_cell_val(dataframe, col_name, matching_val):
    return dataframe[dataframe[col_name] == matching_val]


def select_rows_w_vals_in_range(dataframe, col_name, lower_range, higher_range):
    return dataframe[dataframe[col_name].between(lower_range, higher_range)]







file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
df_cocoa = load_csv(file_name)
print(df_cocoa)
