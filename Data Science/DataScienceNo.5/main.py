import pandas as pd
import numpy as np
df = pd.read_csv(r'C:\Users\eytan\PycharmProjects\DataScienceNo.5\survivor.csv')

# print(df.duplicated('Ticket').sum())
# print(df.shape[0])
# print(df.Ticket.unique().size)


def remove_corrupt_rows(df, num_max_missing_cols):
    df.dropna(axis=0,thresh= df.columns.size - num_max_missing_cols,inplace=True)
    return df

"""The following is expected:
--- Complete the 'remove_duplicatives' function to return a copy of a the given 'df' dataframe,
    after removing the duplicatives.

    If the 'col_name' parameter is None, you should remove full duplicative rows.
    Otherwise, identify and remove duplicates in regard to the column 'col_name'.

    Note that for a single duplication, you should keep only the first occurrence."""


def remove_duplicatives(df, col_name=None):
    df.drop_duplicates(subset=col_name ,inplace=True)

    return df


def replace_missing_values(df, col_to_def_val_dict):
    df.fillna(value=col_to_def_val_dict, inplace=True)

    for i in df.columns:

        if df[i].dtype != 'object':
            t = df[i].median()
            df[i].fillna(t, inplace=True)
        else:
            df[i].fillna(df[i].mode()[0], inplace=True)

    return df



df =pd.read_csv(r'C:\Users\eytan\PycharmProjects\DataScienceNo.5\imdb_movies.csv')
col_name = 'movie_imdb_link'
col_to_def_val = {'director_name':'unknown',
    'actor_1_name':'unknown', 'actor_2_name':'unknown', 'actor_3_name':'unknown', 'genres':'unknown',
    'plot_keywords':'unknown', 'movie_title':'unknown', 'movie_imdb_link':'unknown', 'country':'unknown'}

df_cln = remove_duplicatives(df, col_name)
df_rem_corrupt = remove_corrupt_rows(df_cln, 3)
df_rpl_missing = replace_missing_values(df_rem_corrupt, col_to_def_val)


def outlier_detection_iqr(df):
    for i in df.columns:
        if df[i].dtype != 'object':
            print(df[i].isnull().sum())
            Q1 = np.percentile(df[i], 25)  # Why work hard when you can copy paste
            Q3 = np.percentile(df[i], 75)  # Why work hard when you can copy paste
            IQR = Q3 - Q1  # Why work hard when you can copy paste

            df[i].mask(((df[i] < Q1 - 1.5 * IQR) | (df[i] > Q3 + 1.5 * IQR)), np.nan, inplace=True)
            print(df[i].isnull().sum())
            
    return df


col_name = 'movie_imdb_link'

col_to_def_val = {'director_name':'unknown',
    'actor_1_name':'unknown', 'actor_2_name':'unknown', 'actor_3_name':'unknown', 'genres':'unknown',
    'plot_keywords':'unknown', 'movie_title':'unknown', 'movie_imdb_link':'unknown', 'country':'unknown'}
df_imdb_movies = df
df_cln = remove_duplicatives(df_imdb_movies, col_name)
df_rem_corrupt = remove_corrupt_rows(df_cln, 3)
df_rpl_missing = replace_missing_values(df_rem_corrupt, col_to_def_val)
df_outlier_rem = outlier_detection_iqr(df_rpl_missing)
df_outliers = df_outlier_rem.isnull().sum().to_frame('iqr_outliers')
#print(df_outliers['iqr_outliers']['num_critic_for_reviews'])

