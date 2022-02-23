# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# === CELL TYPE: IMPORTS AND SETUP

import pandas as pd
import numpy as np

import requests
import json
# ---
import os # for testing only
# ---

def get_res_for_API_query(included_field_value_pairs, excluded_field_value_pairs):
    url = "http://moocdsand.ml:8080/api/records/1.0/search?dataset=chocolate-bars"
    for include in included_field_value_pairs:
        url += "&refine." + include[0] + "=" +  include[1]
    for exclude in excluded_field_value_pairs:
        url += "&exclude." + exclude[0] + "=" +  exclude[1]
    return requests.get(url)

def load_query_result_to_df(api_query_res):
    json_dict = api_query_res.json()

    json = [i['fields'] for i in json_dict['records']]

    return pd.DataFrame.from_dict(json)




include_field_value_pairs = [('review_date', '2013'),
                         ('company_location', 'Belgium')]
exlude_field_value_pairs = [('rating', '3.25'), ('bean_type', 'Trinitario')]
res_query = get_res_for_API_query(include_field_value_pairs, exlude_field_value_pairs)
df_results = load_query_result_to_df(res_query)