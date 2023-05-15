# @Time : 2023/5/4 18:13 
# @Author : zhongyu 
# @File : itu_data_handling.py
from jddb.meta_db import MetaDB
from jddb.file_repo import FileRepo
from jddb.processor import Shot
import numpy as np
import matplotlib.pyplot as plt
import json

# %% connect to the MetaDB
connection_str = {
    "host": "localhost",
    "port": 8107,
"username":"DDBUser",
"password":"tokamak!",
    "database": "DDB"
}
collection = "tags"
with open('my_list.json', 'r') as f:
    json_string = f.read()
    loaded_list = json.loads(json_string)

db = MetaDB(connection_str, collection)

# %%
#  find all the shot with shot_no in range [10000, 20000] && [IP, BT] tags available && is disruption
shot_list = [shot for shot in range(1000000, 2000000)]
complete_disruption_shots = db.query_valid(
    shot_list=shot_list,
    label_true=loaded_list
)
print(complete_disruption_shots)
json_string = json.dumps(complete_disruption_shots)
with open('shots.json', 'w') as f:
    f.write(json_string)
print(len(complete_disruption_shots))
# %%
# find all the shot with IP>160kA, 0.45s<Tcq<0.5s  && is disruption && with those diagnostis [ip, bt] available
ip_range = [160, None]
tcq_range = [0.45, 0.5]
chosen_shots = db.query_range(
    ["IpFlat", "DownTime"],
    lower_limit=[ip_range[0], tcq_range[0]],
    upper_limit=[ip_range[1], tcq_range[1]],
    shot_list=complete_disruption_shots
)
print(chosen_shots)
print(len(chosen_shots))