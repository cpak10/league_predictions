import pandas as pd
from tensorflow import keras
import joblib

# inputs
team_blue = "Evil Geniuses"
team_red = "Cloud9"

# update the date before running
date = "20220819"
data_oe = pd.read_csv(f"oe_extract_{date}.csv")

"""
# saving new data
link_oe_data = f"https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com/2022_LoL_esports_match_data_from_OraclesElixir_{date}.csv"
data_oe = pd.read_csv(link_oe_data)
data_oe.to_csv(f"oe_extract_{date}.csv", index = False)
"""

# bring down only the relevant columns
data_oe_correlated = data_oe[[
    "inhibitors",
    "opp_inhibitors",
    "earned gpm",
    "gspd",
    "xpdiffat10",
    "golddiffat10",
    "xpdiffat15",
    "csdiffat10",
    "firstbaron",
    "firsttothreetowers",
    "firstmidtower",
    "firsttower",
    "teamname",
    "date",
    "position"
]]
print(f"NOTE: Observations in raw data: {data_oe_correlated.shape[0]}")

# only keep team observations
data_oe_narrow_row = data_oe_correlated[(data_oe_correlated["position"] == "team") & (data_oe_correlated["teamname"].isin([team_blue, team_red]))]
print(f"NOTE: Observations after keeping team statistics: {data_oe_narrow_row.shape[0]}")

# drop position, drop missing values
data_oe_drop_column = data_oe_narrow_row.drop(columns = ["position"])
data_oe_drop_missing = data_oe_drop_column.dropna()
print(f"NOTE: Observations after dropping missing values: {data_oe_drop_missing.shape[0]}")

# create the sort and assign an index
data_oe_new_key = data_oe_drop_missing.sort_values(by = ["teamname", "date"])
data_oe_new_key["row_id"] = range(1, len(data_oe_new_key) + 1)

# create the rolling averages
data_oe_rolling_values = pd.DataFrame()
for team in data_oe_new_key["teamname"].unique():
    data_oe_team = data_oe_new_key.loc[data_oe_new_key["teamname"] == team].drop(columns = ["teamname", "date"])
    data_to_stack_mean = data_oe_team.rolling(10, on = "row_id", min_periods = 3, closed = "left").mean()
    data_to_stack_std = data_oe_team.rolling(10, on = "row_id", min_periods = 3, closed = "left").std()
    data_to_stack = data_to_stack_mean.merge(data_to_stack_std, on = ["row_id"], how = "left")
    data_to_stack["team"] = team
    data_to_stack["dummy_key"] = 1
    data_keep_one = data_to_stack[data_to_stack.row_id == data_to_stack.row_id.max()].drop(columns = ["row_id"])
    data_oe_rolling_values = pd.concat([data_oe_rolling_values, data_keep_one], ignore_index=True, axis=0)

# drop missing from the rolling averages
data_oe_rolling_drop_missing = data_oe_rolling_values.dropna()
print(f"NOTE: Observations after dropping missing from rolling: {data_oe_rolling_drop_missing.shape[0]}")

# separate sides and merge on game
data_oe_red_side = data_oe_rolling_drop_missing[data_oe_rolling_drop_missing["team"] == team_red].drop(columns = ["team"])
data_oe_blue_side = data_oe_rolling_drop_missing[data_oe_rolling_drop_missing["team"] == team_blue].drop(columns = ["team"])
data_sides_merged = data_oe_blue_side.merge(data_oe_red_side, on = ["dummy_key"], how = "left").drop(columns = ["dummy_key"]).dropna()
print(f"NOTE: Observations after merging in the two sides and dropping missing: {data_sides_merged.shape[0]}")

# reorder data
data_sides_order = data_sides_merged[[
    "inhibitors_x_x",
    "opp_inhibitors_x_x",
    "earned gpm_x_x",
    "gspd_x_x",
    "xpdiffat10_x_x",
    "golddiffat10_x_x",
    "xpdiffat15_x_x",
    "csdiffat10_x_x",
    "firstbaron_x_x",
    "firsttothreetowers_x_x",
    "firstmidtower_x_x",
    "inhibitors_x_y",
    "opp_inhibitors_x_y",
    "earned gpm_x_y",
    "gspd_x_y",
    "xpdiffat10_x_y",
    "golddiffat10_x_y",
    "xpdiffat15_x_y",
    "csdiffat10_x_y",
    "firstbaron_x_y",
    "firsttothreetowers_x_y",
    "firstmidtower_x_y",
    "inhibitors_y_x",
    "opp_inhibitors_y_x",
    "earned gpm_y_x",
    "gspd_y_x",
    "xpdiffat10_y_x",
    "golddiffat10_y_x",
    "xpdiffat15_y_x",
    "csdiffat10_y_x",
    "firstbaron_y_x",
    "firsttothreetowers_y_x",
    "firstmidtower_y_x",
    "inhibitors_y_y",
    "opp_inhibitors_y_y",
    "earned gpm_y_y",
    "gspd_y_y",
    "xpdiffat10_y_y",
    "golddiffat10_y_y",
    "xpdiffat15_y_y",
    "csdiffat10_y_y",
    "firstbaron_y_y",
    "firsttothreetowers_y_y",
    "firstmidtower_y_y",
    "firsttower_x_x",
    "firsttower_y_x",
    "firsttower_x_y",
    "firsttower_y_y"
]]

# bring in the model
normalizer = joblib.load("std_scaler.bin")
model = keras.models.load_model("saved_model/league_oe_data")
prediction = model.predict(normalizer.transform(data_sides_order))
print(f"\nChance of {team_blue} winning against {team_red}: {int(prediction[0][0] * 100)}%")