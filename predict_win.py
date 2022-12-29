import pandas as pd
from pandas.tseries.offsets import *
from datetime import datetime
from tensorflow import keras
import joblib
import seaborn
import matplotlib.pyplot as plt

# inputs
team_blue = "Evil Geniuses"
team_red = "Team Liquid"
date = "20221129"
new_data = 0

# update the date before running
if new_data == 0:
    data_oe = pd.read_csv(f"2022_match_data.csv")

# saving new data
if new_data == 1:
    link_oe_data = f"https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com/2022_LoL_esports_match_data_from_OraclesElixir_{date}.csv"
    data_oe = pd.read_csv(link_oe_data)
    data_oe.to_csv(f"oe_extract_{date}.csv", index = False)

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
data_oe_drop_missing["date"] = pd.to_datetime(data_oe_drop_missing["date"], format = "%Y-%m-%d %X")
data_oe_new_key = data_oe_drop_missing.sort_values(by = ["teamname", "date"])
data_oe_new_key["row_id"] = range(1, len(data_oe_new_key) + 1)
data_row_date_finder = data_oe_new_key[["row_id", "date"]]

# create the rolling averages
data_oe_rolling_values = pd.DataFrame()
for team in data_oe_new_key["teamname"].unique():
    data_oe_team = data_oe_new_key.loc[data_oe_new_key["teamname"] == team].drop(columns = ["teamname", "date"])
    data_to_stack_mean = data_oe_team.rolling(10, on = "row_id", min_periods = 3, closed = "left").mean()
    data_to_stack_std = data_oe_team.rolling(10, on = "row_id", min_periods = 3, closed = "left").std()
    data_to_stack = data_to_stack_mean.merge(data_to_stack_std, on = "row_id", how = "left")
    data_to_stack_date = data_to_stack.merge(data_row_date_finder, on = "row_id", how = "left")
    data_to_stack_date["team"] = team
    data_to_stack_date["week_end"] = data_to_stack_date["date"] + Week(weekday = 4)
    data_to_stack_date["week_end"] = data_to_stack_date["week_end"].dt.date
    data_by_date = data_to_stack_date.groupby(["week_end"]).nth(-1).reset_index()
    data_oe_rolling_values = pd.concat([data_oe_rolling_values, data_by_date], ignore_index=True, axis=0)

# drop missing from the rolling averages
data_oe_rolling_drop_missing = data_oe_rolling_values.dropna()
print(f"NOTE: Observations after dropping missing from rolling: {data_oe_rolling_drop_missing.shape[0]}")

# separate sides and merge on game
data_oe_red_side = data_oe_rolling_drop_missing[data_oe_rolling_drop_missing["team"] == team_red].drop(columns = ["team"])
data_oe_blue_side = data_oe_rolling_drop_missing[data_oe_rolling_drop_missing["team"] == team_blue].drop(columns = ["team"])
data_sides_merged = data_oe_blue_side.merge(data_oe_red_side, on = "week_end", how = "left").dropna()
data_date_list = data_sides_merged["week_end"].unique()
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

# bring in the model for sequential
normalizer = joblib.load("std_scaler.bin")
model = keras.models.load_model("saved_model/league_oe_data")
prediction = model.predict(normalizer.transform(data_sides_order))
dict_prediction = {}
for i in range(len(data_date_list)):
    dict_prediction[i] = [data_date_list[i].strftime("%m-%d"), int(prediction[i][0] * 100), team_blue]
    dict_prediction[i +.5] = [data_date_list[i].strftime("%m-%d"), 100 - int(prediction[i][0] * 100), team_red]
data_date_prediction = pd.DataFrame.from_dict(dict_prediction, orient = "index", columns = ["date", "prediction", "Team"])

# bring in the model for random forest
normalizer_rf = joblib.load("std_scaler_rf.bin")
model_rf = joblib.load("model_random_forest.bin")
prediction_rf = model_rf.predict_proba(normalizer_rf.transform(data_sides_order))
dict_prediction_rf = {}
for i in range(len(data_date_list)):
    dict_prediction_rf[i] = [data_date_list[i].strftime("%m-%d"), int(prediction_rf[i][1] * 100), team_blue]
    dict_prediction_rf[i +.5] = [data_date_list[i].strftime("%m-%d"), int(prediction_rf[i][0] * 100), team_red]
data_date_prediction_rf = pd.DataFrame.from_dict(dict_prediction_rf, orient = "index", columns = ["date", "prediction", "Team"])

# plot data for sequential
plt.figure(1)
graph = seaborn.lineplot(data = data_date_prediction, x = "date", y = "prediction", hue = "Team", palette = "colorblind")
graph.axhline(70, linestyle = "--", color = "r", alpha = .5)
graph.axhline(30, linestyle = "--", color = "r", alpha = .5)
plt.title(f"Win Predictions For {team_blue} v. {team_red} (Seq)")
plt.xticks(rotation = 90)
plt.xlabel("End of Week Date")
plt.ylabel("% Chance of Win")

# plot data for random forest
plt.figure(2)
graph_rf = seaborn.lineplot(data = data_date_prediction_rf, x = "date", y = "prediction", hue = "Team", palette = "colorblind")
graph_rf.axhline(70, linestyle = "--", color = "r", alpha = .5)
graph_rf.axhline(30, linestyle = "--", color = "r", alpha = .5)
plt.title(f"Win Predictions For {team_blue} v. {team_red} (RF)")
plt.xticks(rotation = 90)
plt.xlabel("End of Week Date")
plt.ylabel("% Chance of Win")

# show plots
plt.show()