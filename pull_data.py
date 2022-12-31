import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# get file root
user_profile = os.environ["USERPROFILE"]
file_root = f"{user_profile}\\OneDrive\\Documents\\GitHub\\league_predictions"

# read the data   
data_oe = pd.read_csv(f"{file_root}\\intake\\2022_match_data.csv")

# narrow the data, choose variables that are team defining, not luck, and moderated by a time metric
data_oe_narrow = data_oe[[
    "gameid",
    "teamname",
    "league",
    "side",
    "date",
    "position",
    "firstblood",
    "team kpm",
    "ckpm",
    "firstdragon",
    "elementaldrakes",
    "opp_elementaldrakes",
    "elders",
    "opp_elders",
    "firstherald",
    "heralds",
    "firstbaron",
    "firsttower",
    "towers",
    "opp_towers",
    "firstmidtower",
    "firsttothreetowers",
    "turretplates",
    "opp_turretplates",
    "inhibitors",
    "opp_inhibitors",
    "dpm",
    "damagetakenperminute",
    "wpm",
    "wcpm",
    "vspm",
    "earned gpm",
    "gspd",
    "cspm",
    "golddiffat10",
    "xpdiffat10",
    "csdiffat10",
    "killsat10",
    "assistsat10",
    "deathsat10",
    "opp_assistsat10",
    "golddiffat15",
    "xpdiffat15",
    "csdiffat15",
    "killsat15",
    "assistsat15",
    "deathsat15",
    "opp_assistsat15",
    "result"
]]
print(f"NOTE: Observations in raw data: {data_oe_narrow.shape[0]}")

# only keep team observations
data_oe_narrow_row = data_oe_narrow[(data_oe_narrow["position"] == "team") & (data_oe_narrow["league"].isin(["LCS", "LCK", "LEC"]))]
print(f"NOTE: Observations after keeping team statistics: {data_oe_narrow_row.shape[0]}")

# drop position, drop missing values
data_oe_drop_column = data_oe_narrow_row.drop(columns = ["position", "league"])
data_oe_drop_missing = data_oe_drop_column.dropna()
print(f"NOTE: Observations after dropping missing values: {data_oe_drop_missing.shape[0]}")

# create the sort and assign an index
data_oe_new_key = data_oe_drop_missing.sort_values(by = ["teamname", "date"])
data_oe_new_key["row_id"] = range(1, len(data_oe_new_key) + 1)

# create the rolling averages
data_oe_rolling_merge_key = data_oe_new_key[["row_id", "result", "gameid", "side"]]

data_oe_rolling_values = pd.DataFrame()
for team in data_oe_new_key["teamname"].unique():
    data_oe_team = data_oe_new_key.loc[data_oe_new_key["teamname"] == team].drop(columns = ["result", "teamname", "date", "gameid", "side"])
    data_to_stack_mean = data_oe_team.rolling(10, on = "row_id", min_periods = 3, closed = "left").mean()
    data_to_stack_std = data_oe_team.rolling(10, on = "row_id", min_periods = 3, closed = "left").std()
    data_to_stack = data_to_stack_mean.merge(data_to_stack_std, on = ["row_id"], how = "left")
    data_oe_rolling_values = pd.concat([data_oe_rolling_values, data_to_stack], ignore_index=True, axis=0)

# drop missing from the rolling averages
data_oe_rolling_drop_missing = data_oe_rolling_values.dropna()
print(f"NOTE: Observations after dropping missing from rolling: {data_oe_rolling_drop_missing.shape[0]}")

# merge in the results
data_oe_rolling_merge = data_oe_rolling_drop_missing.merge(data_oe_rolling_merge_key, on = ["row_id"], how = "left").drop(columns = ["row_id"])
print(f"NOTE: Observations after merging in results to rolling: {data_oe_rolling_merge.shape[0]}")
print(f"NOTE: Check for any missing in results to rolling: {data_oe_rolling_merge.isna().sum().sum()}")

# separate sides and merge on game
data_oe_red_side = data_oe_rolling_merge[data_oe_rolling_merge["side"] == "Red"].drop(columns = ["side", "result"])
data_oe_blue_side = data_oe_rolling_merge[data_oe_rolling_merge["side"] == "Blue"].drop(columns = ["side"])
data_sides_merged = data_oe_blue_side.merge(data_oe_red_side, on = ["gameid"], how = "left").drop(columns = ["gameid"]).dropna()
print(f"NOTE: Observations after merging in the two sides and dropping missing: {data_sides_merged.shape[0]}")

"""
# check correlation
plt.figure(figsize = (20, 15))
mask = np.triu(np.ones_like(data_oe_rolling_merge.corr(), dtype=bool))
sns.heatmap(
    data_oe_rolling_merge.corr(), 
    annot = False, 
    mask = mask, 
    vmin = -1, 
    vmax = 1,
    # fmt = ".1f"
)
plt.title('Correlation Coefficient Of Game Stats')
plt.show()
"""
"""
the correlations did not show to high for the volatility (STD) metrics. assumption is that
if the correlated mean values are kicked out the volatility metrics should be ok.
"""

# keep only the metrics somewhat correlated with result
data_oe_correlated = data_oe_rolling_merge[[
    "csdiffat10_x",
    "csdiffat10_y",
    "csdiffat15_x",
    "csdiffat15_y",
    "earned gpm_x",
    "earned gpm_y",
    "firstbaron_x",
    "firstbaron_y",
    "firstmidtower_x",
    "firstmidtower_y",
    "firsttothreetowers_x",
    "firsttothreetowers_y",
    "firsttower_x",
    "firsttower_y",
    "golddiffat10_x",
    "golddiffat10_y",
    "golddiffat15_x",
    "golddiffat15_y",
    "gspd_x",
    "gspd_y",
    "inhibitors_x",
    "inhibitors_y",
    "opp_inhibitors_x",
    "opp_inhibitors_y",
    "opp_towers_x",
    "opp_towers_y",
    "towers_x",
    "towers_y",
    "xpdiffat10_x",
    "xpdiffat10_y",
    "xpdiffat15_x",
    "xpdiffat15_y",
    "result"
]]
# Calculate the VIF
def compute_vif(considered_features):
    """ Compute VIF
    Take the high correlated game stats and slowly remove 
    variables until VIF is under the threshold < 5
    """
    data_oe_considered = data_oe_correlated[considered_features]
    data_oe_considered = data_oe_considered.assign(intercept = 1)
    
    vif = pd.DataFrame()
    vif["Variable"] = data_oe_considered.columns
    vif["VIF"] = [variance_inflation_factor(data_oe_considered.values, i) for i in range(data_oe_considered.shape[1])]
    vif = vif[vif["Variable"] != "intercept"]
    return vif

# Send in the variables for VIF
print("\nNOTE: VIF of all correlated variables:")
considered_features = [
    "towers_x",
    "opp_towers_x",
    "inhibitors_x",
    "opp_inhibitors_x",
    "earned gpm_x",
    "gspd_x",
    "xpdiffat10_x",
    "golddiffat10_x",
    "golddiffat15_x",
    "xpdiffat15_x",
    "csdiffat15_x",
    "csdiffat10_x",
    "firstbaron_x",
    "firsttothreetowers_x",
    "firstmidtower_x"
]
print(compute_vif(considered_features).sort_values("VIF", ascending = False))

# Send in the variables for VIF
print("\nNOTE: VIF, drop csdiffat15_x, golddiffat15_x, opp_towers_x, towers_x:")
considered_features = [
    "inhibitors_x",
    "opp_inhibitors_x",
    "earned gpm_x",
    "gspd_x",
    "xpdiffat10_x",
    "golddiffat10_x",
    "xpdiffat15_x",
    "csdiffat10_x",
    "firstbaron_x",
    "firsttothreetowers_x",
    "firstmidtower_x"
]
print(compute_vif(considered_features).sort_values("VIF", ascending = False))

# drop the columns that are too highly correlated
data_oe_keep_corr = data_sides_merged[[
    "result",
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

# save the data to csv
data_oe_keep_corr.to_csv(f"{file_root}\\working\\data_oe_training.csv", index = False)
print(f"\nNOTE: Final number of columns: {data_oe_keep_corr.shape[1]}")