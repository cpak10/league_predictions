import pandas as pd
import os

# input
league = "LCK"

# get file root
user_profile = os.environ["USERPROFILE"]
file_root = f"{user_profile}\\OneDrive\\Documents\\GitHub\\league_predictions"

# read data
data_oe = pd.read_csv(f"{file_root}\\intake\\2022_match_data.csv")

# narrow data
data_oe_narrow = data_oe[["league", "teamname"]]

# find the unique teams in league
data_oe_league = data_oe_narrow[data_oe_narrow["league"] == league]
list_teams = data_oe_league["teamname"].unique()
print(f"\nTeams in {league}: {list_teams}")