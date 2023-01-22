import pandas as pd

# input
league = input("INPUT - Enter league here (e.g. LCK, LCS, LPL): ")

# get file root
file_root = "C:\\GitHub\\league_predictions"

# read data
data_oe = pd.read_csv(f"{file_root}\\intake\\2022_match_data.csv")

# narrow data
data_oe_narrow = data_oe[["league", "teamname"]]

# find the unique teams in league
data_oe_league = data_oe_narrow[data_oe_narrow["league"] == league]
list_teams = data_oe_league["teamname"].unique()
print(f"\nTeams in {league}: {list_teams}")