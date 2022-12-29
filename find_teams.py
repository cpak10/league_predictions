import pandas as pd

# input
league = "LCK"

# read data
data_oe = pd.read_csv("2022_match_data.csv")

# narrow data
data_oe_narrow = data_oe[["league", "teamname"]]

# find the unique teams in league
data_oe_league = data_oe_narrow[data_oe_narrow["league"] == league]
list_teams = data_oe_league["teamname"].unique()
print(f"\nTeams in {league}: {list_teams}")