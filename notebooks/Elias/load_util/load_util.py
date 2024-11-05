import os
import pandas as pd


def fetch_data_into_file(data_folder, file_name, start_year, end_year, leagues) -> None:
    url_template = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    cols = [
        "Div",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HTHG",
        "HTAG",
        "HTR",
        "Referee",
        "HS",
        "AS",
        "HST",
        "AST",
        "HF",
        "AF",
        "HC",
        "AC",
        "HY",
        "AY",
        "HR",
        "AR",
        "HBP",
    ]

    # Generate seasons list
    seasons = []
    for year in range(start_year, end_year):
        start = str(year)[-2:]
        end = str(year + 1)[-2:]
        seasons.append(start + end)

    df_tmp = []
    for season in seasons:
        for league in leagues:
            try:
                try:
                    df = pd.read_csv(url_template.format(season=season, league=league))
                except:
                    df = pd.read_csv(
                        url_template.format(season=season, league=league),
                        encoding="latin",
                    )
                try:
                    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y")
                except ValueError:
                    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
            except:
                print(
                    "No data for",
                    season,
                    league,
                    url_template.format(season=season, league=league),
                )
                continue

            existing_cols = [col for col in cols if col in df.columns]
            df = df[existing_cols]
            df["Season"] = str(season).zfill(4)
            df_tmp.append(df)
    df = pd.concat(df_tmp)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    file_path = os.path.join(data_folder, file_name + ".csv")
    df.to_csv(file_path, index=False)
    print("Data fetched and saved to", file_path)


def load_data(data_folder, file_name) -> pd.DataFrame:
    file_path = os.path.join(data_folder, file_name + ".csv")
    df = pd.read_csv(file_path)
    return df


def get_all_teams(data) -> list:
    home_teams = data["HomeTeam"].unique()
    away_teams = data["AwayTeam"].unique()
    return list(set(home_teams) | set(away_teams))


def get_all_matches_for_team(data, team_name) -> pd.DataFrame:
    matches = data[(data["HomeTeam"] == team_name) | (data["AwayTeam"] == team_name)]
    return matches


def get_elo_ratings_for_team(data, team_name) -> pd.DataFrame:
    new_rows = []
    matches = get_all_matches_for_team(data, team_name)
    for index, row in matches.iterrows():
        if row["HomeTeam"] == team_name:
            Elo = row["Home ELO"]
            Elo_change = row["Home ELO change"]
        elif row["AwayTeam"] == team_name:
            Elo = row["Away ELO"]
            Elo_change = row["Away ELO change"]
        new_row = {"Date": row["Date"], "ELO": Elo, "ELO change": Elo_change}
        new_rows.append(new_row)
    return pd.DataFrame(new_rows)


def split_matches_into_seasons(
    data, matches_per_season=380, start_year=2005, end_year=2024
):

    seasons = [f"{year}/{year+1}" for year in range(start_year, end_year)]

    # Initialize an empty list to store the DataFrames
    season_dfs = []

    # Split data into seasons (assumes data is sorted by date)
    for i, season in enumerate(seasons):
        start_idx = i * matches_per_season
        end_idx = start_idx + matches_per_season

        # Extract each season's data
        season_data = data.iloc[start_idx:end_idx].copy()

        # Add a 'Season' column to identify the season
        season_data["Season"] = season

        # Append the DataFrame for this season to the list
        season_dfs.append(season_data)

    # Concatenate all season DataFrames into a single DataFrame
    combined_data = pd.concat(season_dfs, ignore_index=True)

    return combined_data


def get_cleaned_data(data):
    data = data.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data.reset_index(drop=True)
    return data
