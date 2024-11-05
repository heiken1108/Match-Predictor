import pandas as pd
import numpy as np


def get_last_n_matches(team, match_index, data, n=5):
    """
    Get the last N matches for a given team.
    """
    team_home_matches = data[
        (data["HomeTeam"] == team) & (data.index < match_index)
    ].tail(n)
    team_away_matches = data[
        (data["AwayTeam"] == team) & (data.index < match_index)
    ].tail(n)
    return pd.concat([team_home_matches, team_away_matches]).sort_index().tail(n)


matches_not_calc = 0


def calculate_team_stats(data, n=6):

    data = data.copy()

    data.loc[:, "Home Goals Last 5"] = None
    data.loc[:, "Away Goals Last 5"] = None
    data.loc[:, "Home Conceded Last 5"] = None
    data.loc[:, "Away Conceded Last 5"] = None
    data.loc[:, "Home Goal Difference Last 5"] = None
    data.loc[:, "Away Goal Difference Last 5"] = None
    data.loc[:, "Matchrating"] = None  # New feature
    skipped_matches = 0  # Counter for skipped matches
    for index, row in data.iterrows():
        # Home team stats
        home_team = row["HomeTeam"]
        last_home_matches = get_last_n_matches(home_team, index, data, n)

        # Away team stats
        away_team = row["AwayTeam"]
        last_away_matches = get_last_n_matches(away_team, index, data, n)

        # Ensure both teams have played at least 5 matches
        if len(last_home_matches) < 5 or len(last_away_matches) < 5:
            skipped_matches += 1
            continue  # Skip this match if either team hasn't played 5 matches

        home_goals_scored = 0
        home_goals_conceded = 0
        # Iterate over home team last matches
        for match in last_home_matches.itertuples():
            if (
                home_team == match.HomeTeam
            ):  # If home team is the home team in the match
                home_goals_scored += match.FTHG  # Goals scored at home
                home_goals_conceded += match.FTAG  # Goals conceded at home
            elif (
                home_team == match.AwayTeam
            ):  # If home team was the away team in the match
                home_goals_scored += match.FTAG  # Goals scored away
                home_goals_conceded += match.FTHG  # Goals conceded away

        # Similarly, calculate goals scored and conceded for the away team in their last 5 matches
        away_goals_scored = 0
        away_goals_conceded = 0
        for match in last_away_matches.itertuples():
            if away_team == match.HomeTeam:
                away_goals_scored += match.FTHG  # Goals scored at home
                away_goals_conceded += match.FTAG  # Goals conceded at home
            elif away_team == match.AwayTeam:
                away_goals_scored += match.FTAG  # Goals scored away
                away_goals_conceded += match.FTHG  # Goals conceded away

        # Update the columns with the calculated values
        data.loc[index, "Home Goals Last 5"] = home_goals_scored
        data.loc[index, "Away Goals Last 5"] = away_goals_scored
        data.loc[index, "Home Conceded Last 5"] = home_goals_conceded
        data.loc[index, "Away Conceded Last 5"] = away_goals_conceded
        data.loc[index, "Home Goal Difference Last 5"] = (
            home_goals_scored - home_goals_conceded
        )
        data.loc[index, "Away Goal Difference Last 5"] = (
            away_goals_scored - away_goals_conceded
        )

        # Calculate the Matchrating
        home_goal_diff = home_goals_scored - home_goals_conceded
        away_goal_diff = away_goals_scored - away_goals_conceded
        matchrating = home_goal_diff - away_goal_diff

        # Add Matchrating to the dataframe
        data.loc[index, "Matchrating"] = matchrating

    print(f"Skipped {skipped_matches} matches due to insufficient previous matches.")
    return data


def generate_matchrating_for_each_season_and_league(data):
    """
    Generate Matchrating for each season and league.
    """
    # Ensure the new columns are added to the main DataFrame
    data = data.copy()
    data["Home Goals Last 5"] = None
    data["Away Goals Last 5"] = None
    data["Home Conceded Last 5"] = None
    data["Away Conceded Last 5"] = None
    data["Home Goal Difference Last 5"] = None
    data["Away Goal Difference Last 5"] = None
    data["Matchrating"] = None

    # Iterate over each season and league
    for season in data["Season"].unique():
        for league in data["Div"].unique():
            # Filter data for the current season and league
            season_league_data = data[
                (data["Season"] == season) & (data["Div"] == league)
            ]

            # Calculate team stats for the season and league subset
            season_league_data = calculate_team_stats(season_league_data)

            # Update the main DataFrame with the calculated values
            data.update(season_league_data)

    return data


def calculate_outcome_percentages(data):
    # Ensure we do not modify the original DataFrame
    data = data.copy()

    # Step 1: Add a column for the match outcome (1 for Home Win, 0 for Draw, -1 for Away Win)
    data["Outcome"] = data.apply(
        lambda row: (
            1 if row["FTHG"] > row["FTAG"] else (-1 if row["FTHG"] < row["FTAG"] else 0)
        ),
        axis=1,
    )

    # Step 2: Calculate outcome percentages by Matchrating
    outcome_percentages = (
        data.groupby("Matchrating")["Outcome"]
        .value_counts(
            normalize=True
        )  # Get the proportion of each outcome per Matchrating
        .unstack(fill_value=0)  # Spread outcomes across columns and replace NaNs with 0
        * 100  # Convert proportions to percentages
    )

    # Step 3: Rename columns for clarity
    outcome_percentages = outcome_percentages.rename(
        columns={1: "Home Wins %", 0: "Draw %", -1: "Away Wins %"}
    )
    outcome_percentages = outcome_percentages[["Home Wins %", "Draw %", "Away Wins %"]]

    # Step 4: Calculate counts for each outcome by Matchrating
    outcome_counts = (
        data.groupby("Matchrating")["Outcome"]
        .value_counts()  # Get the count of each outcome per Matchrating
        .unstack(fill_value=0)  # Spread outcomes across columns and replace NaNs with 0
    )

    # Step 5: Rename columns for clarity in counts
    outcome_counts = outcome_counts.rename(
        columns={
            1: "Number of Home Wins",
            0: "Number of Draws",
            -1: "Number of Away Wins",
        }
    )

    # Step 6: Combine percentages and counts into a single DataFrame
    outcome_stats = outcome_percentages.join(outcome_counts)

    return outcome_stats
