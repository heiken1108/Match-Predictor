import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

anomaly_color = "sandybrown"
prediction_color = "yellowgreen"
training_color = "yellowgreen"
validation_color = "gold"
test_color = "coral"
figsize = (9, 3)

# Loading


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
    df = pd.read_csv(file_path, parse_dates=["Date"], dtype={"Season": str})
    return df


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = data.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    if "Referee" in data.columns:
        data.drop(columns="Referee", inplace=True)  # Fjerner kolonnen Referee
    data.dropna(inplace=True)  # Fjerner rader med manglende verdier
    data = data.reset_index(drop=True)
    return data


# Plotting


def plot_histogram(
    y_values,
    figsize=(9, 3),
    title="Distribution of Games According to Matchrating",
    xlabel="Matchrating",
    ylabel="Number of Matches",
):
    plt.close("all")
    # Calculate bins based on the data range
    bins = np.arange(y_values.min() - 0.5, y_values.max() + 1.5, 1)

    # Create the histogram plot
    ax = y_values.hist(bins=bins, figsize=figsize, edgecolor="black", rwidth=0.9)

    # Set the title and labels
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Add grid lines for better readability
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

    # Show the plot
    plt.tight_layout()  # Adjust layout to make room for labels
    plt.show()


def plot_sub_plots(data, cols, figsize=figsize):
    plt.close("all")
    _, axes = plt.subplots(nrows=2, ncols=int(np.ceil(len(cols) // 2)), figsize=figsize)
    for ax, cname in zip(axes.ravel(), cols):
        data.hist(cname, ax=ax)
    plt.tight_layout()


def plot_series(
    data,
    labels=None,
    windows=None,
    predictions=None,
    highlights=None,
    val_start=None,
    test_start=None,
    threshold=None,
    figsize=figsize,
    xlabel=None,
    ylabel=None,
):
    # Open a new figure
    plt.close("all")
    plt.figure(figsize=figsize)

    x = data.index
    # Plot data
    plt.plot(x, data.values, zorder=0)
    # Rotated x ticks
    plt.xticks(rotation=45)
    # Plot labels
    if labels is not None:
        plt.scatter(labels.values, data.loc[labels], color=anomaly_color, zorder=2)
    # Plot windows
    if windows is not None:
        for _, wdw in windows.iterrows():
            plt.axvspan(
                wdw["begin"], wdw["end"], color=anomaly_color, alpha=0.3, zorder=1
            )

    # Plot training data
    if val_start is not None:
        plt.axvspan(x[0], val_start, color=training_color, alpha=0.1, zorder=-1)
    if val_start is None and test_start is not None:
        plt.axvspan(x[0], test_start, color=training_color, alpha=0.1, zorder=-1)
    if val_start is not None:
        plt.axvspan(val_start, test_start, color=validation_color, alpha=0.1, zorder=-1)
    if test_start is not None:
        plt.axvspan(test_start, x[-1], color=test_color, alpha=0.3, zorder=0)
    # Predictions
    if predictions is not None:
        plt.scatter(
            predictions.values,
            data.loc[predictions],
            color=prediction_color,
            alpha=0.4,
            zorder=3,
        )
    # Plot threshold
    if threshold is not None:
        plt.plot([x[0], x[-1]], [threshold, threshold], linestyle=":", color="tab:red")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(":")
    plt.tight_layout()


def plot_grouped_bars(data, figsize=figsize, title=None, xlabel="X", ylabel="Y"):
    plt.close("all")
    data = data.copy()
    bar_positions = np.arange(len(data.index)) * 1.2
    home_wins = data["Number of Home Wins"]
    draws = data["Number of Draws"]
    away_wins = data["Number of Away Wins"]

    bar_width = 0.8  # Width less than 1 creates space between bars
    plt.bar(
        bar_positions,
        home_wins,
        width=bar_width,
        label="Share home win",
        color="#4CAF50",
        edgecolor="black",
        linewidth=1,
    )
    plt.bar(
        bar_positions,
        draws,
        width=bar_width,
        bottom=home_wins,
        label="Share draw",
        color="#FFA726",
        edgecolor="black",
        linewidth=1,
    )
    plt.bar(
        bar_positions,
        away_wins,
        width=bar_width,
        bottom=home_wins + draws,
        label="Share away win",
        color="#EF5350",
        edgecolor="black",
        linewidth=1,
    )

    # Customize the plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Set x-ticks to show actual goal difference values
    plt.xticks(bar_positions, data.index, rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.show()


# Modelling


class ELO:  # Kan gjøre slik at home_advantage lages slik at den helles gir home_factor for kamper der hjemme og borte har samme rating
    def __init__(
        self, data, init_rating=1500, draw_factor=0.25, k_factor=32, home_advantage=100
    ):
        self.data = data
        self.init_rating = init_rating
        self.draw_factor = draw_factor
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {}
        self.add_teams(data)

    def add_teams(self, data: pd.DataFrame):
        home_teams = data["HomeTeam"].unique()
        away_teams = data["AwayTeam"].unique()
        teams = list(set(home_teams) | set(away_teams))
        for team in teams:

            r = data[data["HomeTeam"] == team].iloc[0]

            if r["Div"] == "E0":
                self.ratings[team] = self.init_rating
            elif r["Div"] == "E1":
                self.ratings[team] = self.init_rating - 200
            elif r["Div"] == "E2":
                self.ratings[team] = self.init_rating - 400
            elif r["Div"] == "E3":
                self.ratings[team] = self.init_rating - 600
            else:
                self.ratings[team] = self.init_rating - 800

    def calculate_new_rating(self, home_elo, away_elo, result):
        if result == "H":
            s_home, s_away = 1, 0
        elif result == "D":
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0, 1
        e_home, e_d, e_away = self.expect_result(
            home_elo + self.home_advantage, away_elo
        )

        new_rating_home = home_elo + self.k_factor * (s_home - (e_home + e_d / 2))
        new_rating_away = away_elo + self.k_factor * (s_away - (e_away + e_d / 2))
        return new_rating_home, new_rating_away

    def expect_result(self, home_elo, away_elo):
        elo_diff = home_elo - away_elo
        excepted_home_without_draws = 1 / (1 + 10 ** (-elo_diff / 400))
        expected_away_without_draws = 1 / (1 + 10 ** (elo_diff / 400))
        real_expected_draw = self.draw_factor * (
            1 - abs(excepted_home_without_draws - expected_away_without_draws)
        )
        real_expected_home = excepted_home_without_draws - real_expected_draw / 2
        real_expected_away = expected_away_without_draws - real_expected_draw / 2
        return real_expected_home, real_expected_draw, real_expected_away

    def perform_matchup(self, home_team, away_team, result) -> None:
        try:
            old_rating_home = self.ratings[home_team]
            old_rating_away = self.ratings[away_team]
            new_rating_home, new_rating_away = self.calculate_new_rating(
                old_rating_home, old_rating_away, result
            )
            self.ratings[home_team] = new_rating_home
            self.ratings[away_team] = new_rating_away
            return old_rating_home, old_rating_away
        except KeyError:
            print("One or both teams does not exist")
            return None

    def perform_simulations(self, data) -> pd.DataFrame:
        data["Home ELO"] = None
        data["Away ELO"] = None
        data["ELO diff"] = None
        for index, row in data.iterrows():
            old_rating_home, old_rating_away = self.perform_matchup(
                row["HomeTeam"], row["AwayTeam"], row["FTR"]
            )
            data.at[index, "Home ELO"] = old_rating_home
            data.at[index, "Away ELO"] = old_rating_away
            data.at[index, "ELO diff"] = old_rating_home - old_rating_away
        for column in ["Home ELO", "Away ELO", "ELO diff"]:
            data[column] = pd.to_numeric(data[column])
        return data

    def get_probabilities(self, data) -> pd.DataFrame:
        data["Home_prob_ELO"] = None
        data["Draw_prob_ELO"] = None
        data["Away_prob_ELO"] = None
        for index, row in data.iterrows():
            home_prob, draw_prob, away_prob = self.expect_result(
                row["Home ELO"] + self.home_advantage, row["Away ELO"]
            )
            data.at[index, "Home_prob_ELO"] = home_prob
            data.at[index, "Draw_prob_ELO"] = draw_prob
            data.at[index, "Away_prob_ELO"] = away_prob
        for column in ["Home_prob_ELO", "Draw_prob_ELO", "Away_prob_ELO"]:
            data[column] = pd.to_numeric(data[column])
        return data


def extract_elo_history(data, team) -> pd.DataFrame:
    elo_history = []
    for index, row in data.iterrows():
        if row["HomeTeam"] == team:
            elo_history.append(
                {
                    "Date": row["Date"],
                    "Opponent": row["AwayTeam"],
                    "ELO": row["Home ELO"],
                    "Result": row["FTR"],
                }
            )
        elif row["AwayTeam"] == team:
            elo_history.append(
                {
                    "Date": row["Date"],
                    "Opponent": row["HomeTeam"],
                    "ELO": row["Away ELO"],
                    "Result": row["FTR"],
                }
            )
    return pd.DataFrame(elo_history)


def pick_highest_probabilites(data: pd.DataFrame):
    wrong = 0
    correct = 1
    for index, row in data.iterrows():
        max_prob = max(row["Home Prob"], row["Draw Prob"], row["Away Prob"])
        if (
            (row["Home Prob"] == max_prob and row["FTR_H"])
            or (row["Draw Prob"] == max_prob and row["FTR_D"])
            or (row["Away Prob"] == max_prob and row["FTR_A"])
        ):
            correct += 1
        else:
            wrong += 1
    return correct, wrong


def get_all_matches_of_team(data, team):
    c = data.copy()
    return c[(c["HomeTeam"] == team) | (c["AwayTeam"] == team)]


def add_form_column(
    data: pd.DataFrame,
    home_column,
    away_column,
    n=5,
    operation="Sum",
    regard_opponent=False,
    include_current=False,
):
    """
    Function that performs the operation on the n last matches for each team.
    If regard_opponent is True, the operation is performed on the opponents column instead.
    Example: Home_column = FTHG, Away_column=FTAG, Operation = Sum, n = 5, regard_opponent = False creates columns to describe how many goals the team has scored in the last 5 matches.
    Example: Home_column = FTHG, Away_column=FTAG, Operation = Sum, n = 5, regard_opponent = True creates columns to describe how many of goals the team has conceded in the last 5 matches.
    Args:
            data (pd.DataFrame): The dataframe to add the columns to.
            home_column (str): The column to use if the team is at home.
            away_column (str): The column to use if the team is away.
            n (int): The number of matches to consider.
            operation (str): The operation to perform. Can be 'Sum', 'Mean' or 'Change'.
            regard_opponent (bool): If True, the operation is performed on the opponents column instead. E.g. Can be used to get mean of opponent ELO
            include_current (bool): If True, the current match is included in the operation. Used if column is already dependent on previous matches, such as Home ELO and Away ELO.
    """
    new_column_name_home = (
        home_column
        + "_"
        + operation
        + "_"
        + str(n)
        + ("_opponent" if regard_opponent else "")
    )
    new_column_name_away = (
        away_column
        + "_"
        + operation
        + "_"
        + str(n)
        + ("_opponent" if regard_opponent else "")
    )
    data[new_column_name_home] = None
    data[new_column_name_away] = None
    teams = data["HomeTeam"].unique()
    for team in teams:
        matches = get_all_matches_of_team(data, team)
        scores = {}
        pos = 0 if not include_current else 1
        for index, row in matches.iterrows():
            start_pos = max(0, pos - n)
            relevant_matches = matches.iloc[start_pos:pos]
            s = 0
            if operation == "Sum":
                for index_r, row_r in relevant_matches.iterrows():
                    if row_r["HomeTeam"] == team:
                        if regard_opponent:
                            s += row_r[away_column]
                        else:
                            s += row_r[home_column]
                    else:
                        if regard_opponent:
                            s += row_r[home_column]
                        else:
                            s += row_r[away_column]
            elif operation == "Mean":
                for index_r, row_r in relevant_matches.iterrows():
                    if row_r["HomeTeam"] == team:
                        if regard_opponent:
                            s += row_r[away_column]
                        else:
                            s += row_r[home_column]
                    else:
                        if regard_opponent:
                            s += row_r[home_column]
                        else:
                            s += row_r[away_column]
                if len(relevant_matches) == 0:
                    s = 0
                else:
                    s = s / len(relevant_matches)
            elif operation == "Change":
                if len(relevant_matches) == 0:
                    s = 0
                else:
                    first_row = relevant_matches.iloc[0]
                    last_row = relevant_matches.iloc[-1]
                    first_score = (
                        first_row[home_column]
                        if first_row["HomeTeam"] == team
                        else first_row[away_column]
                    )
                    last_score = (
                        last_row[home_column]
                        if last_row["HomeTeam"] == team
                        else last_row[away_column]
                    )
                    s = last_score - first_score
            elif operation == "Points":
                for index_r, row_r in relevant_matches.iterrows():
                    if row_r["HomeTeam"] == team:
                        if row_r["FTHG"] > row_r["FTAG"]:
                            s += 3
                        elif row_r["FTHG"] == row_r["FTAG"]:
                            s += 1
                        else:
                            s += 0
                    else:
                        if row_r["FTAG"] > row_r["FTHG"]:
                            s += 3
                        elif row_r["FTAG"] == row_r["FTHG"]:
                            s += 1
                        else:
                            s += 0
            scores[index] = s
            pos += 1

        for key, value in scores.items():
            if data.at[key, "HomeTeam"] == team:
                data.at[key, new_column_name_home] = value
            else:
                data.at[key, new_column_name_away] = value
    data[new_column_name_home] = pd.to_numeric(
        data[new_column_name_home], errors="coerce"
    )
    data[new_column_name_away] = pd.to_numeric(
        data[new_column_name_away], errors="coerce"
    )
    return data


def remove_the_first_n_matches_in_a_season_for_each_team(data, n=5):
    # Create a copy of the DataFrame to avoid modifying the original
    datacopy = data.copy()

    # Initialize a set to store indices of rows to remove (using a set to avoid duplicates)
    rows_to_remove = set()

    # Dictionary to store the count of removed rows for each season
    removed_rows_count = {}

    # Iterate over each season
    for season in datacopy["Season"].unique():
        # Filter data for the current season
        season_data = datacopy[datacopy["Season"] == season]

        for team in season_data["HomeTeam"].unique():
            # Get the first N matches for the current team
            team_matches = season_data[
                (season_data["HomeTeam"] == team) | (season_data["AwayTeam"] == team)
            ].head(n)
            removing_indices = team_matches.index
            # print("Removing indexes:", removing_indices)
            # Add the indices of the first N matches to the set
            for index in removing_indices:
                rows_to_remove.add(index)

    # Remove all identified rows in one step
    print("number of matches removed: ", len(rows_to_remove))

    datacopy = datacopy.drop(index=rows_to_remove).reset_index(drop=True)

    return datacopy


def calculate_outcome_percentages(data):
    data = data.copy()

    data["Outcome"] = data.apply(
        lambda row: (
            1 if row["FTHG"] > row["FTAG"] else (-1 if row["FTHG"] < row["FTAG"] else 0)
        ),
        axis=1,
    )

    outcome_percentages = (
        data.groupby("Match Rating")["Outcome"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    )

    outcome_percentages = outcome_percentages.rename(
        columns={1: "Home Wins %", 0: "Draw %", -1: "Away Wins %"}
    )
    outcome_percentages = outcome_percentages[["Home Wins %", "Draw %", "Away Wins %"]]

    outcome_counts = (
        data.groupby("Match Rating")["Outcome"].value_counts().unstack(fill_value=0)
    )

    outcome_counts = outcome_counts.rename(
        columns={
            1: "Number of Home Wins",
            0: "Number of Draws",
            -1: "Number of Away Wins",
        }
    )

    outcome_stats = outcome_percentages.join(outcome_counts)

    return outcome_stats


def plot_auc_per_class(auc_per_class, class_names, auc_macro):
    # Append the macro-average AUC for plotting
    auc_values = auc_per_class.tolist() + [auc_macro]
    class_names_with_macro = class_names + ["Macro-Average"]

    # Create the bar plot
    plt.figure(figsize=(8, 5))
    plt.bar(class_names_with_macro, auc_values, color="skyblue")
    plt.ylim(0, 1)  # AUC values range from 0 to 1
    plt.xlabel("Class")
    plt.ylabel("AUC")
    plt.title("AUC for Each Class and Macro-Average AUC")

    # Add value labels on top of each bar
    for i, auc in enumerate(auc_values):
        plt.text(i, auc + 0.02, f"{auc:.2f}", ha="center", va="bottom")

    plt.show()


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_multi_class_roc(y_test, y_pred_proba, classes, class_names=None):
    """
    Plots ROC curves for multi-class classification.

    Parameters:
    - y_test: Array-like, true labels
    - y_pred_proba: Array-like, probability predictions for each class
    - classes: List of unique class labels (e.g., [-1, 0, 1])
    - class_names: List of class names for display in the legend (default is None,
                   which will use string representations of classes)
    """
    # Binarize the labels for multi-class ROC AUC calculation
    y_test_binarized = label_binarize(y_test, classes=classes)
    n_classes = y_test_binarized.shape[1]

    # Initialize dictionaries to hold FPR, TPR, and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate FPR, TPR, and AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Set default class names if none provided
    if class_names is None:
        class_names = [str(cls) for cls in classes]

    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    colors = ["blue", "red", "green"]  # Customize or expand colors as needed

    for i, color in enumerate(colors[:n_classes]):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve for {class_names[i]} (AUC = {roc_auc[i]:.2f})",
        )

    # Plot the diagonal line representing a random classifier
    plt.plot([0, 1], [0, 1], "k--", lw=2)

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Multi-Class Classification")
    plt.legend(loc="lower right")

    # Show the plot
    plt.show()
