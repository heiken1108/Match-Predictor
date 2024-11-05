import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

anomaly_color = 'sandybrown'
prediction_color = 'yellowgreen'
training_color = 'yellowgreen'
validation_color = 'gold'
test_color = 'coral'
figsize=(9, 3)

def plot_histogram(
    y_values,
    figsize=(9, 3),
    title="Distribution of Games According to Matchrating",
    xlabel="Matchrating",
    ylabel="Number of Matches",
):
	plt.close('all')
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

def plot_discrete_histogram(data: pd.DataFrame, vmin=None, vmax=None, figsize=figsize):
	plt.close('all')
	plt.figure(figsize=figsize)
	bins = np.arange(data.values.min() - 0.5, data.values.max() + 1.5, 1)
	plt.hist(data.values, bins=bins, edgecolor='black', rwidth=0.9)
	lims = plt.xlim()
	if vmin is not None:
		lims = (vmin, lims[1])
	if vmax is not None:
		lims = (lims[0], vmax)
	plt.xlim(lims)
	plt.grid()
	plt.tight_layout()

def plot_histogram2d(xdata, ydata, bins=10, figsize=figsize):
    # Build a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Plot a histogram
    plt.hist2d(xdata, ydata, density=True, bins=bins)
    plt.tight_layout()

def plot_discrete_histogram2d(x_data, y_data, x_label='X axis', y_label='Y axis', vmin=None, vmax=None, figsize=figsize):
    plt.close('all')
    plt.figure(figsize=figsize)

    # Define bin edges for both x and y with a width and height of 1 unit
    x_bins = np.arange(np.min(x_data) - 0.5, np.max(x_data) + 1, 1)
    y_bins = np.arange(np.min(y_data) - 0.5, np.max(y_data) + 1, 1)
    
    # Calculate 2D histogram with specified bins
    counts, xedges, yedges = np.histogram2d(x_data, y_data, bins=[x_bins, y_bins])

    # Define a colormap and plot with plt.pcolormesh for discrete box-dots appearance
    plt.pcolormesh(
        xedges - 0.05, yedges - 0.05, counts.T, 
        cmap='Blues', vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.3
    )
    
    # Add color bar for reference
    plt.colorbar(label='Counts')

    # Set axis limits
    plt.xlim(xedges[0], xedges[-1] - 0.1)
    plt.ylim(yedges[0], yedges[-1] - 0.1)
    
    # Set axis labels and grid
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data: pd.DataFrame, figsize=(10, 8), cmap='Blues', annot=False):
    plt.close('all')
    matrix = data.corr(method='pearson')
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, cmap=cmap, annot=annot, linewidth=0.5)  # Added linewidth parameter
    plt.tight_layout()
    plt.show()
	


def plot_series(data, labels=None,
										windows=None,
										predictions=None,
										highlights=None,
										val_start=None,
										test_start=None,
										threshold=None,
										figsize=figsize,
										xlabel=None,
										ylabel=None):
		# Open a new figure
		plt.close('all')
		plt.figure(figsize=figsize)

		
		x = data.index
		# Plot data
		plt.plot(x, data.values, zorder=0)
		# Rotated x ticks
		plt.xticks(rotation=45)
		# Plot labels
		if labels is not None:
				plt.scatter(labels.values, data.loc[labels],
										color=anomaly_color, zorder=2)
		# Plot windows
		if windows is not None:
				for _, wdw in windows.iterrows():
						plt.axvspan(wdw['begin'], wdw['end'],
												color=anomaly_color, alpha=0.3, zorder=1)
		
		# Plot training data
		if val_start is not None:
				plt.axvspan(x[0], val_start,
										color=training_color, alpha=0.1, zorder=-1)
		if val_start is None and test_start is not None:
				plt.axvspan(x[0], test_start,
										color=training_color, alpha=0.1, zorder=-1)
		if val_start is not None:
				plt.axvspan(val_start, test_start,
										color=validation_color, alpha=0.1, zorder=-1)
		if test_start is not None:
				plt.axvspan(test_start, x[-1],
										color=test_color, alpha=0.3, zorder=0)
		# Predictions
		if predictions is not None:
				plt.scatter(predictions.values, data.loc[predictions],
										color=prediction_color, alpha=.4, zorder=3)
		# Plot threshold
		if threshold is not None:
				plt.plot([x[0], x[-1]], [threshold, threshold], linestyle=':', color='tab:red')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.grid(':')
		plt.tight_layout() 

def load_data(data_folder, file_name) -> pd.DataFrame:
	file_path = os.path.join(data_folder, file_name + '.csv')
	df = pd.read_csv(file_path, parse_dates=['Date'], dtype={'Season': str})
	return df

def fetch_data_into_file(data_folder, file_name, start_year, end_year, leagues) -> None:
	url_template = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
	cols = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR", "HBP"]

	#Generate seasons list
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
					df = pd.read_csv(
							url_template.format(season=season, league=league)
					)
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
				print("No data for", season, league, url_template.format(season=season, league=league))
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

def extract_elo_history(data, team) -> pd.DataFrame:
	elo_history = []
	for index, row in data.iterrows():
		if row['HomeTeam'] == team:
			elo_history.append({
				'Date': row['Date'],
				'Opponent': row['AwayTeam'],
				'ELO': row['Home ELO'],
				'Result': row['FTR']
			})
		elif row['AwayTeam'] == team:
			elo_history.append({
				'Date': row['Date'],
				'Opponent': row['HomeTeam'],
				'ELO': row['Away ELO'],
				'Result': row['FTR']
			})
	return pd.DataFrame(elo_history)

def add_discrete_league_columns(data: pd.DataFrame) -> pd.DataFrame:
	leagues = data['Div'].unique()
	for league in leagues:
		data[league] = False

	for league in leagues:
		data.loc[data['Div'] == league, league] = True

	return data

def add_discrete_result_columns(data: pd.DataFrame) -> pd.DataFrame:
	results = data['FTR'].unique()
	for result in results:
		data[result] = False

	for result in results:
		data.loc[data['FTR'] == result, result] = True
	
	return data

def add_discrete_season_columns(data: pd.DataFrame) -> pd.DataFrame:
	seasons = data['Season'].unique()
	for season in seasons:
		data[season] = False

	for season in seasons:
		data.loc[data['Season'] == season, season] = True
	
	return data

class ELO(): #Kan gjøre slik at home_advantage lages slik at den helles gir home_factor for kamper der hjemme og borte har samme rating
	def __init__(self, data, init_rating = 1500, draw_factor=0.25, k_factor=32, home_advantage=100):
		self.data = data
		self.init_rating = init_rating
		self.draw_factor = draw_factor
		self.k_factor = k_factor
		self.home_advantage = home_advantage
		self.ratings = {}
		self.add_teams(data)

	def add_teams(self, data: pd.DataFrame):
		home_teams = data['HomeTeam'].unique()
		away_teams = data['AwayTeam'].unique()
		teams = list(set(home_teams) | set(away_teams))
		for team in teams:
			
			r = data[data['HomeTeam'] == team].iloc[0]
	
			if r['Div'] == 'E0':
				self.ratings[team] = self.init_rating
			elif r['Div'] == 'E1':
				self.ratings[team] = self.init_rating - 200
			elif r['Div'] == 'E2':
				self.ratings[team] = self.init_rating - 400
			elif r['Div'] == 'E3':
				self.ratings[team] = self.init_rating - 600
			else:
				self.ratings[team] = self.init_rating - 800
	
	def calculate_new_rating(self, home_elo, away_elo, result):
		if result == 'H':
			s_home, s_away = 1,0
		elif result == 'D':
			s_home, s_away = 0.5,0.5
		else:
			s_home, s_away = 0,1
		e_home, e_d, e_away = self.expect_result(home_elo + self.home_advantage, away_elo)

		new_rating_home = home_elo + self.k_factor * (s_home - (e_home+e_d/2))
		new_rating_away = away_elo + self.k_factor * (s_away - (e_away+e_d/2))
		return new_rating_home, new_rating_away
	
	def expect_result(self, home_elo, away_elo):
		elo_diff = home_elo - away_elo
		excepted_home_without_draws = 1 / (1 + 10 ** (-elo_diff / 400))
		expected_away_without_draws = 1 / (1 + 10 ** (elo_diff / 400))
		real_expected_draw = self.draw_factor*(1-abs(excepted_home_without_draws - expected_away_without_draws))
		real_expected_home = excepted_home_without_draws - real_expected_draw/2
		real_expected_away = expected_away_without_draws - real_expected_draw/2
		return real_expected_home, real_expected_draw, real_expected_away
	
	def perform_matchup(self, home_team, away_team, result) -> None:
		try:
			old_rating_home = self.ratings[home_team]
			old_rating_away = self.ratings[away_team]
			new_rating_home, new_rating_away = self.calculate_new_rating(old_rating_home, old_rating_away, result)
			self.ratings[home_team] = new_rating_home
			self.ratings[away_team] = new_rating_away
			return old_rating_home, old_rating_away
		except KeyError:
			print("One or both teams does not exist")
			return None
		
	def perform_simulations(self, data) -> pd.DataFrame:
		data['Home ELO'] = None
		data['Away ELO'] = None
		data['ELO diff'] = None
		for index, row in data.iterrows():
			old_rating_home, old_rating_away = self.perform_matchup(row['HomeTeam'], row['AwayTeam'], row['FTR'])
			data.at[index, 'Home ELO'] = old_rating_home
			data.at[index, 'Away ELO'] = old_rating_away
			data.at[index, 'ELO diff'] = old_rating_home - old_rating_away
		return data
	
	def get_probabilities(self, data) -> pd.DataFrame:
		data['Home_prob_ELO'] = None
		data['Draw_prob_ELO'] = None
		data['Away_prob_ELO'] = None
		for index, row in data.iterrows():
			home_prob, draw_prob, away_prob = self.expect_result(row['Home ELO'] + self.home_advantage, row['Away ELO'])
			data.at[index, 'Home_prob_ELO'] = home_prob
			data.at[index, 'Draw_prob_ELO'] = draw_prob
			data.at[index, 'Away_prob_ELO'] = away_prob
		return data
	

def get_metrics(data: pd.DataFrame, threshold: float):
	tp = []
	fp = []
	tn = []
	fn = []
	for index, row in data.iterrows():
		if row['Home_prob_ELO'] >= threshold and row['H']:
			tp.append(row)
		elif row['Home_prob_ELO'] >= threshold and not row['H']:
			fp.append(row)
		elif row['Home_prob_ELO'] < threshold and not row['H']:
			tn.append(row)
		elif row['Home_prob_ELO'] < threshold and row['H']:
			fn.append(row)
	return pd.Series(tp), pd.Series(fp), pd.Series(tn), pd.Series(fn)
class SimpleCostModel():
	def __init__(self, wrong, missed):
		self.wrong = wrong
		self.missed = missed

	def cost(self, data, threshold):
		tp, fp, tn, fn = get_metrics(data, threshold)
		return len(fp)*self.wrong + len(fn)*self.missed

def opt_thr(data, cmodel, thr_range):
	costs = [cmodel.cost(data, thr)for thr in thr_range]
	costs = np.array(costs)
	best_idx = np.argmin(costs)
	return thr_range[best_idx], costs[best_idx]


def get_all_matches_of_team(data, team):
	c = data.copy()
	return c[(c['HomeTeam'] == team) | (c['AwayTeam'] == team)]


def add_sequential_column(data: pd.DataFrame, home_column, away_column, n=5, operation='Sum', regard_opponent=False, include_current=False):
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
	new_column_name_home = home_column + '_' + operation + '_' + str(n) + ('_opponent' if regard_opponent else '')
	new_column_name_away = away_column + '_' + operation + '_' + str(n) + ('_opponent' if regard_opponent else '')
	data[new_column_name_home] = None
	data[new_column_name_away] = None
	teams = data['HomeTeam'].unique()
	for team in teams:
		matches = get_all_matches_of_team(data, team)
		scores = {}
		pos = 0 if not include_current else 1
		for index, row in matches.iterrows():
			start_pos = max(0, pos-n)
			relevant_matches = matches.iloc[start_pos:pos]
			s = 0
			if operation == 'Sum':
				for index_r,row_r in relevant_matches.iterrows():
					if row_r['HomeTeam'] == team:
						if regard_opponent:
							s += row_r[away_column]
						else:
							s += row_r[home_column]
					else:
						if regard_opponent:
							s += row_r[home_column]
						else:
							s += row_r[away_column]
			elif operation == 'Mean':
				for index_r, row_r in relevant_matches.iterrows():
					if row_r['HomeTeam'] == team:
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
			elif operation == 'Change':
				if len(relevant_matches) == 0:
					s = 0
				else:
					first_row = relevant_matches.iloc[0]
					last_row = relevant_matches.iloc[-1]
					first_score = first_row[home_column] if first_row['HomeTeam'] == team else first_row[away_column]
					last_score = last_row[home_column] if last_row['HomeTeam'] == team else last_row[away_column]
					s = last_score - first_score
			elif operation == 'Points':
				for index_r, row_r in relevant_matches.iterrows():
					if row_r['HomeTeam'] == team:
						if row_r['FTR'] == 'H':
							s += 3
						elif row_r['FTR'] == 'D':
							s += 1
						else:
							s += 0
					else:
						if row_r['FTR'] == 'A':
							s += 3
						elif row_r['FTR'] == 'D':
							s += 1
						else:
							s += 0
			scores[index] = s
			pos += 1
				
		
		for key, value in scores.items():
			if data.at[key, 'HomeTeam'] == team:
				data.at[key, new_column_name_home] = value
			else:
				data.at[key, new_column_name_away] = value
	return data



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

def get_cleaned_data(data: pd.DataFrame):
	data = data.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
	data.drop(columns='Referee', inplace=True) #Fjerner kolonnen Referee
	data.dropna(inplace=True) #Fjerner rader med manglende verdier
	data = data.reset_index(drop=True)
	return data