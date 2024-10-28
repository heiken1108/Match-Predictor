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

def plot_histogram(data: pd.DataFrame, figsize=figsize):
	bins = np.arange(data.values.min() - 0.5, data.values.max() + 1.5, 1)
	data.values.hist(bins=bins, figsize=figsize, edgecolor='black', rwidth=0.9)

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

def plot_histogram(data, bins=10, vmin=None, vmax=None, figsize=figsize):
    # Build a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Plot a histogram
    plt.hist(data, density=True, bins=bins)
    # Update limits
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
	cols = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR", "PSH", "PSD", "PSA", "HBP"]

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

class ELO():
	def __init__(self, data, init_rating = 1500, draw_factor=0.25, k_factor=32, home_advantage=0):
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
			home_prob, draw_prob, away_prob = self.expect_result(row['Home ELO'], row['Away ELO'])
			data.at[index, 'Home_prob_ELO'] = home_prob
			data.at[index, 'Draw_prob_ELO'] = draw_prob
			data.at[index, 'Away_prob_ELO'] = away_prob
		return data