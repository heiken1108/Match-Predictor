import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize



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


def plot_multiple_lines(data, x_col, line_cols=[], line_labels=[], x_label="Threshold", y_label="Metric Value", title="Model Performance Metrics vs Threshold", figsize=figsize):
	plt.close('all')
	plt.figure(figsize=figsize)
	colors = ['blue', 'orange', 'green', 'red', 'purple', 'black', 'gray', 'lightgreen', 'yellow', 'pink']
	for index, line in enumerate(line_cols):
		plt.plot(data[x_col], data[line], label=line_labels[index], color=colors[index])
	#plt.plot(results_df["threshold"], results_df["training_accuracy"], label="Training Accuracy", color="blue")
	#plt.plot(results_df["threshold"], results_df["test_accuracy"], label="Test Accuracy", color="orange")
	#plt.plot(results_df["threshold"], results_df["AUC_0"], label="AUC Class 0", color="green")
	#plt.plot(results_df["threshold"], results_df["AUC_1"], label="AUC Class 1", color="red")
	#plt.plot(results_df["threshold"], results_df["AUC_2"], label="AUC Class 2", color="purple")
	#plt.plot(results_df["threshold"], results_df["fraction"], label="Fraction of Data", color="black")

	# Add plot labels and legend
	plt.xlabel("Threshold")
	plt.ylabel("Metric Value")
	plt.title("Model Performance Metrics vs Threshold")
	plt.legend()
	plt.grid(True)
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
	df = pd.read_csv(file_path, dtype={'Season': str})
    
	if 'Date' in df.columns:
		df = pd.read_csv(file_path, parse_dates=['Date'], dtype={'Season': str})
    
	return df

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

def fetch_data_into_file(data_folder, file_name, start_year, end_year, leagues) -> None:
	url_template = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
	cols = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR", "HBP", "PSH", "PSD", "PSA"]

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


def perform_picks(data, confidence_threshold):
    correct, wrong, skipped = [], [], []
    for index, row in data.iterrows():
        max_prob = max(row['Home Prob'], row['Draw Prob'], row['Away Prob'])
        if max_prob < confidence_threshold:
            skipped.append(row)
            continue
        GD = row['FTHG'] - row['FTAG']
        if row['Home Prob'] == max_prob and GD > 0:
            correct.append(row)
        elif row['Draw Prob'] == max_prob and GD == 0:
            correct.append(row)
        elif row['Away Prob'] == max_prob and GD < 0:
            correct.append(row)
        else:
            wrong.append(row)
    return pd.Series(correct), pd.Series(wrong), pd.Series(skipped)
        


class PickCostModel:
    def __init__(self, wrong, skipped):
        self.wrong = wrong
        self.skipped = skipped
    
    def cost(self, data, confidence_threshold):
        correct, wrong, skipped = perform_picks(data, confidence_threshold)
        return len(wrong) * self.wrong + len(skipped) * self.skipped
	



def prepare_binary_data(df):
    # Create binary target variable (y): 1 for win, 0 for draw or loss
    df['win'] = (df['FTHG'] > df['FTAG']).astype(int)
    
    # Select features (excluding unnecessary columns)
    feature_columns = ['ELO diff',
                      'Diff_goals_scored', 'Diff_goals_conceded', 'Matchrating',
                      'Diff_points', 'Diff_change_in_ELO', 'Diff_opposition_mean_ELO',
                      'Diff_shots_on_target_attempted', 'Diff_shots_on_target_allowed',
                      'Diff_shots_attempted', 'Diff_shots_allowed', 'Diff_corners_awarded',
                      'Diff_corners_conceded', 'Diff_fouls_commited', 'Diff_fouls_suffered',
                      'Diff_yellow_cards', 'Diff_red_cards']
    
    X = df[feature_columns].copy()
    y = df['win'].copy()
    
    return X, y

def prepare_data(df):
    # Create target variable (y)
    df['outcome'] = np.where(df['FTHG'] > df['FTAG'], 'win',
                           np.where(df['FTHG'] == df['FTAG'], 'draw', 'loss'))
    
    # Select features (excluding unnecessary columns)
    feature_columns = ['ELO diff', 'Home_prob_ELO', 'Draw_prob_ELO', 'Away_prob_ELO',
                      'Diff_goals_scored', 'Diff_goals_conceded', 'Matchrating',
                      'Diff_points', 'Diff_change_in_ELO', 'Diff_opposition_mean_ELO',
                      'Diff_shots_on_target_attempted', 'Diff_shots_on_target_allowed',
                      'Diff_shots_attempted', 'Diff_shots_allowed', 'Diff_corners_awarded',
                      'Diff_corners_conceded', 'Diff_fouls_commited', 'Diff_fouls_suffered',
                      'Diff_yellow_cards', 'Diff_red_cards']
    
    X = df[feature_columns]
    y = df['outcome']
    
    return X, y

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = data.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    if 'Referee' in data.columns:
        data.drop(columns="Referee", inplace=True)  # Fjerner kolonnen Referee
    data.dropna(inplace=True)  # Fjerner rader med manglende verdier
    data = data.reset_index(drop=True)
    return data

def plot_binary_roc_curve(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a binary classifier (using Random Forest)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Get predictions
    y_score = clf.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Win Prediction')
    plt.legend(loc="lower right")
    
    # Calculate and print additional metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_pred = clf.predict(X_test)
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    
    return roc_auc, clf

def plot_roc_curves(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a multi-class classifier (using Random Forest as an example)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Binarize the labels
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    
    # Get predictions
    y_score = clf.predict_proba(X_test)
    
    # Calculate ROC curve and ROC area for each class
    n_classes = len(lb.classes_)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    for i, color, cls in zip(range(n_classes), colors, lb.classes_):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve for {cls} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Match Outcome Classification')
    plt.legend(loc="lower right")
    
    # Return the AUC scores and the classifier
    return roc_auc, clf

def analyze_feature_importance(clf, feature_names):
    # Get feature importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("\nFeature Importance Ranking:")
    for f in range(len(feature_names)):
        print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    
    return importances, indices


def plot_bars(data, figsize=None, tick_gap=1, series=None, title=None,
              xlabel=None, ylabel=None, std=None):
    plt.figure(figsize=figsize)
    # x = np.arange(len(data))
    # x = 0.5 + np.arange(len(data))
    # plt.bar(x, data, width=0.7)
    # x = data.index-0.5
    x = data.index
    plt.bar(x, data, width=0.7, yerr=std)
    # plt.bar(x, data, width=0.7)
    if series is not None:
        # plt.plot(series.index-0.5, series, color='tab:orange')
        plt.plot(series.index, series, color='tab:orange')
    if tick_gap > 0:
        plt.xticks(x[::tick_gap], data.index[::tick_gap], rotation=45)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(linestyle=':')
    plt.tight_layout()