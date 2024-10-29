import pandas as pd
import numpy as np
np.random.seed(42)


class ELO():
	def __init__(self, init_rating = 1500, teams = [], draw_factor=0.25, k_factor=32, home_advantage=0):
		self.init_rating = init_rating
		self.ratings = {}
		self.draw_factor = draw_factor
		self.k_factor = k_factor
		self.home_advantage = home_advantage
		for team_name in teams:
			self.add_team(team_name)

	def add_team(self, team_name):
		self.ratings[team_name] = self.init_rating

	def calculate_new_rating(self, home_elo, away_elo, result) -> float:
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
	
	def expect_result_teams(self, home_team, away_team):
		try: 
			return self.expect_result(self.ratings[home_team], self.ratings[away_team], self.draw_factor)
		except KeyError:
			print("One or both teams does not exist")
			return None
	
	def expect_result(self, home_elo, away_elo): #Her kan draw_factor tweakes. 0.25 er standard
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
			return old_rating_home, old_rating_away, new_rating_home, new_rating_away
		except KeyError:
			print("One or both teams does not exist")
			return None
		
	def perform_simulations(self, data) -> pd.DataFrame:
		data['Home ELO before match'] = None
		data['Away ELO before match'] = None
		data['Home ELO'] = None
		data['Away ELO'] = None
		data['Home ELO change'] = None
		data['Away ELO change'] = None
		for index, row in data.iterrows():
			old_rating_home, old_rating_away , new_rating_home, new_rating_away = self.perform_matchup(row['HomeTeam'], row['AwayTeam'], row['FTR'])
			data.at[index, 'Home ELO before match'] = old_rating_home
			data.at[index, 'Away ELO before match'] = old_rating_away
			data.at[index, 'Home ELO'] = new_rating_home
			data.at[index, 'Away ELO'] = new_rating_away
			data.at[index, 'Home ELO change'] = new_rating_home - old_rating_home
			data.at[index, 'Away ELO change'] = new_rating_away - old_rating_away
		return data
	
	def get_probabilities(self, data) -> pd.DataFrame:
		data['Home_prob'] = None
		data['Draw_prob'] = None
		data['Away_prob'] = None
		for index, row in data.iterrows():
			home_prob, draw_prob, away_prob = self.expect_result(row['Home ELO before match'], row['Away ELO before match'])
			data.at[index, 'Home_prob'] = home_prob
			data.at[index, 'Draw_prob'] = draw_prob
			data.at[index, 'Away_prob'] = away_prob
		return data


class ShortTermForm():
	def __init__(self):
		pass


def evaluate_probability_prediction(data: pd.DataFrame) -> float:
    data['Correct_guess'] = None
    for index, row in data.iterrows():
        row_max = max(row['Home_prob'], row['Draw_prob'], row['Away_prob'])
        max_cols = row[row == row_max].index
        max_col = np.random.choice(max_cols)
        max_res = {
			'Home_prob': 'H',
			'Draw_prob': 'D',
			'Away_prob': 'A'
		}.get(max_col)
        if row['FTR'] == max_res:
            data.at[index, 'Correct_guess'] = True
        else:
            data.at[index, 'Correct_guess'] = False
    return data['Correct_guess'].sum() / len(data)


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


def calculate_team_stats(data, n=5):
    """
    Calculate the goals scored, goals conceded, and goal difference over the last N matches for each team.
    """
    data["Home Goals Last 5"] = None
    data["Away Goals Last 5"] = None
    data["Home Conceded Last 5"] = None
    data["Away Conceded Last 5"] = None
    data["Home Goal Difference Last 5"] = None
    data["Away Goal Difference Last 5"] = None

    for index, row in data.iterrows():
        # Home team stats
        home_team = row["HomeTeam"]
        last_home_matches = get_last_n_matches(home_team, index, data, n)
        home_goals_scored = last_home_matches["FTHG"].sum()  # Goals scored at home
        home_goals_conceded = last_home_matches["FTAG"].sum()  # Goals conceded at home

        # Away team stats
        away_team = row["AwayTeam"]
        last_away_matches = get_last_n_matches(away_team, index, data, n)
        away_goals_scored = last_away_matches["FTAG"].sum()  # Goals scored away
        away_goals_conceded = last_away_matches["FTHG"].sum()  # Goals conceded away

        # Update the columns with the calculated values
        data.at[index, "Home Goals Last 5"] = home_goals_scored
        data.at[index, "Away Goals Last 5"] = away_goals_scored
        data.at[index, "Home Conceded Last 5"] = home_goals_conceded
        data.at[index, "Away Conceded Last 5"] = away_goals_conceded
        data.at[index, "Home Goal Difference Last 5"] = (
            home_goals_scored - home_goals_conceded
        )
        data.at[index, "Away Goal Difference Last 5"] = (
            away_goals_scored - away_goals_conceded
        )

    return data
