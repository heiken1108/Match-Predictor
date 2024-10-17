import os
import pandas as pd
from matplotlib import pyplot as plt

anomaly_color = 'sandybrown'
prediction_color = 'yellowgreen'
training_color = 'yellowgreen'
validation_color = 'gold'
test_color = 'coral'
figsize=(9, 3)

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
      df_tmp.append(df)
  df = pd.concat(df_tmp)

  if not os.path.exists(data_folder):
    os.makedirs(data_folder)
  file_path = os.path.join(data_folder, file_name + ".csv")
  df.to_csv(file_path, index=False)
  print("Data fetched and saved to", file_path)

def load_data(data_folder, file_name) -> pd.DataFrame:
  file_path = os.path.join(data_folder, file_name + '.csv')
  df = pd.read_csv(file_path)
  return df

#Div, Date, Team, Opponent, Home/Away, Result (W, D, L), Goals, Goals conceded, Goal difference, Shots, Shots on Target, Percentage of shorts on target, Shots against, Shots against on target, Percentage of shots against on target, Referee 
def transform_dataframe_to_base_format(df) -> pd.DataFrame:
  new_rows = []
    
  for index, row in df.iterrows():
    # Home team row
    home_row = {
      'Match index': index,
      'Div': row['Div'],
      'Date': row['Date'],
      'Team': row['HomeTeam'],
      'Opponent': row['AwayTeam'],
      'Home/Away': 'H',
      'Result': 'W' if row['FTR'] == 'H' else ('D' if row['FTR'] == 'D' else 'L'),
      'Goals scored': row['FTHG'],
      'Goals conceded': row['FTAG'],
      'Goal difference': row['FTHG'] - row['FTAG'],
      'Shots': row['HS'],
      'Shots on Target': row['HST'],
      'Percentage of shots on target': round(row['HST'] / row['HS'] * 100, 2) if row['HS'] > 0 else 0,
      'Shots against': row['AS'],
      'Shots against on target': row['AST'],
      'Percentage of shots against on target': round(row['AST'] / row['AS'] * 100, 2) if row['AS'] > 0 else 0,
      'Referee': row['Referee']
    }
    new_rows.append(home_row)
    
    # Away team row
    away_row = {
      'Match index': index,
      'Div': row['Div'],
      'Date': row['Date'],
      'Team': row['AwayTeam'],
      'Opponent': row['HomeTeam'],
      'Home/Away': 'A',
      'Result': 'W' if row['FTR'] == 'A' else ('D' if row['FTR'] == 'D' else 'L'),
      'Goals scored': row['FTAG'],
      'Goals conceded': row['FTHG'],
      'Goal difference': row['FTAG'] - row['FTHG'],
      'Shots': row['AS'],
      'Shots on Target': row['AST'],
      'Percentage of shots on target': round(row['AST'] / row['AS'] * 100, 2) if row['AS'] > 0 else 0,
      'Shots against': row['HS'],
      'Shots against on target': row['HST'],
      'Percentage of shots against on target': round(row['HST'] / row['HS'] * 100, 2) if row['HS'] > 0 else 0,
      'Referee': row['Referee']
    }
    new_rows.append(away_row)
  
  # Create the new DataFrame
  new_df = pd.DataFrame(new_rows)
  
  return new_df

def get_team_matches(df, teamname) -> pd.DataFrame: 
  return df[(df['Team'] == teamname)]

def get_all_teams(df: pd.DataFrame) -> list:
  return df['Team'].unique().tolist()

def plot_series(data, x_values=None, labels=None,
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

    if x_values == []:
      x = range(len(data))
    elif x_values is not None:
      x = x_values
    else:
      x = data.index
    print(x)
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

class EloRating():
  def __init__(self, init_rating = 1500, teams = [], draw_factor=0.25, k_factor=32):
    self.init_rating = init_rating
    self.ratings = {}
    self.draw_factor = draw_factor
    self.k_factor = k_factor
    for team_name in teams:
      self.add_team(team_name)

  def add_team(self, team_name, rating = None):
    self.ratings[team_name] = rating if rating is not None else self.init_rating

  def perform_matchup(self, team, opponent, result) -> None:
    try:
      new_rating_team, new_rating_opponent = self.calculate_new_rating(self.ratings[team], self.ratings[opponent], result)
      self.ratings[team] = new_rating_team
    except KeyError:
      print("One or both teams does not exist")
      return None

  def calculate_new_rating(self, team_elo, opponent_elo, result) -> float:
    if result == 'W':
      s_team, s_opponent = 1,0
    elif result == 'D':
      s_team, s_opponent = 0.5,0.5
    else:
      s_team, s_opponent = 0,1

    e_team, e_d, e_opponent = self.expect_result(team_elo, opponent_elo)

    new_rating_team = team_elo + self.k_factor * (s_team - (e_team+e_d/2)) #Blir goofy her fordi det er alltid litt lavere prob for win nå man bryr seg om draw også
    new_rating_opponent = opponent_elo + self.k_factor * (s_opponent - (e_opponent+e_d/2))
    return new_rating_team, new_rating_opponent

  def expect_result_teams(self, team, opponent):
    try: 
      return self.expect_result(self.ratings[team], self.ratings[opponent], self.draw_factor)
    except KeyError:
      print("One or both teams does not exist")
      return None

  def expect_result(self, team_elo, opponent_elo): #Her kan draw_factor tweakes. 0.25 er standard
    elo_diff = team_elo - opponent_elo
    excepted_win_without_draws = 1 / (1 + 10 ** (-elo_diff / 400))
    expected_loss_without_draws = 1 / (1 + 10 ** (elo_diff / 400))
    real_expected_draw = self.draw_factor*(1-abs(excepted_win_without_draws - expected_loss_without_draws))
    real_expected_win = excepted_win_without_draws - real_expected_draw/2
    real_expected_loss = expected_loss_without_draws - real_expected_draw/2
    return real_expected_win, real_expected_draw, real_expected_loss
  
  def perform_simulations(self, matches) -> None:
    for index, row in matches.iterrows():
      self.perform_matchup(row['Team'], row['Opponent'], row['Result'])

class ShortTermForm():
  def __init__(self):
    pass