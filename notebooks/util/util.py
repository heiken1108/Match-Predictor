import os
import pandas as pd

def test_method():
  print("Hello World")

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
      url = url_template.format(season=season, league=league)
      df = pd.read_csv(url)
      existing_cols = [col for col in cols if col in df.columns]
      df = df[existing_cols]
      df_tmp.append(df)
  df = pd.concat(df_tmp)
  df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

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
