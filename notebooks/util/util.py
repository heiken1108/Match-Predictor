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

  if not os.path.exists(data_folder):
    os.makedirs(data_folder)
  file_path = os.path.join(data_folder, file_name + ".csv")
  df.to_csv(file_path, index=False)
  print("Data fetched and saved to", file_path)

def load_data(data_folder, file_name) -> pd.DataFrame:
  file_path = os.path.join(data_folder, file_name + '.csv')
  df = pd.read_csv(file_path)
  return df