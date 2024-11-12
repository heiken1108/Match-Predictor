# Match-Predictor

Project work for the subject Atrifical Intelligence in Industry

### Description of columns in the dataset

-   Div = League Division
-   Date = Match Date
-   HomeTeam = Home Team
-   AwayTeam = Away Team
-   FTHG = Full Time Home Team Goals
-   FTAG = Full Time Away Team Goals
-   FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
-   HTHG = Half Time Home Team Goals
-   HTAG = Half Time Away Team Goals
-   HTR = Half Time Result (H=Home Win, D=Draw, A=Away Win)
-   Referee = Match Referee
-   HS = Home Team Shots
-   AS = Away Team Shots
-   HST = Home Team Shots on Target
-   AST = Away Team Shots on Target
-   HF = Home Team Fouls Committed
-   AF = Away Team Fouls Committed
-   HC = Home Team Corners
-   AC = Away Team Corners
-   HY = Home Team Yellow Cards
-   AY = Away Team Yellow Cards
-   HR = Home Team Red Cards
-   AR = Away Team Red Cards
-   PSH = Pinnacle home win odds
-   PSD = Pinnacle draw odds
-   PSA = Pinnacle away win odds

## How to run

```
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```


1. Data Exploration.
 - Forklare problemet
 - Forklare datasettet
 - Cleaning, vise rader som har problemer
 - Vis statstikk om datasettet
 - Vise bottom-line av å bare tippe hjemmeseier. 43%
2. ELO rating
 - Forklare grunnlag bak langsiktig form
 - Forklare konseptet med ELO
 - Få inn kolonner med ELO
 - Se på litt utvikling
 - Vise evaluering av ELO med accuracy
3. Match rating
 - Forklare grunnlag bak kortvarig form
 - Forklare mulig match rating
 - Vise statistikk som kan hjelpe med å klassifisere
 - Vise evaluering av match rating med accuracy
4. Data Pipeline
 - Bruke forrige notebook til å forklare at vi ønsker å lage features som kan beskrive konteksten til kampen for å predikere utfall
 - Lage features
 - Cleane dataen, konvertere til numerisk for alle
 - Fjerne
 - Lagre til fil
5. Random Forest
 - Prøve ut det vi har
 - Komme fram til at det fungerer dårlig. Må gjøre endringer
 - Skal bytte ut accuracy, skal bare ta inn premier league, skal heller predikere målforskjell (expected goal difference)
6. 
