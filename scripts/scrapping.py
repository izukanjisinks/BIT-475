# Import necessary libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Step 1: Scrape Standings Data
standings_url = "https://fbref.com/en/comps/9/2021-2022/2021-2022-Premier-League-Stats"
data = requests.get(standings_url)

# Step 2: Parse HTML using BeautifulSoup
soup = BeautifulSoup(data.text, 'html.parser')
standings_table = soup.select('table.stats_table')[0]

# Step 3: Extract Team URLs from Standings Table
links = standings_table.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if '/squads/' in l]
team_urls = [f"https://fbref.com{l}" for l in links]

# Step 4: Scrape Team Data
data = requests.get(team_urls[0])

# Step 5: Read Match Data from HTML Tables
all_matches = pd.read_html(data.text, match="Scores & Fixtures")
match_df = pd.concat(all_matches)

# Step 6: Clean and Rename Columns
match_df.columns = [c.lower() for c in match_df.columns]

# Step 7: Save the Match Data to CSV
match_df.to_csv("matches.csv")
