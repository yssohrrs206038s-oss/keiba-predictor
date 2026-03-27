import pandas as pd
from keiba_predictor.scraper.shutuba_scraper import scrape_shutuba
result = scrape_shutuba('202606030111')
horses = result['horses']
if isinstance(horses, list):
    df = pd.DataFrame(horses)
else:
    df = horses
print(df[['horse_name', 'odds']].head(5))
