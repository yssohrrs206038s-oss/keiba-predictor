from keiba_predictor.scraper.shutuba_scraper import scrape_shutuba
result = scrape_shutuba('202606030111')
for h in result['horses'][:3]:
    print(h['horse_name'], h['odds'])
