from keiba_predictor.scraper.shutuba_scraper import scrape_shutuba
result = scrape_shutuba('202606030111')
print(type(result))
print(result.keys() if isinstance(result, dict) else result[:2])
