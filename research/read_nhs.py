import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

chrome_options = Options()
chrome_options.add_argument("--headless")

links = []
titles = []
rel_categories = []

base_url = "https://www.nhs.uk/"
categories = ["conditions", "symptoms", "tests-and-treatments", "medicines"]
for category in categories:

    url = f"{base_url}{category}"
    print(f"Fetching data from {url}")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    time.sleep(2)  

    
    
    page_source = driver.page_source
    driver.quit()

    soup = BeautifulSoup(page_source, 'html.parser')
    content_card = soup.find_all("div", attrs={"class":'nhsuk-card__content nhsuk-card__content--feature'})
    # print(len(content_card))
    for alphabet in content_card:
        content_items = alphabet.find_all("li")
        # print(len(content_items))
        for i in content_items:
            # print(i)
            a = i.find("a")
            link = a["href"]
            title = a.text.strip()
            titles.append(title)
            links.append(link)
            rel_categories.append(category)

d = {
    "title": titles,
    "link": links,
    "category": rel_categories
}
df = pd.DataFrame(data=d)
df.to_csv("data/nhs_articles_links.csv", index=False)

    
            
        
    
   