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

def get_article_data(url):
    driver.get(url)
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    body = soup.find("body")
    # print(body)
    main_content = body.find("main", attrs={"id": "maincontent"})
    return main_content

articles = []
titles = []
categories = []
links = []
df = pd.read_csv("data/nhs_articles_links.csv")
print(len(df))
# print(df["category"].value_counts())
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)

print(df)

base_url = "https://www.nhs.uk/"
for i in range(len(df)):
    try:
        url = base_url + df.iloc[i]["link"]
        category = df.iloc[i]["category"]
        title = df.iloc[i]['title']
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        continue
    if category != "medicines":
        
        print(f"Fetching data from {url}")
        whole_article = get_article_data(url).get_text().replace("\n\n", "\n").strip()
        
        
    else:
        print(f"Fetching data from {url}")
        main_content = get_article_data(url)
        if "pregnancy, breastfeeding and fertility" in main_content.get_text().lower() or "side effects" in main_content.get_text().lower():
            whole_article = ""
            # print("True")
            key_links = main_content.find_all("li", attrs={"class": "nhsuk-hub-key-links__list-item beta-hub-key-links__list-item"})
            for url in key_links:
                a = url.find("a")
                if a:
                    link = a["href"]
                    whole_article += get_article_data(link).get_text().replace("\n\n", "\n").strip()
                    # print(whole_article)

        else:
            whole_article = main_content.get_text().replace("\n\n", "\n").strip()
            # print("False")
        # print(main_content)
    articles.append(whole_article)
    titles.append(title)
    links.append(url)
    categories.append(category)

 
    d = {
        "title": titles,
        "link": links,
        "category": categories,
        "article": articles
    }
    df2 = pd.DataFrame(data=d)
    df2.to_csv("data/nhs_articles.csv", index=False)

       
        