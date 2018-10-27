import numpy as np
import pandas as pd
import bs4
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup

my_url ='https://www.newegg.com/Product/ProductList.aspx?Submit=ENE&DEPA=0&Order=BESTMATCH&Description=graphics+card&ignorear=0&N=-1&isNodeId=1'
uClient = uReq(my_url)

page_html = uClient.read()
uClient.close()
page_soup = soup(page_html , "html.parser")
containers = page_soup.findAll("div", {"class" : "item-container"})

file_name = "products.csv"
f = open(file_name, "w")
headers = "brand, Product_name, shipping\n"
f.write(headers)
for container in containers:
    brand = container.div.div.a.img["title"]
    
    title_container = container.findAll("a", {"class" : "item-title"})
    product_name = title_container[0].text
    
    shipping_container = container.findAll("li", {"class" : "price-ship"})
    shipping = shipping_container[0].text.strip()
    
#     print("brand : " + brand)
#     print("product_name : " + product_name)
#     print("shipping : " + shipping)
#     print("")
    
    f.write(brand + "," + product_name.replace(",", "|") + "," + shipping + "\n")
f.close()

train_df = pd.read_csv("products.csv")