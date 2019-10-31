#import vital tools
from selenium import webdriver
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'IPAexGothic'

#browser open (chrome)
browser=webdriver.Chrome()

#ready for scraping
columnNames=[]
ETFComparisonsTable=[]
for num in range(99):
    browser.get("https://kabuoji3.com/stock/")
    stockSearch=browser.find_element_by_class_name("form_inputs")
    stockSearchForm=stockSearch.find_element_by_class_name("form_txt")
    stockSearchForm.send_keys("ETF")
    btnClick=browser.find_element_by_class_name("btn_submit")
    btnClick.click()

    #choose a stock out of list
    stockClick=browser.find_elements_by_class_name("clickable")
    if stockClick[num] == stockClick[-1]:
        break
    stockClick[num].find_element_by_tag_name("a").click()

    stockTable=browser.find_element_by_class_name("table_wrap")
    stockLine=stockTable.find_elements_by_tag_name("tr")

    #price scraping with calculation
    if len(stockLine)==302:
        ETFComparisons=[]
        for i in range(2,152):
            stockETFPriceAfter=stockLine[i-1].find_elements_by_tag_name("td")
            stockETFPriceBefore=stockLine[i].find_elements_by_tag_name("td")
            ETFComparison=float(stockETFPriceAfter[6].text)-float(stockETFPriceBefore[6].text)
            ETFComparisons.append(ETFComparison)

        stockETFPriceAfter=stockLine[151].find_elements_by_tag_name("td")
        stockETFPriceBefore=stockLine[153].find_elements_by_tag_name("td")
        ETFComparison=float(stockETFPriceAfter[6].text)-float(stockETFPriceBefore[6].text)
        ETFComparisons.append(ETFComparison)

        for i in range(154,302):
            stockETFPriceAfter=stockLine[i-1].find_elements_by_tag_name("td")
            stockETFPriceBefore=stockLine[i].find_elements_by_tag_name("td")
            ETFComparison=float(stockETFPriceAfter[6].text)-float(stockETFPriceBefore[6].text)
            ETFComparisons.append(ETFComparison)

        ETFComparisonsTable.append(ETFComparisons)

        #pick up title
        stockTitleBox=browser.find_element_by_class_name("base_box_ttl")
        stockTitle=stockTitleBox.find_element_by_class_name("jp").text
        columnNames.append(stockTitle)

#making ETF table
ETFTable=pd.DataFrame(ETFComparisonsTable)
ETFTable=ETFTable.T
ETFTable.columns=columnNames

#date scraping
browser.get("https://kabuoji3.com/stock/{}/".format(4307))
stockTable=browser.find_element_by_class_name("table_wrap")
stockLine=stockTable.find_elements_by_tag_name("tr")
dates=[]
for i in range(1,152):
    stockDate=stockLine[i].find_elements_by_tag_name("td")
    stockDate=stockDate[0].text
    dates.append(stockDate)
for i in range(153,302):
    stockDate=stockLine[i].find_elements_by_tag_name("td")
    stockDate=stockDate[0].text
    dates.append(stockDate)
df_date=pd.DataFrame()
df_date["date"]=dates
df_date["year"]=df_date["date"].apply(lambda x:int(x.split("-")[0]))
df_date["month"]=df_date["date"].apply(lambda x:int(x.split("-")[1]))
df_date["day"]=df_date["date"].apply(lambda x:int(x.split("-")[2]))

#stock scraping (comparison with yesterday)
browser.get("https://kabuoji3.com/stock/{}/".format(4307))
stockTable=browser.find_element_by_class_name("table_wrap")
stockLine=stockTable.find_elements_by_tag_name("tr")
targetStockComparisons=[]
for i in range(2,152):
    targetStockPriceAfter=stockLine[i-1].find_elements_by_tag_name("td")
    targetStockPriceBefore=stockLine[i].find_elements_by_tag_name("td")
    targetStockComparison=float(targetStockPriceAfter[6].text)-float(targetStockPriceBefore[6].text)
    targetStockComparisons.append(targetStockComparison)
targetStockPriceAfter=stockLine[151].find_elements_by_tag_name("td")
targetStockPriceBefore=stockLine[153].find_elements_by_tag_name("td")
targetStockComparison=float(targetStockPriceAfter[6].text)-float(targetStockPriceBefore[6].text)
targetStockComparisons.append(targetStockComparison)
for i in range(154,302):
    targetStockPriceAfter=stockLine[i-1].find_elements_by_tag_name("td")
    targetStockPriceBefore=stockLine[i].find_elements_by_tag_name("td")
    targetStockComparison=float(targetStockPriceAfter[6].text)-float(targetStockPriceBefore[6].text)
    targetStockComparisons.append(targetStockComparison)
df=pd.DataFrame(targetStockComparisons)
df.columns=["(株)野村総合研究所：前日比"]

#add table
stockPriceTable=pd.concat([df_date,ETFTable],axis=1)
stockPriceTable=pd.concat([stockPriceTable,df],axis=1)

#prepare for making target values
df_next=df.copy()
df_next.columns=["(株)野村総合研究所：翌日比"]

#date scraping for target values
browser.get("https://kabuoji3.com/stock/{}/".format(4307))
stockTable=browser.find_element_by_class_name("table_wrap")
stockLine=stockTable.find_elements_by_tag_name("tr")
dates=[]
for i in range(2,152):
    stockDate=stockLine[i].find_elements_by_tag_name("td")
    stockDate=stockDate[0].text
    dates.append(stockDate)
for i in range(153,302):
    stockDate=stockLine[i].find_elements_by_tag_name("td")
    stockDate=stockDate[0].text
    dates.append(stockDate)
df_date2=pd.DataFrame()
df_date2["date"]=dates

#making target values table
df_next=pd.concat([df_date2,df_next],axis=1)
df_next.index=df_date2["date"]

#prepare for complete table
table=stockPriceTable[1:299].copy()
table.index=table["date"]

#making complete table
table["(株)野村総合研究所：翌日比"]=df_next["(株)野村総合研究所：翌日比"]

#making csv file
table.to_csv("stockPriceData.csv",index=False)

#import vital tools
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from selenium import webdriver
import requests
import json

#reading csv file (*ETF=Exhange Traded Funds)
train=pd.read_csv("stockPriceData.csv")

df = pd.read_csv("stockPriceData.csv")
dfd = df.reset_index().T.reset_index().T.values.tolist().pop(0)
del dfd[0:5]

#ready for making machine learning model
features = dfd
del features[-1]
x=train[features]
y=train["(株)野村総合研究所：翌日比"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.27)

#model making and prediction
model=RandomForestRegressor(n_estimators=1000)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#make result score and get accuracy score
testUpDown=[]
for test in y_test:
    if test>0:
        testUpDown.append(1)
    else:
        testUpDown.append(-1)
predUpDown=[]
for pred in y_pred:
    if pred>0:
        predUpDown.append(1)
    else:
        predUpDown.append(-1)

#feature evaluation and plots
feature_imp = pd.Series(model.feature_importances_,index=features).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.figure(figsize=(30,50))
plt.rcParams['font.family'] = 'IPAexGothic'

#ready for scraping
browser=webdriver.Chrome()
columnNames=[]
ETFComparisonsTable=[]
ETFfeatures = features.copy()
del ETFfeatures[-1]
for feature in ETFfeatures:
    feature=feature.split(" ")[0]

    browser.get("https://kabuoji3.com/stock/{}/".format(feature))
    stockTable=browser.find_element_by_class_name("table_wrap")
    stockLine=stockTable.find_elements_by_tag_name("tr")

    #price scraping with calculation
    if len(stockLine)==302:
        ETFComparisons=[]
        stockETFPriceAfter=stockLine[1].find_elements_by_tag_name("td")
        stockETFPriceBefore=stockLine[2].find_elements_by_tag_name("td")
        ETFComparison=float(stockETFPriceAfter[6].text)-float(stockETFPriceBefore[6].text)
        ETFComparisons.append(ETFComparison)
        ETFComparisonsTable.append(ETFComparisons)

        #pick up title
        stockTitleBox=browser.find_element_by_class_name("base_box_ttl")
        stockTitle=stockTitleBox.find_element_by_class_name("jp").text
        columnNames.append(stockTitle)

#making ETF table
ETFTable=pd.DataFrame(ETFComparisonsTable)
ETFTable=ETFTable.T
ETFTable.columns=columnNames


#date scraping and stock scraping (comparison with yesterday)
browser.get("https://kabuoji3.com/stock/{}/".format(4307))
stockTable=browser.find_element_by_class_name("table_wrap")
stockLine=stockTable.find_elements_by_tag_name("tr")

dates=[]
stockDate=stockLine[1].find_elements_by_tag_name("td")
stockDate=stockDate[0].text
dates.append(stockDate)

df_date=pd.DataFrame()
df_date["date"]=dates
df_date["year"]=df_date["date"].apply(lambda x:int(x.split("-")[0]))
df_date["month"]=df_date["date"].apply(lambda x:int(x.split("-")[1]))
df_date["day"]=df_date["date"].apply(lambda x:int(x.split("-")[2]))

targetStockComparisons=[]
targetStockPriceAfter=stockLine[1].find_elements_by_tag_name("td")
targetStockPriceBefore=stockLine[2].find_elements_by_tag_name("td")
targetStockComparison=float(targetStockPriceAfter[6].text)-float(targetStockPriceBefore[6].text)
targetStockComparisons.append(targetStockComparison)

df=pd.DataFrame(targetStockComparisons)
df.columns=["(株)野村総合研究所：前日比"]

#add table
stockPriceTable=pd.concat([df_date,ETFTable],axis=1)
stockPriceTable=pd.concat([stockPriceTable,df],axis=1)


#ready for future price prediction
valueX=stockPriceTable[features]
pred=model.predict(valueX)
#make result score and get accuracy score
predPriceUpDown="?"
if pred>0:
    predPriceUpDown="上昇"
else:
    predPriceUpDown="下落"

#telling result
resultNotification="株価予測:「4307:(株)野村総合研究所」\n"+stockDate+"現時点での予測値は"+str(float(targetStockPriceAfter[6].text)+float(pred))+"円。\nよって価格は"+predPriceUpDown+"見込みです。"
browser.quit()

slackURL="https://hooks.slack.com/services/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"

def send_slack(content):
    payload={
        "text":content,
        "username":"PythonStockForecast",
        "icon_emoji":":snake:"
    }
    data=json.dumps(payload)
    requests.post(slackURL,data)
send_slack(resultNotification)