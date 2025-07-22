import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from scipy import stats
import statistics


# load irctc data from excel file
# -- IRCTC Stock Price Data ----
def irctc_data(pth, sht):
    d2 = pd.read_excel(pth, sheet_name=sht)
    print(d2.head())
    return d2

# calculating mean and variance of stock price
#EXP A3 1)
def calc_basicstatistics(d2):
    print(statistics.mean(d2.iloc[:, 3]))
    print(statistics.variance(d2["Price"]))

# find mean of prices where day is wednesday
#EXP A3 2)
def mean_wed(d2, day="Wed"):
    d = [d2.iloc[i]["Price"] for i in range(d2.shape[0]) if d2.iloc[i]["Day"] == day]
    print(statistics.mean(d))

# avg price in data where month is april
#EXP A3 3)
def m_mp(d2, month="Apr"):
    d = [d2.iloc[i]["Price"] for i in range(d2.shape[0]) if d2.iloc[i]["Month"] == month]
    print(statistics.mean(d))

# get mean of price drops from Chg% column
#EXP A3 4)
def negativechange_mean(d2):
    d = [d2.iloc[i]["Chg%"] for i in range(d2.shape[0]) if d2.iloc[i]["Chg%"] < 0]
    print(statistics.mean(d))

# compare positive change on wed vs other days
#EXP A3 5)
def wed_vs_others_positivechange(d2):
    a, b = [], []
    for i in range(d2.shape[0]):
        if d2.iloc[i]["Chg%"] > 0 and d2.iloc[i]["Day"] == "Wed":
            a.append(d2.iloc[i]["Chg%"])
        elif d2.iloc[i]["Chg%"] > 0:
            b.append(d2.iloc[i]["Chg%"])
    print(len(a)/(len(b)+len(a)))

# calc how often stock went up on wednesday
#EXP A3 
def wednesday_positive_chg_fraction(d2):
    Wed_Prof, Wed = 0, 0
    for i in range(d2.shape[0]):
        if d2.iloc[i]["Chg%"] > 0 and d2.iloc[i]["Day"] == "Wed":
            Wed_Prof += 1
            Wed += 1
        elif d2.iloc[i]["Day"] == "Wed":
            Wed += 1
    print(Wed_Prof/Wed)

# draw scatter plot between Chg% and Day col
#EXP A3 7)
def p_dailychange(data2):
    plt.scatter(data2["Day"], data2["Chg%"], alpha=0.6, color='blue', edgecolor='black')
    plt.title("Scatter Plot of Chg% vs Day of the Week")
    plt.xlabel("Day of the Week")
    plt.ylabel("Chg%")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# run everything for IRCTC data experiment
# IRCTC stock data
def main():
    data2 = irctc_data(r"C:/Users/Chandaluri Vaishnavi/Downloads/BLENU4CSE23267sem5ML/data/Lab Session Data.xlsx", 'IRCTC Stock Price')
    calc_basicstatistics(data2)
    mean_wed(data2, day="Wed")
    m_mp(data2, month="Apr")
    negativechange_mean(data2)
    wed_vs_others_positivechange(data2)
    wednesday_positive_chg_fraction(data2)
    p_dailychange(data2)

main()
