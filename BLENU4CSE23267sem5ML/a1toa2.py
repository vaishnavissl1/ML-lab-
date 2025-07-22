import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from scipy import stats
import statistics


# load purchase sheet and limit to 5 useful columns
# -- Purchase Data ----
def purchase_data01(pth, sht):
    dff = pd.read_excel(pth, sheet_name=sht)
    dff = dff.iloc[:, :5]
    dff.info()
    return dff

# return size of the dataset
#EXP A1 1,2)
def obtainshape(dff):
    return dff.shape

# check the rank of numeric part of matrix
#EXP A1 3)
def matrix_rank(dff):
    print(np.linalg.matrix_rank(dff.iloc[1:, 1:]))

# print header list, payments and item cols
#EXP A1 4)
def cd(dff):
    A = dff.columns.tolist()
    print(A)
    B = dff.iloc[:, 4]
    print(B)
    X = dff.iloc[:, :4]
    print(X)
    return A, B, X

# add new col that marks payment above 200 as rich
#EXP A2
def appendRichcoloumn(dff):
    dff["Rich"] = dff.iloc[:, 4].apply(lambda x: "Rich" if x > 200 else "POOR")
    print(dff)
    return dff

# run whole task for purchase data section
# Purchase data 
def main():
    print(" in main program ")
    
    dff = purchase_data01(r"C:/Users/Chandaluri Vaishnavi/Downloads/BLENU4CSE23267sem5ML/data/Lab Session Data.xlsx", 'Purchase data')
    A = dff[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    C = dff[['Payment (Rs)']].values

    cd(dff)

    dimensionality = obtainshape(dff)
    print(f"Dimensionality of the vector: {dimensionality}")

    rank = np.linalg.matrix_rank(dff[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].to_numpy())
    print(f"The rank of this matrix A is {rank}")

    X = np.linalg.pinv(A) @ C
    print(f"Sale Price {X}")

    R = appendRichcoloumn(dff)

main()
