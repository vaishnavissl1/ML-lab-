import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from scipy import stats
import statistics

# reads thyroid data from given excel sheet
# -- Thyroid Data ----
#EXP A4
def thyroid_data(pth, sht):
    d3 = pd.read_excel(pth, sheet_name=sht)
    print(d3.info())
    return d3

# encode ordinal and one-hot features from thyroid data
def encode_tdata(d3, ordinal_c, onehot_c):
    encode = OrdinalEncoder()
    data_ord = encode.fit_transform(d3[ordinal_c])
    data__ord_df = pd.DataFrame(data_ord, columns=ordinal_c)
    print(data__ord_df.head())

    oneh = OneHotEncoder(sparse_output=False, drop='first')
    data__onehot = oneh.fit_transform(d3[onehot_c])
    col_names = oneh.get_feature_names_out(onehot_c)
    enc_data = pd.DataFrame(data__onehot, columns=col_names)
    print(enc_data.head())
    return data__ord_df, enc_data

# checks ranges and missing for numeric columns
def analyzenumcol(d3, num_c):
    for x in num_c:
        d3[x] = pd.to_numeric(d3[x], errors='coerce')
        min_num = d3[x].min()
        max_num = d3[x].max()
        print(f"column {x} range is {min_num} to {max_num}")
        missingvals = d3[x].isna().sum()
        print(f"column {x} has {missingvals} missing values")
    return d3

# plots boxplot for all numeric features
def boxplots(df):
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        plt.figure(figsize=(4, 6))
        sns.boxplot(y=df[col].dropna())
        plt.title(f"Boxplot for {col}")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

# makes a boxplot for 'age' without outliers
def age_boxplot_no_outliers(df):
    age = df['age'].dropna()
    Q1 = age.quantile(0.25)
    Q3 = age.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    age_no_outliers = age[(age >= lower_bound) & (age <= upper_bound)]

    plt.figure(figsize=(4, 7))
    sns.boxplot(y=age_no_outliers)
    plt.title("Boxplot for Age (Outliers Removed)")
    plt.ylabel("age")
    plt.tight_layout()
    plt.show()

# calculates jc and smc between first two enc rows
#EXP A5
def compute_jaccard_and_smc(enc_data):
    obs11 = enc_data.iloc[0].astype(int).tolist()
    obs22 = enc_data.iloc[1].astype(int).tolist()
    f11 = sum(a == b == 1 for a, b in zip(obs11, obs22))
    f00 = sum(a == b == 0 for a, b in zip(obs11, obs22))
    f10 = sum(a == 1 and b == 0 for a, b in zip(obs11, obs22))
    f01 = sum(a == 0 and b == 1 for a, b in zip(obs11, obs22))
    den_jc = f11 + f10 + f01
    jc = f11 / den_jc if den_jc > 0 else 0.0
    den_smc = f11 + f10 + f01 + f00
    smc = (f11 + f00) / den_smc if den_smc > 0 else 0.0
    print("Jaccard Coefficient", jc)
    print("Simple Matching Coefficient basically SMC", smc)

# gets cosine sim of first two encoded obs
#EXP A6
def cosine_similarity(enc_data1):
    vec1 = enc_data1.iloc[0].astype(int).to_numpy()
    vec2 = enc_data1.iloc[1].astype(int).to_numpy()
    cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    print("Cosine Similarity", cos_sim)

# makes 3 similarity heatmaps for first 20 encoded obs
def prwsimilarity_heatmaps(enc_data1):
    data_20 = enc_data1.iloc[:20].astype(int).to_numpy()
    n = data_20.shape[0]
    jc_matrix = np.zeros((n, n))
    smccc = np.zeros((n, n))
    cos_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = data_20[i], data_20[j]
            f11 = np.sum((a == 1) & (b == 1))
            f00 = np.sum((a == 0) & (b == 0))
            f10 = np.sum((a == 1) & (b == 0))
            f01 = np.sum((a == 0) & (b == 1))
            denom_jc = f11 + f10 + f01
            jc_matrix[i, j] = f11 / denom_jc if denom_jc > 0 else 0.0
            t = f11 + f10 + f01 + f00
            smccc[i, j] = (f11 + f00) / t if t > 0 else 0.0
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cos_matrix[i, j] = cos_sim

    #EXP A7
    def heatmap(m, t):
        plt.figure(figsize=(10, 8))
        sns.heatmap(m, annot=False, cmap='viridis')
        plt.title(t)
        plt.xlabel("Observation ")
        plt.ylabel("Observation ")
        plt.show()

    heatmap(jc_matrix, "Jaccard Coefficient Heatmap that is First 20 Observations")
    heatmap(smccc, "Simple Matching Coefficient Heatmap")
    heatmap(cos_matrix, "Cosine Similarity Heatmap")

# fills missing values using mean, median, or mode
#EXP A8
def impute_missingvalues(d3):
    d_imputed = d3.copy()
    for col in d_imputed.columns:
        if d_imputed[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(d_imputed[col]):
            cdata = d_imputed[col].dropna()
            zscr = np.abs(stats.zscore(cdata))
            outliers = (zscr > 3).any()
            iv = cdata.median() if outliers else cdata.mean()
            d_imputed[col] = d_imputed[col].fillna(iv)
        else:
            iv = d_imputed[col].mode()[0]
            d_imputed[col] = d_imputed[col].fillna(iv)
    print(d_imputed.head())
    return d_imputed

# normalizes selected numeric cols with scaling
#EXP A9 
def normalize_numcols(df, creplace, ustandard=True):
    df[creplace] = df[creplace].replace('?', pd.NA)
    for col in creplace:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df_cleaned = df.copy()
    df_cleaned[creplace] = df_cleaned[creplace].astype(float)
    for col in creplace:
        if df_cleaned[col].isna().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    scaler = StandardScaler() if ustandard else MinMaxScaler()
    df_scaled = df_cleaned.copy()
    df_scaled[creplace] = scaler.fit_transform(df_cleaned[creplace])
    print(df_scaled[creplace].head())
    return df_scaled

# runs all parts for thyroid data work
# Thyroid data
def main():
    d3 = thyroid_data(r"C:/Users/Chandaluri Vaishnavi/Downloads/BLENU4CSE23267sem5ML/data/Lab Session Data.xlsx", 'thyroid0387_UCI')
    
    oc = ["referral source", "Condition"]
    onehot_cls = ["sex", "on thyroxine", "query on thyroxine", "on antithyroid medication", "sick",
                   "pregnant", "thyroid surgery", "I131 treatment", "query hypothyroid", "query hyperthyroid",
                   "lithium", "goitre", "tumor", "hypopituitary", "psych", "TSH measured", "T3 measured",
                   "TT4 measured", "FTI measured", "TBG measured"]
    nc = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]

    data__ord_df, enc_d1 = encode_tdata(d3, oc, onehot_cls)
    analyzenumcol(d3, nc)
    boxplots(d3)
    age_boxplot_no_outliers(d3)
    compute_jaccard_and_smc(enc_d1)
    cosine_similarity(enc_d1)
    prwsimilarity_heatmaps(enc_d1)
    impute_missingvalues(d3)

    creplacee = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age']
    normalize_numcols(
        pd.read_excel(r"C:/Users/Chandaluri Vaishnavi/Downloads/BLENU4CSE23267sem5ML/data/Lab Session Data.xlsx", sheet_name="thyroid0387_UCI"),
        creplacee,
        ustandard=True
    )

main()
