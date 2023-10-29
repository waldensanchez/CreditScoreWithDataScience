import pandas as pd
import numpy as np

def dropUselessInfo(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns = ['Name', 'Age', 'SSN', 'Num_Bank_Accounts', 'Month', 'Occupation', 'Type_of_Loan', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Total_EMI_per_month', 'Credit_Mix', 'Interest_Rate', 'Changed_Credit_Limit', 'Annual_Income', 'Amount_invested_monthly'], inplace=True)
    return df

def cleanMIS(df: pd.DataFrame) -> pd.DataFrame:
    df['Monthly_Inhand_Salary']= df.groupby('Customer_ID')[['Monthly_Inhand_Salary']].ffill().set_index(df.index)
    return df

def cleanNCC(df: pd.DataFrame) -> pd.DataFrame:
    df['Num_Credit_Card'] = df['Num_Credit_Card'].apply(lambda x: 11 if x >= 11 else x)
    return df

def cleanNOF(df: pd.DataFrame):
    df['Num_of_Loan'] = df['Num_of_Loan'].apply(lambda x: int(x.split('_')[0])).apply(lambda x: 0 if x < 0 else x).apply(lambda x: 10 if x > 10 else x)
    return df

def cleanCHA(df: pd.DataFrame) -> pd.DataFrame:
    df['Credit_History_Age'] = df['Credit_History_Age'].fillna('NA')
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(lambda x: int(str(x).split(' ')[0]) if str(x) != 'NA' else np.nan)
    df['Credit_History_Age'] = df.groupby(['Customer_ID'])[['Credit_History_Age']].ffill().set_index(df.index)
    return df

def cleanNoDP(df: pd.DataFrame) -> pd.DataFrame:
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].fillna('NA')
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].apply(lambda x: int(str(x).split('_')[0]) if str(x) != 'NA' else np.nan).apply(lambda x: 0 if x < 0 else x).apply(lambda x: 27 if x > 27 else x)
    df['Num_of_Delayed_Payment'] = df.groupby('Customer_ID')[['Num_of_Delayed_Payment']].ffill().set_index(df.index)
    return df

def cleanOD(df: pd.DataFrame) -> pd.DataFrame:
    df['Outstanding_Debt'] = df['Outstanding_Debt'].apply(lambda x: float(str(x).split('_')[0]))
    return df

def cleanMB(df: pd.DataFrame) -> pd.DataFrame:
    df['Monthly_Balance'] = df['Monthly_Balance'].fillna('NA')
    df['Monthly_Balance'] = df['Monthly_Balance'].apply(lambda x: float(str(x).split('__')[1]) if str(x)[0] == '_' else str(x))
    df['Monthly_Balance'] = df['Monthly_Balance'].apply(lambda x: float(x) if x != 'NA' else np.nan)
    df['Monthly_Balance'] = df['Monthly_Balance'].apply(lambda x: 0 if x < 0 else x)
    df['Monthly_Balance'] = df.groupby(['Customer_ID'])[['Monthly_Balance']].ffill().set_index(df.index)
    return df

def cleanNCI(df: pd.DataFrame) -> pd.DataFrame:
    df['Num_Credit_Inquiries'] = df.groupby(['Customer_ID'])[['Num_Credit_Inquiries']].ffill().set_index(df.index)
    df['Num_Credit_Inquiries'] = df['Num_Credit_Inquiries'].apply(lambda x: 18 if x > 18 else x)
    return df

def scoreToNum(df: pd.DataFrame) -> pd.DataFrame:
    df['Score'] = df['Credit_Score'].map({'Poor': 1, 'Standard': 2, 'Good': 3})
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index('ID')
    dropUselessInfo(df)
    cleanMIS(df)
    cleanNCC(df)
    cleanNOF(df)
    cleanCHA(df)
    cleanNoDP(df)
    cleanOD(df)
    cleanMB(df)
    scoreToNum(df)
    cleanNCI(df)
    df = df.drop('Customer_ID', axis=1)
    return df