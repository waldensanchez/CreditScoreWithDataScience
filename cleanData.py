import pandas as pd
import numpy as np

def cleanMIS(df: pd.DataFrame):
    df['Monthly_Inhand_Salary']= df.groupby('Custumer_ID')[['Monthly_Inhand_Salary']].ffill().set_index('ID')
    return df

def cleanAI(df: pd.DataFrame):
    df['Annual_Income'] = df['Annual_Income'].apply(lambda x: float(x.split('_')[0]))
    return df

def dropUselessInfo(df: pd.DataFrame):
    df.drop(['Name', 'SSN', 'Num_Bank_Accounts', 'Month',])
    return df

def cleanAge(df:pd.DataFrame):
    df['Age'] = df['Age'].apply(lambda x: int(x.split('_')[0])).apply(lambda x: np.nan if x < 18 else x)
    df['Age'] = df['Age'].reset_index().groupby('Customer_ID')[['ID', 'Age']].ffill().set_index('ID')
    return df

def cleanNCC(df: pd.DataFrame):
    df['Num_Credit_Card'] = df['Num_Credit_Card'].apply(lambda x: 11 if x >= 11 else x)
    return df

def cleanNOF(df: pd.DataFrame):
    df['Num_of_Loan'] = df['Num_of_Loan'].apply(lambda x: int(x.split('_')[0])).apply(lambda x: 0 if x < 0 else x)
    return df

def cleanCHA(df: pd.DataFrame):
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(lambda x: int(x.split(' ')[0]) if x != 'NA' else x)
    df['Credit_History_Age'] = df['Credit_History_Age'].reset_index().groupby(['Customer_ID'])[['ID', 'Credit_History_Age']].ffill().set_index('ID')
    return df

def clean(df: pd.DataFrame):
    df = df.set_index('ID')
    dropUselessInfo(df)
    cleanMIS(df)
    cleanAI(df)
    cleanAge(df)
    cleanNCC(df)
    cleanNOF(df)
    cleanCHA(df)