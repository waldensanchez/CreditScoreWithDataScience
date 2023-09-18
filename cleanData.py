import pandas as pd
import numpy as np

def cleanMIS(df: pd.DataFrame):
    df['Monthly_Inhand_Salary']= df.groupby('Customer_ID')['Monthly_Inhand_Salary'].ffill()
    return df

def cleanAI(df: pd.DataFrame):
    df['Annual_Income'] = df['Annual_Income'].apply(lambda x: float(x.split('_')[0]))
    return df

def dp_try(x):
    try:
        return int(str(x).split('_')[0])
    except:
        return x

def cleanDP(df: pd.DataFrame):
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].apply(dp_try)
    return df

def cleanOD(df: pd.DataFrame):
    df['Outstanding_Debt'] = df['Outstanding_Debt'].apply(lambda x: float(str(x).split('_')[0]))
    return df

def dropUselessInfo(df: pd.DataFrame):
    df = df.drop([
        'Name', 'SSN', 'Num_Bank_Accounts', 'Month','Occupation','Type_of_Loan','Payment_of_Min_Amount','Payment_Behaviour',
        'Total_EMI_per_month','Credit_Mix','Interest_Rate','Changed_Credit_Limit','Annual_Income','Amount_invested_monthly'
        ], axis = 1)
    return df

def cleanAge(df:pd.DataFrame):
    df['Age'] = df['Age'].apply(lambda x: int(x.split('_')[0])).apply(lambda x: np.nan if x < 18 else x)
    df = df.reset_index().groupby('Customer_ID')[['ID', 'Age']].ffill().set_index('ID')
    return df

def cleanNCC(df: pd.DataFrame):
    df['Num_Credit_Card'] = df['Num_Credit_Card'].apply(lambda x: 11 if x >= 11 else x)
    return df

def cleanNOF(df: pd.DataFrame):
    df['Num_of_Loan'] = df['Num_of_Loan'].apply(lambda x: int(x.split('_')[0])).apply(lambda x: 0 if x < 0 else x)
    return df

def clean_double_hyphen(x):
    try:
        return float(x)
    except:
        return float(x.split('__')[1])

def cleanMB(df: pd.DataFrame):
    df["Monthly_Balance"] = df['Monthly_Balance'].apply(clean_double_hyphen)
    return df

def clean_date(x):
    try:
        divided = x.split(" ")
        return int(divided[0])
    except:
        return x

def cleanCHA(df: pd.DataFrame):
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(clean_date)
    df['Credit_History_Age'] = df.reset_index().groupby(['Customer_ID'])[['ID','Credit_History_Age']].ffill().set_index('ID')
    return df

def dropCI(df: pd.DataFrame):
    df = df.drop('Customer_ID', axis = 1)
    return df

score_df = pd.DataFrame([1,2,3],index = ['Poor','Standard','Good'], columns = ['Score'])

def y_value(x: str, score_df: pd.DataFrame = score_df):
    return score_df.loc[x]['Score']

def to_numerical_var(df: pd.DataFrame):
    df['Credit_Score'] = df['Credit_Score'].apply(y_value)
    return df

def handle_missing_values(df: pd.DataFrame):
    return df.fillna(0)

def final_cleansing(df: pd.DataFrame):
    df = df.reset_index().drop(['ID'], axis = 1)
    return df

def clean(df: pd.DataFrame):
    df = df.set_index('ID')
    df = dropUselessInfo(df)
    df = cleanMIS(df)
    df['Age'] = cleanAge(df)
    df = cleanNCC(df)
    df = cleanNOF(df)
    df = cleanCHA(df)
 #   df = cleanAI(df)
    df = cleanDP(df)
    df = cleanOD(df)
    df = cleanMB(df)
    df = dropCI(df)
    df = handle_missing_values(df)
    df = final_cleansing(df)
    return df

def test(df: pd.DataFrame):
    l = []
    for i in range(len(df)):
        if df['Prediction'].iloc[i] == df['Real'].iloc[i]:
            value = 1
        else:
            value = 0
        l.append(value)
    return print(f'Accuracy: {round(np.array(l).sum()/len(df)*100,1)}%')

def round_rules(x):
    x = round(x,0)
    if x <= 1:
        return 1
    elif x >= 3:
        return 3
    else:
        return 2