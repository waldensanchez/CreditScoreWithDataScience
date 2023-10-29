import numpy as np
import pandas as pd


def monthly_salary_bracket(x):
    brackets = lambda n: {-np.inf<n<1822: 50, 1822<=n<6711: 20, 6711<=n<np.inf: 0}
    try:
        return brackets(x)[0]
    except:
        return brackets(x)['True']

def credit_card_bracket(x):
    brackets = lambda n: {-np.inf<n<=1: 30, 1<n<4: 0, 4<=n<7: 20, 7<=n<np.inf: 50}
    try:
        return brackets(x)[0]
    except:
        return brackets(x)['True']

def num_loan_bracket(x):
    brackets = lambda n: {-np.inf<n<2: 0, 2<=n<=4: 20, 4<n<np.inf: 50}
    try:
        return brackets(x)[0]
    except:
        return brackets(x)['True']
    
def delay_bracket(x):
    brackets = lambda n: {-np.inf<n<11: 0, 11<=n<30: 20, 30<=n<np.inf: 50}
    try:
        return brackets(x)[0]
    except:
        return brackets(x)['True']

def num_delay_bracket(x):
    brackets = lambda n: {-np.inf<n<8: 0, 8<=n<17: 20, 17<=n<np.inf: 50}
    try:
        return brackets(x)[0]
    except:
        return brackets(x)['True']

def credit_inqueries_bracket(x):
    brackets = lambda n: {-np.inf<n<3: 0, 3<=n<10: 20, 10<=n<np.inf: 50}
    try:
        return brackets(x)[0]
    except:
        return brackets(x)['True']

def debt_bracket(x):
    brackets = lambda n: {-np.inf<n<1000: 0, 1000<=n<2500: 1, 2500<=n<np.inf: 2}
    try:
        return brackets(x)[0]
    except:
        return brackets(x)['True']

def credit_utilization_bracket(x):
    brackets = lambda n: {-np.inf<n<28: 0, 28<=n<36: 20, 36<=n<np.inf: 50}
    return brackets(x)[1]

def credit_history_bracket(x):   
    brackets = lambda n: {-np.inf<n<13: 50, 13<=n<26: 20, 26<=n<np.inf: 0}
    try:
        return brackets(x)[0]
    except:
        return brackets(x)['True']

def monthly_balance_bracket(x):
    brackets = lambda n: {-np.inf<n<4: 0, 4<=n<7: 20, 7<=n<np.inf: 50}
    try:
        return brackets(x)[0]
    except:
        return brackets(x)['True']

def score_predicted_in_bracket(x):
    if x < 708:
        return 1
    elif 708 <= x < 798:
        return 2
    else:
        return 3

def setScore(df: pd.DataFrame) -> pd.DataFrame:
    df['MIS_penalty'] = df['Monthly_Inhand_Salary'].apply(monthly_salary_bracket)
    df['NCC_penalty'] = df['Num_Credit_Card'].apply(credit_card_bracket)
    df['NoL_penalty'] = df['Num_of_Loan'].apply(num_loan_bracket)
    df['Delay_penalty'] = df['Delay_from_due_date'].apply(delay_bracket)
    df['NDP_penalty'] = df['Num_of_Delayed_Payment'].apply(num_delay_bracket)
    df['NCI_penalty'] = df['Num_Credit_Inquiries'].apply(credit_inqueries_bracket)
    df['Debt_penalty'] = df['Outstanding_Debt'].apply(debt_bracket)
    df['CUR_penalty'] = df['Credit_Utilization_Ratio'].apply(credit_utilization_bracket)
    df['CHA_penalty'] = df['Credit_History_Age'].apply(credit_history_bracket)
    df['MB_penalty'] = df['Monthly_Balance'].apply(monthly_balance_bracket)
    df['Total_penalty'] = df[['MIS_penalty', 'NCC_penalty', 'NoL_penalty', 'Delay_penalty', 'NDP_penalty', 'NCI_penalty', 'Debt_penalty', 'CUR_penalty', 'CHA_penalty', 'MB_penalty']].sum(axis=1)
    df['Score_Predicted_in_points'] = 1000-df['Total_penalty']
    df['Score_Predicted_in_bracket'] = df['Score_Predicted_in_points'].apply(score_predicted_in_bracket)
    return df