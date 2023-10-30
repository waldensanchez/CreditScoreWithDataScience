import pandas as pd
import numpy as np

# Function to drop columns that are not useful
def dropUselessInfo(df):
    df.drop(columns=['Name', 'Age', 'SSN', 'Num_Bank_Accounts', 'Month', 'Occupation', 'Type_of_Loan', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Total_EMI_per_month', 'Credit_Mix', 'Interest_Rate', 'Changed_Credit_Limit', 'Annual_Income', 'Amount_invested_monthly'], inplace=True)
    return df

# Function to clean 'Monthly_Inhand_Salary' column
def cleanMIS(df):
    df['Monthly_Inhand_Salary'] = df.groupby('Customer_ID')['Monthly_Inhand_Salary'].apply(lambda x: x.ffill().bfill())
    return df

# Function to cap the 'Num_Credit_Card' values
def cleanNCC(df):
    df['Num_Credit_Card'] = df['Num_Credit_Card'].apply(lambda x: 11 if x >= 11 else x)
    return df

# Function to clean 'Num_of_Loan' column
def cleanNOF(df):
    df['Num_of_Loan'] = df['Num_of_Loan'].apply(lambda x: int(x.split('_')[0])).apply(lambda x: 0 if x < 0 else x).apply(lambda x: 10 if x > 10 else x)
    return df

# Function to clean 'Credit_History_Age' column
def cleanCHA(df):
    df['Credit_History_Age'] = df['Credit_History_Age'].fillna('NA')
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(lambda x: int(str(x).split(' ')[0]) if ' ' in str(x) and str(x) != 'NA' else np.nan)
    df['Credit_History_Age'] = df.groupby('Customer_ID')['Credit_History_Age'].apply(lambda x: x.ffill().bfill())
    return df

# Function to clean 'Num_of_Delayed_Payment' column
def safe_convert_to_float(x):
    try:
        return float(x.split('_')[0])
    except ValueError:
        return np.nan  # Return NaN if conversion fails

def cleanNoDP(df: pd.DataFrame) -> pd.DataFrame:
    # Replace 'NA' and other non-numeric strings with NaN to use fill methods
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].apply(lambda x: safe_convert_to_float(str(x)))
    
    # Apply ffill and bfill within groups to fill NaN values
    df['Num_of_Delayed_Payment'] = df.groupby('Customer_ID')['Num_of_Delayed_Payment'].apply(lambda x: x.ffill().bfill())
    
    # Set remaining NaNs to 0 or another sensible default value
    df['Num_of_Delayed_Payment'].fillna(0, inplace=True)
    
    # Ensure values are within the specified range (e.g., 0 to 27)
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].clip(lower=0, upper=27)
    
    return df


# Function to clean 'Outstanding_Debt' column
def cleanOD(df):
    df['Outstanding_Debt'] = df['Outstanding_Debt'].apply(lambda x: float(str(x).split('_')[0]))
    return df

# Function to clean 'Monthly_Balance' column
def cleanMB(df):
    df['Monthly_Balance'] = df['Monthly_Balance'].fillna('NA')
    df['Monthly_Balance'] = df['Monthly_Balance'].apply(lambda x: float(str(x).split('__')[1]) if '__' in str(x) and str(x)[0] == '_' else (float(x) if x != 'NA' else np.nan))
    df['Monthly_Balance'] = df.groupby('Customer_ID')['Monthly_Balance'].apply(lambda x: x.ffill().bfill())
    return df

# Function to cap 'Num_Credit_Inquiries' values
def cleanNCI(df):
    df['Num_Credit_Inquiries'] = df.groupby('Customer_ID')['Num_Credit_Inquiries'].apply(lambda x: x.ffill().bfill())
    df['Num_Credit_Inquiries'] = df['Num_Credit_Inquiries'].apply(lambda x: 18 if x > 18 else x)
    return df

# Function to map scores to numbers
def scoreToNum(df):
    df['Score'] = df['Credit_Score'].map({'Poor': 1, 'Standard': 2, 'Good': 3})
    return df

# Main cleaning function that applies all the cleaning steps
def clean(df):
    df = df.copy()  # Create a copy to avoid modifying the original dataframe
    df = df.set_index('ID')
    df = dropUselessInfo(df)
    df = cleanMIS(df)
    df = cleanNCC(df)
    df = cleanNOF(df)
    df = cleanCHA(df)
    df = cleanNoDP(df)
    df = cleanOD(df)
    df = cleanMB(df)
    df = scoreToNum(df)
    df = cleanNCI(df)
    df = df.drop(['Customer_ID','Credit_Score'], axis=1).reset_index(drop=True)
    df['Score'] = df['Score'].map({3:2, 2:1, 1:0})
    return df

# Sample usage:
# Load your data
# data = pd.read_csv('train.csv')
# Clean your data
# cleaned_data = clean(data)
