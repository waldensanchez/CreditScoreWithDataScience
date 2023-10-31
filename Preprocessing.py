
import pandas as pd
import numpy as np
from scipy.stats import boxcox

class Preprocessing:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        return self.data

    @staticmethod
    def apply_transformation(column):
        # Handle NaN values by filling them with the median of the column
        column = column.fillna(column.median())
        # Ensure all values are positive (adding a small constant to zero values)
        column += (column <= 0) * (abs(column.min()) + 1e-6)
        transformed, _ = boxcox(column)
        return transformed

    @staticmethod
    def cap_extreme_values(df, column):
        # Calculate Q1 and Q3
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        # Calculate the IQR
        IQR = Q3 - Q1
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Cap the values
        df[column] = np.where(df[column] > upper_bound, upper_bound, 
                              np.where(df[column] < lower_bound, lower_bound, df[column]))
        return df

    def transform_payment_behaviour(self):
        # Split 'Payment_Behaviour' into 'Spent' and 'Payment_Size'
        self.data['Spent'] = self.data['Payment_Behaviour'].str.extract(r'(High_spent|Low_spent)')[0]
        self.data['Payment_Size'] = self.data['Payment_Behaviour'].str.extract(r'(Small_value_payments|High_value_payments|Large_value_payments)')[0]
        
        # Map the extracted strings to categorical values
        spent_mapping = {'Low_spent': 0, 'High_spent': 1}
        payment_size_mapping = {'Small_value_payments': 1, 'High_value_payments': 2, 'Large_value_payments': 3}
        self.data['Spent'] = self.data['Spent'].map(spent_mapping)
        self.data['Payment_Size'] = self.data['Payment_Size'].map(payment_size_mapping)
        
        # Handle errors by grouping by ID and filling values if possible, otherwise allocate 0
        self.data['Spent'] = self.data.groupby('Customer_ID')['Spent'].transform(lambda x: x.fillna(method='bfill').fillna(0))
        self.data['Payment_Size'] = self.data.groupby('Customer_ID')['Payment_Size'].transform(lambda x: x.fillna(method='bfill').fillna(0))

        # Drop the original 'Payment_Behaviour' column
        self.data.drop('Payment_Behaviour', axis=1, inplace=True)

    def preprocess_data(self):
        # Apply the transformations for 'Payment_Behaviour'
        self.transform_payment_behaviour()

        # Other preprocessing steps would go here...

        return self.data

    def save_preprocessed_data(self, output_file_path):
        self.data.to_csv(output_file_path, index=False)

