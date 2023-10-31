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

    def map_month_to_number(self):
        # Map the month names to numbers
        month_mapping = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8
        }
        self.data['Month'] = self.data['Month'].map(month_mapping)

    def transform_credit_history_age(self):
        # Transform 'Credit_History_Age' to numerical format in years
        def age_to_years(age_str):
            if pd.isnull(age_str):
                return np.nan
            parts = age_str.split(' ')
            years = int(parts[0]) if parts[0].isdigit() else 0
            months = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            return years + months / 12

        self.data['Credit_History_Age'] = self.data['Credit_History_Age'].apply(age_to_years)
        
        # Adjust 'Credit_History_Age' based on the 'Month' column progression for the same customer
        self.map_month_to_number()
        self.data.sort_values(by=['Customer_ID', 'Month'], inplace=True)
        self.data['Credit_History_Age'] = self.data.groupby('Customer_ID')['Credit_History_Age'].ffill() + self.data.groupby('Customer_ID').cumcount() / 12

    def transform_payment_behavior(self):
        # Define mappings for 'Spent' and 'Payment_size'
        spent_mapping = {
            'Low_spent_': 0,
            'High_spent_': 1
        }
        payment_size_mapping = {
            'Small_value_payments': 1,
            'Medium_value_payments': 2,
            'Large_value_payments': 3
        }
        
        # Function to split 'Payment_Behaviour' and map to new categorical variables
        def split_payment_behavior(behavior):
            # If the behavior is a special case or NaN, return NaN to handle later
            if pd.isnull(behavior) or behavior == '!@9#%8':
                return np.nan, np.nan
            # Split the behavior into 'spent' and 'payment_size' components
            parts = behavior.split('_value_')
            if len(parts) == 2:
                spent, payment_size = parts
                spent_value = spent_mapping.get(spent, 0)
                payment_size_value = payment_size_mapping.get(payment_size, 0)
                return spent_value, payment_size_value
            return 0, 0  # Default case if the format is not correct

        # Apply the function and create new columns
        self.data[['Spent', 'Payment_size']] = self.data.apply(
            lambda x: split_payment_behavior(x['Payment_Behaviour']), axis=1, result_type="expand"
        )

        # Fill in the special case '!@9#%8' by looking for another value for the same ID
        special_case_mask = self.data['Payment_Behaviour'] == '!@9#%8'
        for customer_id in self.data.loc[special_case_mask, 'Customer_ID'].unique():
            # Find an entry for the customer that does not have the special case
            alternative_entry = self.data[
                (self.data['Customer_ID'] == customer_id) & (~special_case_mask)
            ]['Payment_Behaviour'].first_valid_index()
            if alternative_entry:
                # If an alternative entry exists, use its 'Spent' and 'Payment_size'
                alt_spent, alt_payment_size = split_payment_behavior(self.data.at[alternative_entry, 'Payment_Behaviour'])
                self.data.loc[self.data['Customer_ID'] == customer_id, 'Spent'] = self.data.loc[
                    self.data['Customer_ID'] == customer_id, 'Spent'
                ].fillna(alt_spent)
                self.data.loc[self.data['Customer_ID'] == customer_id, 'Payment_size'] = self.data.loc[
                    self.data['Customer_ID'] == customer_id, 'Payment_size'
                ].fillna(alt_payment_size)
            else:
                # If no alternative entry exists, default to 0
                self.data.loc[self.data['Customer_ID'] == customer_id, ['Spent', 'Payment_size']] = 0

        # Drop the original 'Payment_Behaviour' column
        self.data.drop('Payment_Behaviour', axis=1, inplace=True)

    def map_categorical_to_numerical(self):
        # Map 'Credit_Mix' and 'Payment_of_Min_Amount' to numerical values
        credit_mix_mapping = {
            'Standard': 0,
            'Good': 1,
            'Bad': 2
        }
        payment_of_min_amount_mapping = {
            'No': 0,
            'Yes': 1
        }
        self.data['Credit_Mix'] = self.data['Credit_Mix'].map(credit_mix_mapping).fillna(0)
        self.data['Payment_of_Min_Amount'] = self.data['Payment_of_Min_Amount'].map(payment_of_min_amount_mapping).fillna(0)

    def preprocess_data(self):
        # Transform 'Credit_History_Age' to numerical format and adjust based on 'Month' progression
        self.transform_credit_history_age()
        # Transform 'Payment_Behaviour' into two new categorical variables
        self.transform_payment_behavior()
        # Map 'Credit_Mix' and 'Payment_of_Min_Amount' to numerical values
        self.map_categorical_to_numerical()
        # Other preprocessing steps...

        return self.data

    # Other methods of the class...