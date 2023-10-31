
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

    def preprocess_data(self):
        # Apply transformations to columns with high skewness (arbitrarily chosen as skew > 1)
        numerical_skewness = self.data.select_dtypes(include=['float64', 'int64']).skew().sort_values(ascending=False)
        for col in numerical_skewness.index:
            if col != 'Score':  # Exclude 'Score' from transformations
                if numerical_skewness[col] > 1:
                    self.data[col] = self.apply_transformation(self.data[col])

        # Apply capping to 'Num_Credit_Inquiries' and 'Monthly_Inhand_Salary'
        self.data = self.cap_extreme_values(self.data, 'Num_Credit_Inquiries')
        self.data = self.cap_extreme_values(self.data, 'Monthly_Inhand_Salary')

        return self.data

    def save_preprocessed_data(self, output_file_path):
        self.data.to_csv(output_file_path, index=False)

