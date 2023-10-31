
import pandas as pd
import matplotlib.pyplot as plt

class DQR:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.dqr_report = None

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        return self.data

    @staticmethod
    def clean_numeric_object_columns(column):
        # Remove any non-numeric characters, convert to float, and coerce errors to NaN
        column = pd.to_numeric(column.astype(str).str.replace('[^\d\.-]', '', regex=True), errors='coerce')
        # Replace negative values with NaN
        column[column < 0] = None
        # Fill NaN values with the median
        column.fillna(column.median(), inplace=True)
        return column

    def clean_data(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please load the data first using load_data() method.")
        # Apply the cleaning function to 'Annual_Income' and 'Outstanding_Debt' columns
        self.data['Annual_Income'] = self.clean_numeric_object_columns(self.data['Annual_Income'])
        self.data['Outstanding_Debt'] = self.clean_numeric_object_columns(self.data['Outstanding_Debt'])

        # Since 'Monthly_Inhand_Salary' is already numeric, we just fill missing values
        self.data['Monthly_Inhand_Salary'].fillna(self.data['Monthly_Inhand_Salary'].median(), inplace=True)
        return self.data

    def dqr(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please load the data first using load_data() method.")
        # Generate descriptive statistics
        descriptive_stats = self.data.describe(include='all')

        # Count of unique values
        unique_values_count = self.data.nunique()

        # Count of missing values
        missing_values_count = self.data.isnull().sum()

        # Combining the information into a DataFrame for the DQR
        self.dqr_report = pd.DataFrame({
            'Data Type': self.data.dtypes,
            'Total Values': len(self.data),
            'Unique Values': unique_values_count,
            'Missing Values': missing_values_count,
            'Mean': descriptive_stats.loc['mean'],
            'Median': descriptive_stats.loc['50%'],
            'Std': descriptive_stats.loc['std'],
            'Min': descriptive_stats.loc['min'],
            'Max': descriptive_stats.loc['max']
        })

        # Reordering the columns to match typical DQR reports
        self.dqr_report = self.dqr_report[['Data Type', 'Total Values', 'Unique Values', 'Missing Values', 'Mean', 'Median', 'Std', 'Min', 'Max']]
        self.dqr_report.reset_index(inplace=True)
        self.dqr_report.rename(columns={'index': 'Column Name'}, inplace=True)
        
        return self.dqr_report

    def graphs(self):
        if self.dqr_report is None:
            raise ValueError("DQR report not generated. Please run the dqr() method first.")
        
        # Plot missing values
        plt.figure(figsize=(10, 5))
        plt.bar(self.dqr_report['Column Name'], self.dqr_report['Missing Values'])
        plt.xlabel('Columns')
        plt.ylabel('Number of Missing Values')
        plt.title('Missing Values per Column')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        # Plot unique values
        plt.figure(figsize=(10, 5))
        plt.bar(self.dqr_report['Column Name'], self.dqr_report['Unique Values'])
        plt.xlabel('Columns')
        plt.ylabel('Number of Unique Values')
        plt.title('Unique Values per Column')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def metrics(self):
        if self.dqr_report is None:
            raise ValueError("DQR report not generated. Please run the dqr() method first.")
        return self.dqr_report[['Column Name', 'Missing Values', 'Unique Values', 'Mean', 'Median', 'Std', 'Min', 'Max']]

    def save_to_csv(self, output_file_path):
        if self.dqr_report is None:
            raise ValueError("DQR report not generated. Please run the dqr() method first.")
        self.dqr_report.to_csv(output_file_path, index=False)
