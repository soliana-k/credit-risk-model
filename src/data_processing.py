import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EDA:
    def __init__(self, df:pd.DataFrame):
        self.df=df
        self.results = {}

    def data_overview(self):
        # logger.info('starting the overview of the dataset')
        # row, cols=self.df.shape

        # print(self.df.info())
        # print(f'The number of Rows is: {row}')
        # print(f'The number of Columns is: {cols}')
        # print(f'The datatypes of the Columns:\n {self.df.dtypes}')
        logger.info('Gathering overview stats')
        # Print for the user to see immediately
        print(f'Rows: {self.df.shape[0]}')
        print(f'Cols: {self.df.shape[1]}')
        
        # Return for the orchestrator to store in self.results
        return {
            "rows": self.df.shape[0],
            "cols": self.df.shape[1],
            "dtypes": self.df.dtypes.to_dict()
        }

    def data_summary(self):
        pass

    def numerical_cols_distribution(self):
        pass


    def categroical_cols_distribution(self):
        pass

    def correlation_analysis(self):
        pass

    def indentify_missing_vals(self):
        pass

    def outlier_detection(self):
        pass

    def run(self):
        self.results['overview']=self.data_overview()
        logger.info("EDA process complete.")
        return self.results
    
    def summary_report(self):
        """A professional, non-tacky report generator."""
        print("\n=== DATA OVERVIEW ===")
        # Instead of a for-loop, convert your results to a series
        # and let pandas print it. It aligns perfectly automatically.
        overview = self.results.get('overview', {})
        print(f"Dimensions: {overview['rows']} rows x {overview['cols']} columns")
        
        # This renders as a clean table
        dtypes_df = pd.Series(overview['dtypes'], name="Data Type")
        print("\nData Types:")
        print(dtypes_df)

