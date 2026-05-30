import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EDA:

    SKEW_THRESHOLD = 1.0  
    KURT_THRESHOLD = 3.0

    def __init__(self, df:pd.DataFrame):
        self.df=df
        self.results = {}


    def _get_distribution(self, skew_val, kurt_val):
        """Helper cunftion to automate the textual explanation of distribution shape."""
        insight = []
        if abs(skew_val) > self.SKEW_THRESHOLD:
            direction = "right" if skew_val > 0 else "left"
            insight.append(f"Highly skewed to the {direction} (Skew: {skew_val:.2f}).")
        else:
            insight.append("Approximately symmetric.")
            
        if kurt_val > self.KURT_THRESHOLD:
            insight.append(f"Leptokurtic/Heavy-tailed (Kurtosis: {kurt_val:.2f}).")
        elif kurt_val < -1.0:
            insight.append(f"Platykurtic/Flat-tailed (Kurtosis: {kurt_val:.2f}).")
            
        return " ".join(insight)
    
    def _get_missing_advice(self, pct):
        """Helper function for automated advice for missing data."""
        if pct == 0:
            return "No action needed."
        elif pct < 5:
            return "Impute (Mean/Median/Mode)."
        elif pct < 30:
            return "Consider advanced imputation or dropping."
        else:
            return "High missing rate: Consider dropping column or deep investigation."


    def data_overview(self):
        logger.info('Gathering overview stats')
        print(f'Rows: {self.df.shape[0]}')
        print(f'Cols: {self.df.shape[1]}')
        return {
            "rows": self.df.shape[0],
            "cols": self.df.shape[1],
            "dtypes": self.df.dtypes.to_dict()
        }

    def data_summary(self):
        logger.info('Calculating statistical summary with automated insights...')
        num_df = self.df.select_dtypes(include=['number'])
        summary = pd.DataFrame({
            'Average': num_df.mean(),
            'Middle Value': num_df.median(),
            'Standard Deviation': num_df.std(),
            'Skewness': num_df.skew(),
            'kurtosis': num_df.kurt()
        })
        summary['Insight'] = summary.apply(
            lambda row: self._get_distribution(row['Skewness'], row['kurtosis']), 
            axis=1
        )
        
        return {
            'numeric_stats': summary,
            'categorical_modes': self.df.mode(numeric_only=False).iloc[0]
        }
        

    def numerical_cols_distribution(self):
        logger.info('--- Generating Distribution Visualizations ---')
        num_cols = self.df.select_dtypes(include=['number']).columns

        for column in num_cols:
            unique_count = self.df[column].nunique()
            plt.figure(figsize=(8, 4))
            if unique_count <= 2:
                sns.countplot(x=self.df[column])
                plt.title(f'Count Plot of {column} (Binary)')
            else:
                sns.histplot(self.df[column], kde=True, bins=40)
                plt.title(f'Distribution of {column} (Continuous)')
            
            plt.tight_layout()
            plt.show()
            plt.close()
        


    def categroical_cols_distribution(self):
        logger.info("--- Rendering categorical features ---")
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for column in cat_cols:
            unique_count = self.df[column].nunique()
            plt.figure(figsize=(10, 4))
            if unique_count > 25:
                order = self.df[column].value_counts().iloc[:10].index
                sns.countplot(data=self.df, y=column, order=order)
                plt.title(f'Top 10 Values for {column}')
            else:
                order = self.df[column].value_counts().index
                ax = sns.countplot(data=self.df, x=column, order=order)
                ax.set_yscale('log')
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Distribution of {column}')
            plt.tight_layout()
            plt.show()
            plt.close()


    def correlation_analysis(self):
        logger.info('--- Calculating Correlation Matrix ---')
        num_df = self.df.select_dtypes(include=['number'])
        num_df=num_df.drop('CountryCode', axis=1)
        corr_matrix = num_df.corr(method='pearson')
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True,        
            cmap='coolwarm',   
            fmt=".2f",         
            linewidths=0.5
        )
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        plt.close()
        return corr_matrix

    def indentify_missing_vals(self):
        """Identify missing values to determine missing data and decide on appropriate imputation strategies."""
        logger.info('--- Missing value identification ---')
        missing_count = self.df.isnull().sum()
        missing_pct = self.df.isnull().mean() * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_count,
            'Missing_Pct': missing_pct
        })

        problem_cols = missing_df[missing_df['Missing_Count'] > 0].copy()

        if not problem_cols.empty:
            problem_cols['Advice'] = problem_cols['Missing_Pct'].apply(self._get_missing_advice)
            return problem_cols
        else:
            logger.info("Perfect! No missing data found.")
            return None


    def outlier_detection(self):
        logger.info('--- Running Outlier Analysis ---')
        num_cols = self.df.select_dtypes(include=['number']).columns
        outlier_report = {}

        for col in num_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    
            outlier_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outlier_report[col] = {'Count': outlier_count, 'Percentage': round((outlier_count / len(self.df)) * 100, 2)}

            plt.figure(figsize=(8, 3))
            sns.boxplot(x=self.df[col], color='salmon')
            plt.title(f'Distribution & Outliers: {col}')
            sns.despine()
            plt.tight_layout()
            plt.show()
            plt.close()
            
        return pd.DataFrame(outlier_report).T


    def run(self):
        self.results['overview']=self.data_overview()
        self.results['summary']=self.data_summary()
        self.results['missing']=self.indentify_missing_vals()
        self.numerical_cols_distribution()
        self.categroical_cols_distribution()
        self.results['correlation']=self.correlation_analysis()
        self.correlation_analysis()
        self.results['outliers']=self.outlier_detection()
        self.outlier_detection()
        logger.info("EDA process complete.")
        return self.results
    
    def summary_report(self):
        """A professional report generator."""
        pd.set_option('display.max_colwidth', None)
        print("\n=== DATA OVERVIEW ===")
        overview = self.results.get('overview', {})
        print(f"Dimensions: {overview['rows']} rows x {overview['cols']} columns")
        dtypes_df = pd.Series(overview['dtypes'], name="Data Type")
        print("\nData Types:")
        print(dtypes_df)
        print("\n=== DATA SUMMARY ===")
        results = self.results.get('summary', {})
        print(results['numeric_stats'].T)
        
        print("\n=== CATEGORICAL MODES ===")
        print(results['categorical_modes'])

        print("\n=== MISSING DATA REPORT ===")
        missing_report = self.results.get('missing')
        
        if missing_report is not None:
            print(missing_report)
        else:
            print("Status: Clean dataset. No missing values identified.")


