import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EDAConfig:
    """Configuration object for the EDA Pipeline.

    Attributes:
        skew_threshold (float): Threshold for flagging skewness.
        kurt_threshold (float): Threshold for flagging kurtosis.
        plot_dpi (int): Resolution for saved plots.
        save_plots (bool): Whether to save plots to disk.
        output_dir (str): Relative directory path for outputs.
        sample_size (Optional[int]): Rows to sample; if None, uses full data.
        random_seed (int): Seed for reproducibility.
        include_plots (bool): Whether to generate plots during execution.
        figsize_numerical (tuple): Tuple for numerical plot dimensions.
        figsize_categorical (tuple): Tuple for categorical plot dimensions.
        figsize_heatmap (tuple): Tuple for heatmap dimensions.
        verbose (bool): Whether to print findings to console.
    """
    skew_threshold: float = 1.0
    kurt_threshold: float = 3.0
    plot_dpi: int = 300
    save_plots: bool = True
    output_dir: str = "outputs"
    sample_size: Optional[int] = None
    random_seed: int = 42
    include_plots: bool = True
    figsize_numerical: tuple = (9, 5)
    figsize_categorical: tuple = (11, 6)
    figsize_heatmap: tuple = (12, 10)
    verbose: bool = True


class EDAPipeline:
    """
    Clean, modular, and production-ready EDA Pipeline for credit risk analysis.
    
    Features:
    - Method chaining support
    - Configurable via EDAConfig dataclass
    - Automatic artifact saving with timestamps
    - Separation of concerns (Visualizer, Reporter)
    """

    def __init__(self, config: Optional[EDAConfig] = None):
        """Initializes the pipeline with configuration and setup paths.

        Args:
            config (Optional[EDAConfig]): Custom configuration object.
        """
        self.config = config or EDAConfig()
        self.df: Optional[pd.DataFrame] = None
        self.results: Dict = {}
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.output_path = self._get_project_root() / self.config.output_dir / f"eda_run_{self.run_timestamp}"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self._visualizer = self._Visualizer(self.config, self.output_path)
        self._reporter = self._Reporter(self.output_path)
        
        logger.info(f"EDAPipeline initialized. Output directory: {self.output_path}")

    def _get_project_root(self) -> Path:
        """Locates the project root directory by traversing upwards.

        Returns:
            Path: The root directory path.
        """
        current_path = Path.cwd()
        for _ in range(5):  
            if (current_path / "src").exists() or (current_path / "notebooks").exists() or (current_path / "README.md").exists():
                return current_path
            current_path = current_path.parent
        
        return Path.cwd()
    
    def load_data(self, df: pd.DataFrame) -> 'EDAPipeline':
        """Loads and validates input data.

        Args:
            df (pd.DataFrame): Data to analyze.

        Returns:
            EDAPipeline: Self for method chaining.

        Raises:
            ValueError: If input is None or empty.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None.")
        self.df = df.copy()
        logger.info(f"✅ Data loaded: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        return self

    def run(self) -> Dict:
        """Executes the full EDA process and generates artifacts.

        Returns:
            Dict[str, Any]: Compiled results of the EDA process.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call .load_data(df) first.")

        logger.info(" Starting EDA Pipeline...")
        try:
            self._apply_sampling()

            self.results = {
                "overview": self._data_overview(),
                "summary": self._data_summary(),
                "missing": self._identify_missing_values(),
                "outliers": self._detect_outliers(),
                "correlation": self._correlation_analysis(),
            }
            if self.config.verbose:
                self._display_key_findings()


            if self.config.include_plots:
                self._generate_visualizations()

            self._reporter.generate_summary_report(self.results, self.df)
            self.save_config()

            logger.info(" EDA Pipeline completed successfully!")
            return self.results
        except Exception as e:
            logger.error(f" EDA Pipeline failed: {str(e)}")
            raise

    def save_config(self) -> None:
        """Saves current configuration to a JSON file for reproducibility."""
        config_path = self.output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
        logger.info(f" Configuration saved: {config_path}")

    # ===================== PRIVATE METHODS =====================

    def _display_key_findings(self) -> None:
        """Prints all required rubric items visibly in notebook"""
        print("\n" + "="*60)
        print("\t\tDATA OVERVIEW")
        print("="*60)
        ov = self.results['overview']
        print(f"Dataset Shape     : {ov['rows']} rows × {ov['cols']} columns")
        print(f"Memory Usage      : {ov['memory_usage_mb']} MB\n")

        print("\nSUMMARY STATISTICS (Numerical Features) \n")
        print(self.results['summary']['numeric_summary'])

        print("\nSUMMARY STATISTICS (Categorical Features) Most Frequent\n")
        print(self.results['summary']['categorical_modes'])

        print("\nMISSING VALUES")
        if self.results['missing'] is not None:
            print(self.results['missing'])
        else:
            print("No missing values found.")
        print("\nOUTLIER DETECTION")
        print(self.results['outliers'])

        print("\nCORRELATION MATRIX (Top 10 strongest) --- Mean Absolute Correlations")
        corr = self.results['correlation']
        print(corr.abs().mean().sort_values(ascending=False).head(10))

        print("\n" + "="*60)
        

    def _apply_sampling(self) -> None:
        """Samples the internal DataFrame based on config settings."""
        if self.config.sample_size and len(self.df) > self.config.sample_size:
            self.df = self.df.sample(n=self.config.sample_size, 
                                   random_state=self.config.random_seed).reset_index(drop=True)
            logger.info(f" Sampled to {self.config.sample_size:,} rows")

    def _data_overview(self) -> Dict:
        """Provides basic dataset dimensions and memory info.

        Returns:
            Dict[str, Any]: Dictionary containing shape and memory details.
        """
        return {
            "rows": self.df.shape[0],
            "cols": self.df.shape[1],
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
        }

    def _data_summary(self) -> Dict:
        num_df = self.df.select_dtypes(include=['number'])
        summary = pd.DataFrame({
            'mean': num_df.mean(),
            'median': num_df.median(),
            'std': num_df.std(),
            'skew': num_df.skew(),
            'kurtosis': num_df.kurt()
        })
        summary['insight'] = summary.apply(
            lambda row: self._get_distribution_insight(row['skew'], row['kurtosis']), axis=1
        )
        return {
            'numeric_summary': summary.round(4),
            'categorical_modes': self.df.select_dtypes(include=['object']).mode().iloc[0]
        }

    def _get_distribution_insight(self, skew: float, kurt: float) -> str:
        insights = []
        if abs(skew) > self.config.skew_threshold:
            direction = "right" if skew > 0 else "left"
            insights.append(f"Highly {direction}-skewed (skew={skew:.2f})")
        else:
            insights.append("Symmetric")

        if kurt > self.config.kurt_threshold:
            insights.append(f"Leptokurtic (kurt={kurt:.2f})")
        elif kurt < -1.0:
            insights.append(f"Platykurtic (kurt={kurt:.2f})")
        return " | ".join(insights)

    def _identify_missing_values(self):
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / len(self.df)) * 100
        missing_df = pd.DataFrame({'missing_count': missing_count, 'missing_pct': missing_pct.round(2)})
        missing_df = missing_df[missing_df['missing_count'] > 0]
        if not missing_df.empty:
            missing_df['advice'] = missing_df['missing_pct'].apply(self._get_missing_advice)
            return missing_df.sort_values('missing_pct', ascending=False)
        return None

    def _get_missing_advice(self, pct: float) -> str:
        if pct < 5: return "Impute (mean/median/mode)"
        elif pct < 30: return "Advanced imputation or drop"
        else: return "High missing rate - consider dropping column"

    def _detect_outliers(self) -> pd.DataFrame:
        num_cols = self.df.select_dtypes(include=['number']).columns
        outlier_report = {}
        for col in num_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            count = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            outlier_report[col] = {
                'outlier_count': count,
                'outlier_pct': round((count / len(self.df)) * 100, 2)
            }
        return pd.DataFrame(outlier_report).T

    def _correlation_analysis(self):
        num_df = self.df.select_dtypes(include=['number'])
        num_df = num_df.loc[:, num_df.nunique() > 1]
        return num_df.corr(method='pearson').round(3)

    def _generate_visualizations(self):
        logger.info(" Generating visualizations...")
        self._visualizer.plot_numerical(self.df)
        self._visualizer.plot_categorical(self.df)
        self._visualizer.plot_correlation_heatmap(self.df)
        self._visualizer.plot_outliers_box(self.df)


    class _Visualizer:
        """Helper class for generating and saving plots."""
        def __init__(self, config: EDAConfig, output_path: Path) -> None:
            self.config = config
            self.plot_path = output_path / "plots"
            self.plot_path.mkdir(exist_ok=True)

        def _save_and_show(self, filename: str):
            if self.config.save_plots:
                plt.savefig(self.plot_path / filename, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.show()         
            plt.close()

        def plot_numerical(self, df: pd.DataFrame) -> None:
            """Generates distribution plots for numeric features.

            Args:
                df (pd.DataFrame): The source dataframe.
            """
            for col in df.select_dtypes(include=['number']).columns:
                plt.figure(figsize=self.config.figsize_numerical)
                if df[col].nunique() <= 10:
                    sns.countplot(data=df, x=col)
                else:
                    sns.histplot(data=df, x=col, kde=True, bins=40)
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
                self._save_and_show(f"num_{col}.png")

        def plot_categorical(self, df: pd.DataFrame):
            for col in df.select_dtypes(include=['object']).columns:
                plt.figure(figsize=self.config.figsize_categorical)
                if df[col].nunique() > 25:
                    order = df[col].value_counts().iloc[:10].index
                    sns.countplot(data=df, y=col, order=order)
                    plt.title(f'Top 10 Values - {col}')
                else:
                    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
                    plt.xticks(rotation=45, ha='right')
                    plt.title(f'Distribution of {col}')
                plt.tight_layout()
                self._save_and_show(f"cat_{col}.png")

        def plot_correlation_heatmap(self, df: pd.DataFrame):
            num_df = df.select_dtypes(include=['number'])
            num_df = num_df.loc[:, num_df.nunique() > 1]
            if len(num_df.columns) < 2:
                return
            corr = num_df.corr()
            plt.figure(figsize=self.config.figsize_heatmap)
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            self._save_and_show("correlation_heatmap.png")


        def plot_outliers_box(self, df: pd.DataFrame):
            num_cols = df.select_dtypes(include=['number']).columns
            for col in num_cols:
                plt.figure(figsize=(10, 4))
                sns.boxplot(x=df[col], color='salmon')
                plt.title(f'Box Plot - Outliers in {col}')
                plt.tight_layout()
                self._save_and_show(f"boxplot_{col}.png")

    class _Reporter:
        def __init__(self, output_path: Path):
            self.output_path = output_path

        def generate_summary_report(self, results: Dict, df: pd.DataFrame):
            report_path = self.output_path / "eda_summary_report.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== EDA SUMMARY REPORT ===\n\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Run ID: eda_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}\n\n")

                ov = results['overview']
                f.write(f"Dataset Shape     : {ov['rows']} rows × {ov['cols']} columns\n")
                f.write(f"Memory Usage      : {ov['memory_usage_mb']} MB\n\n")

                f.write("=== NUMERIC SUMMARY ===\n")
                f.write(results['summary']['numeric_summary'].to_string() + "\n\n")

                f.write("=== CATEGORICAL SUMMARY ===\n")
                f.write(results['summary']['categorical_modes'].to_string() + "\n\n")

                f.write("=== MISSING VALUES ===\n")
                if results['missing'] is not None:
                    f.write(results['missing'].to_string() + "\n\n")
                else:
                    f.write("No missing values found.\n\n")

                f.write("=== OUTLIER REPORT ===\n")
                f.write(results['outliers'].to_string() + "\n\n")

                f.write("=== CORRELATION MATRIX (Top 10 features) ===\n")
                corr = results['correlation']
                f.write(corr.abs().mean().sort_values(ascending=False).head(10).to_string())
                f.write("\n\n")
                f.write(f"Full plots and artifacts saved in: {self.output_path}\n")

            logger.info(f"Summary report saved: {report_path}")




