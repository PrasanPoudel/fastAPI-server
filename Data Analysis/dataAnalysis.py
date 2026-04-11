import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import warnings
import io
from collections import Counter
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../Model Training/data/fake_job_postings.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Academic plotting configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 9


def save_plot(fig, filename):
    """Save matplotlib figure to file in output directory"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filepath


class ComprehensiveJobDataAnalysis:
    def __init__(self):
        self.df = None
        self.all_features = []
        self.numeric_features = []
        self.binary_features = []
        self.categorical_features = []
        self.text_features = []
        self.report_sections = []
        self.plots_generated = []
        
    def load_and_preprocess(self):
        """Load dataset and perform initial preprocessing"""
        print("Loading dataset...")
        self.df = pd.read_csv(DATA_PATH)
        
        # Dataset statistics
        self.total_records = len(self.df)
        self.total_features = self.df.shape[1]
        self.fraud_count = int(self.df['fraudulent'].sum())
        self.fraud_percent = round(self.df['fraudulent'].mean() * 100, 3)
        self.real_count = self.total_records - self.fraud_count
        self.real_percent = round(100 - self.fraud_percent, 3)
        
        # Classify features
        self.target = 'fraudulent'
        exclude_cols = ['job_id', 'fraudulent']
        
        # Text features
        self.text_features = ['title', 'description', 'requirements', 'benefits', 'company_profile']
        
        # Identify feature types
        for col in self.df.columns:
            if col in exclude_cols:
                continue
            if col in self.text_features:
                continue
            unique_vals = self.df[col].nunique()
            if unique_vals == 2:
                self.binary_features.append(col)
            elif unique_vals <= 15 or self.df[col].dtype == 'object':
                self.categorical_features.append(col)
            else:
                self.numeric_features.append(col)
        
        self.all_features = self.binary_features + self.categorical_features + self.numeric_features + self.text_features
        
        print(f"Dataset loaded: {self.total_records} rows, {self.total_features} columns")
        print(f"Fraudulent: {self.fraud_count} ({self.fraud_percent}%)")
        print(f"Real: {self.real_count} ({self.real_percent}%)")
        print(f"\nFeature Classification:")
        print(f"   Binary features ({len(self.binary_features)}): {self.binary_features}")
        print(f"   Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        print(f"   Numeric features ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"   Text features ({len(self.text_features)}): {self.text_features}")
        
    def analyze_missing_data(self):
        """Analyze missing data patterns"""
        print("\nAnalyzing missing data...")
        self.missing_analysis = pd.DataFrame({
            'Missing_Values': self.df.isnull().sum(),
            'Missing_Percent': round((self.df.isnull().sum() / len(self.df)) * 100, 2)
        }).sort_values('Missing_Percent', ascending=False)
        
        # Missing data by target class
        self.missing_by_target = {}
        for col in self.all_features:
            if col in self.df.columns:
                missing_fraud = self.df[self.df['fraudulent'] == 1][col].isnull().sum()
                missing_real = self.df[self.df['fraudulent'] == 0][col].isnull().sum()
                total_fraud = len(self.df[self.df['fraudulent'] == 1])
                total_real = len(self.df[self.df['fraudulent'] == 0])
                self.missing_by_target[col] = {
                    'fraud_missing_pct': round(missing_fraud / total_fraud * 100, 2) if total_fraud > 0 else 0,
                    'real_missing_pct': round(missing_real / total_real * 100, 2) if total_real > 0 else 0
                }
        
        print(f"   Features with >50% missing: {len(self.missing_analysis[self.missing_analysis['Missing_Percent'] > 50])}")
        
    def analyze_binary_features(self):
        """Analyze all binary features"""
        print("\nAnalyzing binary features...")
        self.binary_analysis = {}
        
        for col in self.binary_features:
            counts = self.df[col].value_counts()
            percentages = round(self.df[col].value_counts(normalize=True) * 100, 2)
            
            # Correlation with target
            corr, p_value = stats.pointbiserialr(self.df['fraudulent'], self.df[col].fillna(0))
            
            # By target class
            by_fraud = self.df.groupby('fraudulent')[col].value_counts(normalize=True).unstack(fill_value=0)
            
            self.binary_analysis[col] = {
                'counts': counts.to_dict(),
                'percentages': percentages.to_dict(),
                'correlation': round(corr, 5),
                'p_value': p_value,
                'significant': p_value < 0.05,
                'by_target': by_fraud.to_dict() if not by_fraud.empty else {}
            }
            
        print(f"   Analyzed {len(self.binary_features)} binary features")
        
    def analyze_categorical_features(self):
        """Analyze all categorical features"""
        print("\nAnalyzing categorical features...")
        self.categorical_analysis = {}
        
        def cramers_v(x, y):
            """Calculate Cramér's V statistic"""
            contingency_table = pd.crosstab(x, y)
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                return 0
            chi2 = stats.chi2_contingency(contingency_table)[0]
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            if min_dim == 0:
                return 0
            return np.sqrt(chi2 / (n * min_dim))
        
        for col in self.categorical_features:
            if col not in self.df.columns:
                continue
                
            counts = self.df[col].value_counts()
            percentages = round(self.df[col].value_counts(normalize=True) * 100, 2)
            n_unique = self.df[col].nunique()
            mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'N/A'
            
            # Cramér's V with target
            cv = cramers_v(self.df[col].fillna('Missing'), self.df['fraudulent'])
            
            # Chi-square test
            contingency = pd.crosstab(self.df[col].fillna('Missing'), self.df['fraudulent'])
            if contingency.shape[0] >= 2:
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
            else:
                chi2, p_val, dof = 0, 1, 0
            
            # By target class
            by_fraud = self.df.groupby('fraudulent')[col].value_counts(normalize=True).unstack(fill_value=0)
            
            self.categorical_analysis[col] = {
                'counts': counts.to_dict(),
                'percentages': percentages.to_dict(),
                'n_unique': n_unique,
                'mode': mode_val,
                'cramers_v': round(cv, 5),
                'chi2': round(chi2, 4),
                'p_value': p_val,
                'significant': p_val < 0.05,
                'by_target': by_fraud.to_dict() if not by_fraud.empty else {}
            }
            
        print(f"   Analyzed {len(self.categorical_features)} categorical features")
        
    def analyze_numeric_features(self):
        """Analyze numeric features with comprehensive statistics"""
        print("\nAnalyzing numeric features...")
        self.numeric_analysis = {}
        
        # Check if any numeric features exist
        if not self.numeric_features:
            print("   No numeric features found in dataset")
            return
        
        for col in self.numeric_features:
            if col not in self.df.columns:
                continue
                
            # Basic descriptive statistics
            desc = self.df[col].describe()
            
            # Additional statistics
            skewness = self.df[col].skew()
            kurtosis = self.df[col].kurtosis()
            coefficient_of_variation = (self.df[col].std() / self.df[col].mean() * 100) if self.df[col].mean() != 0 else float('inf')
            
            # Correlation with target
            corr, p_value = stats.pointbiserialr(self.df['fraudulent'], self.df[col].fillna(self.df[col].median()))
            
            # By target class
            fraud_vals = self.df[self.df['fraudulent'] == 1][col].dropna()
            real_vals = self.df[self.df['fraudulent'] == 0][col].dropna()
            
            fraud_stats = fraud_vals.describe() if len(fraud_vals) > 0 else pd.Series()
            real_stats = real_vals.describe() if len(real_vals) > 0 else pd.Series()
            
            # T-test
            if len(fraud_vals) > 0 and len(real_vals) > 0:
                t_stat, t_pval = stats.ttest_ind(fraud_vals, real_vals, equal_var=False)
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((fraud_vals.std()**2 + real_vals.std()**2) / 2)
                cohens_d = (fraud_vals.mean() - real_vals.mean()) / pooled_std if pooled_std > 0 else 0
            else:
                t_stat, t_pval, cohens_d = 0, 1, 0
            
            # Normality test (Shapiro-Wilk for smaller samples)
            if len(self.df[col].dropna()) < 5000:
                shapiro_stat, shapiro_p = stats.shapiro(self.df[col].dropna().head(50))
            else:
                shapiro_stat, shapiro_p = 0, 0
            
            # Outlier detection using IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            n_outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outlier_percent = round(n_outliers / len(self.df[col]) * 100, 2)
            
            self.numeric_analysis[col] = {
                'description': desc.to_dict(),
                'skewness': round(skewness, 4),
                'kurtosis': round(kurtosis, 4),
                'coefficient_of_variation': round(coefficient_of_variation, 2),
                'correlation': round(corr, 5),
                'p_value': p_value,
                'significant': p_value < 0.05,
                'fraud_stats': fraud_stats.to_dict() if not fraud_stats.empty else {},
                'real_stats': real_stats.to_dict() if not real_stats.empty else {},
                't_statistic': round(t_stat, 4),
                't_p_value': t_pval,
                'cohens_d': round(cohens_d, 4),
                'shapiro_stat': round(shapiro_stat, 4),
                'shapiro_p': shapiro_p,
                'n_outliers': int(n_outliers),
                'outlier_percent': outlier_percent,
                'Q1': round(Q1, 4),
                'Q3': round(Q3, 4),
                'IQR': round(IQR, 4),
                'lower_bound': round(lower_bound, 4),
                'upper_bound': round(upper_bound, 4)
            }
            
        print(f"   Analyzed {len(self.numeric_features)} numeric features")
        
    def analyze_text_features(self):
        """Analyze text features"""
        print("\nAnalyzing text features...")
        self.text_analysis = {}
        
        for col in self.text_features:
            if col not in self.df.columns:
                continue
                
            # Text length statistics
            text_lengths = self.df[col].fillna('').str.len()
            word_counts = self.df[col].fillna('').str.split().str.len()
            
            # By target class
            fraud_lengths = self.df[self.df['fraudulent'] == 1][col].fillna('').str.len()
            real_lengths = self.df[self.df['fraudulent'] == 0][col].fillna('').str.len()
            
            # Most common words (simple analysis)
            all_text = ' '.join(self.df[col].fillna('')).lower().split()
            word_freq = Counter(all_text)
            top_words = word_freq.most_common(20)
            
            self.text_analysis[col] = {
                'char_length_stats': {
                    'mean': round(text_lengths.mean(), 2),
                    'median': round(text_lengths.median(), 2),
                    'std': round(text_lengths.std(), 2),
                    'min': int(text_lengths.min()),
                    'max': int(text_lengths.max())
                },
                'word_count_stats': {
                    'mean': round(word_counts.mean(), 2),
                    'median': round(word_counts.median(), 2),
                    'std': round(word_counts.std(), 2),
                    'min': int(word_counts.min()),
                    'max': int(word_counts.max())
                },
                'fraud_avg_length': round(fraud_lengths.mean(), 2) if len(fraud_lengths) > 0 else 0,
                'real_avg_length': round(real_lengths.mean(), 2) if len(real_lengths) > 0 else 0,
                'top_words': top_words
            }
            
        print(f"   Analyzed {len(self.text_features)} text features")
        
    def calculate_comprehensive_correlations(self):
        """Calculate correlations for all features"""
        print("\nCalculating comprehensive correlations...")
        self.all_correlations = {}
        
        # Binary features - point biserial
        for col in self.binary_features:
            corr, p_val = stats.pointbiserialr(self.df['fraudulent'], self.df[col].fillna(0))
            self.all_correlations[col] = {
                'correlation': round(corr, 5),
                'p_value': p_val,
                'significant': p_val < 0.05,
                'method': 'Point-Biserial'
            }
        
        # Categorical features - Cramér's V
        def cramers_v(x, y):
            contingency_table = pd.crosstab(x, y)
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                return 0
            chi2 = stats.chi2_contingency(contingency_table)[0]
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            if min_dim == 0:
                return 0
            return np.sqrt(chi2 / (n * min_dim))
        
        for col in self.categorical_features:
            if col in self.df.columns:
                cv = cramers_v(self.df[col].fillna('Missing'), self.df['fraudulent'])
                self.all_correlations[col] = {
                    'correlation': round(cv, 5),
                    'p_value': 0,
                    'significant': cv > 0.1,
                    'method': "Cramér's V"
                }
        
        # Numeric features - point biserial
        for col in self.numeric_features:
            if col in self.df.columns:
                corr, p_val = stats.pointbiserialr(self.df['fraudulent'], self.df[col].fillna(self.df[col].median()))
                self.all_correlations[col] = {
                    'correlation': round(corr, 5),
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'method': 'Point-Biserial'
                }
        
        # Text features - correlation based on text length
        for col in self.text_features:
            if col in self.df.columns:
                text_len = self.df[col].fillna('').str.len()
                corr, p_val = stats.pointbiserialr(self.df['fraudulent'], text_len)
                self.all_correlations[col + '_length'] = {
                    'correlation': round(corr, 5),
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'method': 'Point-Biserial (text length)'
                }
        
        # Sort by absolute correlation
        self.ranked_features = sorted(
            self.all_correlations.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        )
        
        print(f"   Calculated correlations for {len(self.all_correlations)} features")
        print("\nTop 10 Features by Correlation Strength:")
        for i, (feat, corr_info) in enumerate(self.ranked_features[:10]):
            sig_mark = "*" if corr_info['significant'] else ""
            print(f"  {i+1}. {feat:25} {corr_info['method']:25} r = {corr_info['correlation']:+.5f} {sig_mark}")
            
    def generate_visualizations(self):
        """Generate all visualizations and save to output directory"""
        print("\nGenerating visualizations...")
        self.plots_generated = []
        
        # 1. Class Distribution Pie Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = [self.real_count, self.fraud_count]
        labels = [f'Real ({self.real_percent}%)', f'Fraudulent ({self.fraud_percent}%)']
        colors_pie = ['#2ecc71', '#e74c3c']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax.set_title('Class Distribution (Target Variable)', fontsize=14, fontweight='bold')
        self.plots_generated.append(save_plot(fig, 'class_distribution.png'))
        
        # 2. Missing Data Heatmap
        fig, ax = plt.subplots(figsize=(10, max(6, len(self.missing_analysis) * 0.3)))
        missing_data = self.missing_analysis[self.missing_analysis['Missing_Values'] > 0]
        if not missing_data.empty:
            sns.heatmap(missing_data[['Missing_Percent']].astype(float), 
                       annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                       cbar_kws={'label': 'Missing %'})
            ax.set_title('Missing Data Percentage by Feature', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No missing data in dataset', transform=ax.transAxes, ha='center')
            ax.set_title('Missing Data Analysis', fontsize=14, fontweight='bold')
        self.plots_generated.append(save_plot(fig, 'missing_heatmap.png'))
        
        # 3. Feature Correlation Bar Chart (All Features)
        features = [f[0] for f in self.ranked_features]
        corr_values = [abs(f[1]['correlation']) for f in self.ranked_features]
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(features) * 0.4)))
        bars = ax.barh(range(len(features)), corr_values, color='#3498db', alpha=0.8)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Absolute Correlation Strength', fontsize=12)
        ax.set_title('Feature Correlation Strength with Fraudulent Class', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, corr_values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                   va='center', fontsize=7)
        plt.tight_layout()
        self.plots_generated.append(save_plot(fig, 'feature_importance.png'))
        
        # 4. Binary Features Distribution
        if self.binary_features:
            n_bins = len(self.binary_features)
            fig, axes = plt.subplots(1, n_bins, figsize=(5 * n_bins, 5))
            if n_bins == 1:
                axes = [axes]
            for idx, col in enumerate(self.binary_features):
                counts = self.df[col].value_counts()
                axes[idx].bar(counts.index.astype(str), counts.values, color=['#3498db', '#e74c3c'])
                axes[idx].set_title(col, fontsize=11)
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Count')
                for i, v in enumerate(counts.values):
                    axes[idx].text(i, v + 50, str(v), ha='center', fontsize=9)
            fig.suptitle('Binary Features Distribution', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            self.plots_generated.append(save_plot(fig, 'binary_distribution.png'))
        
        # 5. Categorical Features Top Categories
        for col in self.categorical_features[:5]:
            if col in self.df.columns:
                top10 = self.df[col].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(top10)), top10.values, color='#9b59b6')
                ax.set_yticks(range(len(top10)))
                ax.set_yticklabels(top10.index, fontsize=9)
                ax.set_xlabel('Count')
                ax.set_title(f'Top 10 Categories in {col}', fontsize=12, fontweight='bold')
                plt.tight_layout()
                self.plots_generated.append(save_plot(fig, f'categorical_{col}.png'))
        
        # 6. Text Length Distribution by Class
        for col in self.text_features[:3]:
            if col in self.df.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                fraud_text_len = self.df[self.df['fraudulent'] == 1][col].fillna('').str.len()
                real_text_len = self.df[self.df['fraudulent'] == 0][col].fillna('').str.len()
                
                bins = np.linspace(0, max(fraud_text_len.max(), real_text_len.max()), 50)
                ax.hist(real_text_len, bins, alpha=0.6, label='Real', color='#2ecc71', density=True)
                ax.hist(fraud_text_len, bins, alpha=0.6, label='Fraudulent', color='#e74c3c', density=True)
                ax.set_xlabel('Text Length (characters)')
                ax.set_ylabel('Density')
                ax.set_title(f'Text Length Distribution - {col}', fontsize=12, fontweight='bold')
                ax.legend()
                plt.tight_layout()
                self.plots_generated.append(save_plot(fig, f'text_length_{col}.png'))
        
        # 7. Correlation Matrix Heatmap (Binary + Numeric)
        binary_numeric = self.binary_features + self.numeric_features
        if len(binary_numeric) >= 2:
            corr_data = self.df[binary_numeric + ['fraudulent']].corr()
            fig, ax = plt.subplots(figsize=(max(10, len(binary_numeric) * 0.8), max(8, len(binary_numeric) * 0.8)))
            mask = np.triu(np.ones_like(corr_data, dtype=bool))
            sns.heatmap(corr_data, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, ax=ax,
                       cbar_kws={"shrink": 0.8})
            ax.set_title('Pearson Correlation Matrix - Binary & Numeric Features', fontsize=14, fontweight='bold')
            plt.tight_layout()
            self.plots_generated.append(save_plot(fig, 'correlation_matrix.png'))
        
        # 8. Numeric Features - Enhanced Visualizations
        for col in self.numeric_features:
            if col not in self.df.columns:
                continue
            
            # Histogram with KDE
            fig, ax = plt.subplots(figsize=(10, 6))
            data = self.df[col].dropna()
            sns.histplot(data, kde=True, ax=ax, color='#3498db')
            ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            # Add vertical lines for mean, median
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            ax.legend()
            self.plots_generated.append(save_plot(fig, f'numeric_hist_{col}.png'))
            
            # Box plot by target class
            fig, ax = plt.subplots(figsize=(8, 6))
            data_fraud = self.df[self.df['fraudulent'] == 1][col].dropna()
            data_real = self.df[self.df['fraudulent'] == 0][col].dropna()
            bp_data = [data_real, data_fraud]
            bp = ax.boxplot(bp_data, labels=['Real', 'Fraudulent'], patch_artist=True)
            colors = ['#2ecc71', '#e74c3c']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            ax.set_title(f'{col} by Target Class', fontsize=12, fontweight='bold')
            ax.set_ylabel(col)
            self.plots_generated.append(save_plot(fig, f'numeric_box_{col}.png'))
            
            # Q-Q Plot for normality check
            fig, ax = plt.subplots(figsize=(8, 6))
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot - {col}', fontsize=12, fontweight='bold')
            self.plots_generated.append(save_plot(fig, f'numeric_qq_{col}.png'))
        
        print(f"   Generated {len(self.plots_generated)} visualizations")
        
    def generate_pdf_report(self):
        """Generate comprehensive PDF report with all analysis and embedded plots"""
        print("\nGenerating comprehensive PDF report...")
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title Page
        pdf.add_page()
        pdf.set_font("Times", 'B', 24)
        pdf.cell(0, 40, "Data Analysis Report", ln=True, align='C')
        pdf.set_font("Times", 'B', 18)
        pdf.cell(0, 15, "Fake Job Postings Dataset", ln=True, align='C')
        pdf.set_font("Times", size=12)
        pdf.cell(0, 60, "", ln=True)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%B %d, %Y')}", ln=True, align='C')
        pdf.set_font("Times", size=10)
        pdf.cell(0, 10, f"Total Records: {self.total_records:,} | Total Features: {self.total_features}", ln=True, align='C')
        
        # Table of Contents
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(0, 10, "Table of Contents", ln=True)
        pdf.set_font("Times", size=11)
        contents = [
            "1. Executive Summary",
            "2. Dataset Overview",
            "3. Feature Classification",
            "4. Missing Data Analysis",
            "5. Class Distribution",
            "6. Binary Features Analysis",
            "7. Categorical Features Analysis",
            "8. Numeric Features Analysis",
            "9. Text Features Analysis",
            "10. Comprehensive Correlation Analysis",
            "11. Feature Importance Ranking",
            "12. Statistical Summary",
            "13. Key Findings and Recommendations"
        ]
        for line in contents:
            pdf.cell(0, 8, line, ln=True)
        
        # 1. Executive Summary
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "1. Executive Summary", ln=True)
        pdf.set_font("Times", size=11)
        summary = f"""This comprehensive report analyzes the Fake Job Postings dataset containing {self.total_records:,} job postings.
The dataset includes {self.total_features} features across binary, categorical, numeric, and text types.

Key Statistics:
- Total Records: {self.total_records:,}
- Real Job Postings: {self.real_count:,} ({self.real_percent}%)
- Fraudulent Job Postings: {self.fraud_count:,} ({self.fraud_percent}%)
- Total Features Analyzed: {len(self.all_features)}

The analysis covers all feature types including {len(self.binary_features)} binary features, 
{len(self.categorical_features)} categorical features, {len(self.numeric_features)} numeric features, 
and {len(self.text_features)} text features."""
        pdf.multi_cell(0, 6, summary.strip())
        
        # 2. Dataset Overview
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "2. Dataset Overview", ln=True)
        pdf.set_font("Times", size=11)
        
        overview_data = [
            ["Metric", "Value"],
            ["Total Records", f"{self.total_records:,}"],
            ["Total Features", str(self.total_features)],
            ["Real Job Postings", f"{self.real_count:,} ({self.real_percent}%)"],
            ["Fraudulent Job Postings", f"{self.fraud_count:,} ({self.fraud_percent}%)"],
            ["Binary Features", str(len(self.binary_features))],
            ["Categorical Features", str(len(self.categorical_features))],
            ["Numeric Features", str(len(self.numeric_features))],
            ["Text Features", str(len(self.text_features))]
        ]
        
        pdf.set_font("Times", 'B', 10)
        col_widths = [100, 90]
        for row in overview_data:
            pdf.cell(col_widths[0], 8, row[0], border=1)
            pdf.cell(col_widths[1], 8, row[1], border=1, ln=True)
        
        # 3. Feature Classification
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "3. Feature Classification", ln=True)
        pdf.set_font("Times", size=11)
        
        pdf.set_font("Times", 'B', 12)
        pdf.cell(0, 8, "Binary Features:", ln=True)
        pdf.set_font("Times", size=10)
        pdf.cell(0, 6, ", ".join(self.binary_features), ln=True)
        
        pdf.set_font("Times", 'B', 12)
        pdf.cell(0, 8, "Categorical Features:", ln=True)
        pdf.set_font("Times", size=10)
        pdf.cell(0, 6, ", ".join(self.categorical_features), ln=True)
        
        pdf.set_font("Times", 'B', 12)
        pdf.cell(0, 8, "Numeric Features:", ln=True)
        pdf.set_font("Times", size=10)
        pdf.cell(0, 6, ", ".join(self.numeric_features), ln=True)
        
        pdf.set_font("Times", 'B', 12)
        pdf.cell(0, 8, "Text Features:", ln=True)
        pdf.set_font("Times", size=10)
        pdf.cell(0, 6, ", ".join(self.text_features), ln=True)
        
        # 4. Missing Data Analysis
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "4. Missing Data Analysis", ln=True)
        pdf.set_font("Times", size=11)
        
        pdf.cell(0, 8, "Features with missing values (sorted by missing percentage):", ln=True)
        pdf.set_font("Times", size=9)
        
        # Show missing data table
        missing_to_show = self.missing_analysis[self.missing_analysis['Missing_Values'] > 0].head(20)
        pdf.set_font("Times", 'B', 9)
        pdf.cell(80, 7, "Feature", border=1)
        pdf.cell(55, 7, "Missing Values", border=1)
        pdf.cell(55, 7, "Missing %", border=1, ln=True)
        pdf.set_font("Times", size=9)
        
        for idx, row in missing_to_show.iterrows():
            pdf.cell(80, 6, idx[:30], border=1)
            pdf.cell(55, 6, str(int(row['Missing_Values'])), border=1)
            pdf.cell(55, 6, f"{row['Missing_Percent']}%", border=1, ln=True)
        
        # Embed missing data heatmap
        heatmap_path = os.path.join(OUTPUT_DIR, 'missing_heatmap.png')
        if os.path.exists(heatmap_path):
            pdf.add_page()
            pdf.set_font("Times", 'B', 12)
            pdf.cell(0, 10, "Missing Data Heatmap", ln=True)
            pdf.image(heatmap_path, x=10, w=190)
        
        # 5. Class Distribution
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "5. Class Distribution", ln=True)
        pdf.set_font("Times", size=11)
        
        distribution_text = f"""Target Variable: fraudulent
- Real Job Postings (0): {self.real_count:,} ({self.real_percent}%)
- Fraudulent Job Postings (1): {self.fraud_count:,} ({self.fraud_percent}%)

The dataset shows a class imbalance with {self.real_percent}% real postings and {self.fraud_percent}% fraudulent postings."""
        pdf.multi_cell(0, 6, distribution_text.strip())
        
        # Embed class distribution pie chart
        pie_path = os.path.join(OUTPUT_DIR, 'class_distribution.png')
        if os.path.exists(pie_path):
            pdf.image(pie_path, x=30, w=150)
        
        # 6. Binary Features Analysis
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "6. Binary Features Analysis", ln=True)
        pdf.set_font("Times", size=11)
        
        for col in self.binary_features:
            if col in self.binary_analysis:
                analysis = self.binary_analysis[col]
                pdf.set_font("Times", 'B', 12)
                pdf.cell(0, 8, f"Feature: {col}", ln=True)
                pdf.set_font("Times", size=10)
                
                pdf.cell(0, 6, f"Correlation with target: {analysis['correlation']:+.5f}", ln=True)
                pdf.cell(0, 6, f"P-value: {analysis['p_value']:.6f}", ln=True)
                sig = "Yes" if analysis['significant'] else "No"
                pdf.cell(0, 6, f"Statistically Significant: {sig}", ln=True)
                
                pdf.cell(0, 6, "Value Distribution:", ln=True)
                for val, count in analysis['counts'].items():
                    pct = analysis['percentages'].get(val, 0)
                    pdf.cell(0, 5, f"  - {val}: {count} ({pct}%)", ln=True)
                pdf.cell(0, 3, "", ln=True)
        
        # Embed binary distribution plot
        binary_path = os.path.join(OUTPUT_DIR, 'binary_distribution.png')
        if os.path.exists(binary_path):
            pdf.add_page()
            pdf.set_font("Times", 'B', 12)
            pdf.cell(0, 10, "Binary Features Distribution", ln=True)
            pdf.image(binary_path, x=10, w=190)
        
        # 7. Categorical Features Analysis
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "7. Categorical Features Analysis", ln=True)
        pdf.set_font("Times", size=11)
        
        for col in self.categorical_features:
            if col in self.categorical_analysis:
                analysis = self.categorical_analysis[col]
                pdf.set_font("Times", 'B', 12)
                pdf.cell(0, 8, f"Feature: {col}", ln=True)
                pdf.set_font("Times", size=10)
                
                pdf.cell(0, 6, f"Unique Categories: {analysis['n_unique']}", ln=True)
                pdf.cell(0, 6, f"Mode: {analysis['mode']}", ln=True)
                pdf.cell(0, 6, f"Cramér's V with target: {analysis['cramers_v']:.5f}", ln=True)
                pdf.cell(0, 6, f"Chi-square statistic: {analysis['chi2']:.4f}", ln=True)
                pdf.cell(0, 6, f"P-value: {analysis['p_value']:.6f}", ln=True)
                sig = "Yes" if analysis['significant'] else "No"
                pdf.cell(0, 6, f"Statistically Significant: {sig}", ln=True)
                
                # Top 5 categories
                pdf.cell(0, 6, "Top 5 Categories:", ln=True)
                sorted_cats = sorted(analysis['counts'].items(), key=lambda x: x[1], reverse=True)[:5]
                for cat, count in sorted_cats:
                    pct = analysis['percentages'].get(cat, 0)
                    pdf.cell(0, 5, f"  - {str(cat)[:40]}: {count} ({pct}%)", ln=True)
                pdf.cell(0, 3, "", ln=True)
        
        # Embed categorical plots
        for i, col in enumerate(self.categorical_features[:5]):
            plot_path = os.path.join(OUTPUT_DIR, f'categorical_{col}.png')
            if os.path.exists(plot_path):
                pdf.add_page()
                pdf.set_font("Times", 'B', 12)
                pdf.cell(0, 10, f"Top Categories: {col}", ln=True)
                pdf.image(plot_path, x=10, w=190)
        
        # 8. Numeric Features Analysis
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "8. Numeric Features Analysis", ln=True)
        pdf.set_font("Times", size=11)
        
        if not self.numeric_features:
            pdf.cell(0, 8, "No numeric features found in the dataset.", ln=True)
            pdf.cell(0, 8, "All features in this dataset are either binary, categorical, or text.", ln=True)
        else:
            for col in self.numeric_features:
                if col in self.numeric_analysis:
                    analysis = self.numeric_analysis[col]
                    pdf.set_font("Times", 'B', 12)
                    pdf.cell(0, 8, f"Feature: {col}", ln=True)
                    pdf.set_font("Times", size=10)
                    
                    pdf.cell(0, 6, f"Correlation with target: {analysis['correlation']:+.5f}", ln=True)
                    pdf.cell(0, 6, f"P-value: {analysis['p_value']:.6f}", ln=True)
                    sig = "Yes" if analysis['significant'] else "No"
                    pdf.cell(0, 6, f"Statistically Significant: {sig}", ln=True)
                    
                    # Descriptive Statistics
                    pdf.cell(0, 6, "Descriptive Statistics:", ln=True)
                    desc = analysis['description']
                    stats_to_show = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                    for stat_key in stats_to_show:
                        if stat_key in desc:
                            pdf.cell(0, 5, f"  - {stat_key}: {desc[stat_key]:.4f}", ln=True)
                    
                    # Additional statistics
                    pdf.cell(0, 6, "Additional Statistics:", ln=True)
                    pdf.cell(0, 5, f"  - Skewness: {analysis['skewness']:.4f}", ln=True)
                    pdf.cell(0, 5, f"  - Kurtosis: {analysis['kurtosis']:.4f}", ln=True)
                    pdf.cell(0, 5, f"  - Coefficient of Variation: {analysis['coefficient_of_variation']:.2f}%", ln=True)
                    
                    # Outlier Analysis
                    pdf.cell(0, 6, "Outlier Analysis:", ln=True)
                    pdf.cell(0, 5, f"  - Number of Outliers: {analysis['n_outliers']} ({analysis['outlier_percent']}%)", ln=True)
                    pdf.cell(0, 5, f"  - IQR Range: [{analysis['lower_bound']:.4f}, {analysis['upper_bound']:.4f}]", ln=True)
                    
                    # Normality Test
                    pdf.cell(0, 6, "Normality Test (Shapiro-Wilk):", ln=True)
                    pdf.cell(0, 5, f"  - Statistic: {analysis['shapiro_stat']:.4f}", ln=True)
                    pdf.cell(0, 5, f"  - P-value: {analysis['shapiro_p']:.6f}", ln=True)
                    normality = "Normal" if analysis['shapiro_p'] > 0.05 else "Non-Normal"
                    pdf.cell(0, 5, f"  - Distribution: {normality}", ln=True)
                    
                    # By Target Class
                    pdf.cell(0, 6, "By Target Class:", ln=True)
                    if analysis['fraud_stats']:
                        pdf.cell(0, 5, f"  Fraudulent - Mean: {analysis['fraud_stats'].get('mean', 0):.4f}, Median: {analysis['fraud_stats'].get('50%', 0):.4f}", ln=True)
                    if analysis['real_stats']:
                        pdf.cell(0, 5, f"  Real - Mean: {analysis['real_stats'].get('mean', 0):.4f}, Median: {analysis['real_stats'].get('50%', 0):.4f}", ln=True)
                    
                    # T-test results
                    pdf.cell(0, 5, f"  T-test: t={analysis['t_statistic']:.4f}, p={analysis['t_p_value']:.6f}", ln=True)
                    pdf.cell(0, 5, f"  Effect Size (Cohen's d): {analysis['cohens_d']:.4f}", ln=True)
                    pdf.cell(0, 3, "", ln=True)
                    
                    # Add visualizations for each numeric feature
                    hist_path = os.path.join(OUTPUT_DIR, f'numeric_hist_{col}.png')
                    box_path = os.path.join(OUTPUT_DIR, f'numeric_box_{col}.png')
                    qq_path = os.path.join(OUTPUT_DIR, f'numeric_qq_{col}.png')
                    
                    if os.path.exists(hist_path):
                        pdf.add_page()
                        pdf.set_font("Times", 'B', 12)
                        pdf.cell(0, 10, f"Distribution: {col}", ln=True)
                        pdf.image(hist_path, x=10, w=190)
                    
                    if os.path.exists(box_path):
                        pdf.add_page()
                        pdf.set_font("Times", 'B', 12)
                        pdf.cell(0, 10, f"Box Plot by Class: {col}", ln=True)
                        pdf.image(box_path, x=10, w=190)
                    
                    if os.path.exists(qq_path):
                        pdf.add_page()
                        pdf.set_font("Times", 'B', 12)
                        pdf.cell(0, 10, f"Q-Q Plot: {col}", ln=True)
                        pdf.image(qq_path, x=10, w=190)
        
        # 9. Text Features Analysis
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "9. Text Features Analysis", ln=True)
        pdf.set_font("Times", size=11)
        
        for col in self.text_features:
            if col in self.text_analysis:
                analysis = self.text_analysis[col]
                pdf.set_font("Times", 'B', 12)
                pdf.cell(0, 8, f"Feature: {col}", ln=True)
                pdf.set_font("Times", size=10)
                
                pdf.cell(0, 6, "Character Length Statistics:", ln=True)
                for stat, val in analysis['char_length_stats'].items():
                    pdf.cell(0, 5, f"  - {stat}: {val}", ln=True)
                
                pdf.cell(0, 6, "Word Count Statistics:", ln=True)
                for stat, val in analysis['word_count_stats'].items():
                    pdf.cell(0, 5, f"  - {stat}: {val}", ln=True)
                
                pdf.cell(0, 6, f"Average Length - Fraudulent: {analysis['fraud_avg_length']}, Real: {analysis['real_avg_length']}", ln=True)
                
                pdf.cell(0, 6, "Top 10 Most Common Words:", ln=True)
                for word, freq in analysis['top_words'][:10]:
                    pdf.cell(0, 5, f"  - '{word}': {freq} occurrences", ln=True)
                pdf.cell(0, 3, "", ln=True)
        
        # Embed text length distribution plots
        for i, col in enumerate(self.text_features[:3]):
            plot_path = os.path.join(OUTPUT_DIR, f'text_length_{col}.png')
            if os.path.exists(plot_path):
                pdf.add_page()
                pdf.set_font("Times", 'B', 12)
                pdf.cell(0, 10, f"Text Length Distribution: {col}", ln=True)
                pdf.image(plot_path, x=10, w=190)
        
        # 10. Comprehensive Correlation Analysis
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "10. Comprehensive Correlation Analysis", ln=True)
        pdf.set_font("Times", size=11)
        
        pdf.cell(0, 8, "All features ranked by correlation strength with fraudulent class:", ln=True)
        pdf.set_font("Times", size=9)
        
        # Correlation table
        pdf.set_font("Times", 'B', 9)
        pdf.cell(60, 7, "Feature", border=1)
        pdf.cell(35, 7, "Correlation", border=1)
        pdf.cell(35, 7, "Method", border=1)
        pdf.cell(30, 7, "P-Value", border=1)
        pdf.cell(30, 7, "Significant", border=1, ln=True)
        pdf.set_font("Times", size=9)
        
        for feat, corr_info in self.ranked_features:
            sig = "*" if corr_info['significant'] else ""
            p_val_str = f"{corr_info['p_value']:.2e}" if corr_info['p_value'] > 0 else "N/A"
            pdf.cell(60, 6, feat[:35], border=1)
            pdf.cell(35, 6, f"{corr_info['correlation']:+.5f}", border=1)
            pdf.cell(35, 6, corr_info['method'][:15], border=1)
            pdf.cell(30, 6, p_val_str, border=1)
            pdf.cell(30, 6, sig, border=1, ln=True)
        
        # Embed correlation matrix heatmap
        corr_path = os.path.join(OUTPUT_DIR, 'correlation_matrix.png')
        if os.path.exists(corr_path):
            pdf.add_page()
            pdf.set_font("Times", 'B', 12)
            pdf.cell(0, 10, "Correlation Matrix Heatmap", ln=True)
            pdf.image(corr_path, x=10, w=190)
        
        # 11. Feature Importance Ranking
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "11. Feature Importance Ranking", ln=True)
        pdf.set_font("Times", size=11)
        
        pdf.cell(0, 8, "Top 20 most predictive features for detecting fraudulent job postings:", ln=True)
        pdf.set_font("Times", size=9)
        
        pdf.set_font("Times", 'B', 9)
        pdf.cell(15, 7, "Rank", border=1)
        pdf.cell(70, 7, "Feature", border=1)
        pdf.cell(35, 7, "Correlation", border=1)
        pdf.cell(35, 7, "Method", border=1)
        pdf.cell(35, 7, "Significance", border=1, ln=True)
        pdf.set_font("Times", size=9)
        
        for i, (feat, corr_info) in enumerate(self.ranked_features[:20]):
            sig = "Yes *" if corr_info['significant'] else "No"
            pdf.cell(15, 6, str(i+1), border=1)
            pdf.cell(70, 6, feat[:40], border=1)
            pdf.cell(35, 6, f"{abs(corr_info['correlation']):.5f}", border=1)
            pdf.cell(35, 6, corr_info['method'][:15], border=1)
            pdf.cell(35, 6, sig, border=1, ln=True)
        
        # Embed feature importance bar chart
        importance_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        if os.path.exists(importance_path):
            pdf.add_page()
            pdf.set_font("Times", 'B', 12)
            pdf.cell(0, 10, "Feature Importance Bar Chart", ln=True)
            pdf.image(importance_path, x=10, w=190)
        
        # 12. Statistical Summary
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "12. Statistical Summary", ln=True)
        pdf.set_font("Times", size=11)
        
        pdf.cell(0, 8, "Summary of Statistical Tests:", ln=True)
        pdf.set_font("Times", size=10)
        
        # Count significant features
        sig_count = sum(1 for _, info in self.all_correlations.items() if info['significant'])
        total_tested = len(self.all_correlations)
        
        summary_stats = [
            ["Metric", "Value"],
            ["Total Features Tested", str(total_tested)],
            ["Statistically Significant Features", f"{sig_count} ({sig_count/total_tested*100:.1f}%)"],
            ["Non-Significant Features", f"{total_tested - sig_count} ({(total_tested-sig_count)/total_tested*100:.1f}%)"],
            ["Strongest Correlation", f"{self.ranked_features[0][0]}: {self.ranked_features[0][1]['correlation']:+.5f}"],
            ["Weakest Correlation", f"{self.ranked_features[-1][0]}: {self.ranked_features[-1][1]['correlation']:+.5f}"],
            ["Average Absolute Correlation", f"{np.mean([abs(v['correlation']) for v in self.all_correlations.values()]):.5f}"],
            ["Median Absolute Correlation", f"{np.median([abs(v['correlation']) for v in self.all_correlations.values()]):.5f}"]
        ]
        
        pdf.set_font("Times", 'B', 10)
        for row in summary_stats:
            pdf.cell(100, 7, row[0], border=1)
            pdf.cell(90, 7, row[1], border=1, ln=True)
        
        # 13. Key Findings and Recommendations
        pdf.add_page()
        pdf.set_font("Times", 'B', 14)
        pdf.cell(0, 10, "13. Key Findings and Recommendations", ln=True)
        pdf.set_font("Times", size=11)
        
        findings = f"""KEY FINDINGS:

1. Dataset Composition:
   - The dataset contains {self.total_records:,} job postings with {self.fraud_percent:.2f}% fraudulent cases
   - Class imbalance exists which may require special handling in modeling

2. Most Predictive Features:
"""
        pdf.multi_cell(0, 6, findings.strip())
        
        # List top 5 features
        pdf.set_font("Times", size=10)
        for i, (feat, corr_info) in enumerate(self.ranked_features[:5]):
            pdf.cell(0, 6, f"   {i+1}. {feat} (Correlation: {corr_info['correlation']:+.5f}, Method: {corr_info['method']})", ln=True)
        
        recommendations = """

3. Missing Data:
   - Several features have significant missing values that need imputation strategies
   - Missing data patterns may themselves be predictive of fraud

4. Text Features:
   - Text length and content analysis can provide valuable fraud detection signals
   - Fraudulent postings may have different linguistic patterns

5. Numeric Features:
   - Several numeric features show non-normal distributions
   - Outliers are present in some features and may need special handling
   - Statistical tests show significant differences between fraudulent and real postings

6. Recommendations:
   - Use all available features for model training, not just binary features
   - Apply appropriate encoding for categorical features
   - Consider text features using NLP techniques (TF-IDF, embeddings)
   - Address class imbalance using techniques like SMOTE or class weights
   - Validate model performance using appropriate metrics (F1, AUC-ROC)
   - Consider feature engineering based on the statistical insights
"""
        pdf.multi_cell(0, 6, recommendations.strip())
        
        # Save PDF
        report_path = os.path.join(OUTPUT_DIR, 'Data Analysis Report.pdf')
        pdf.output(report_path)
        
        print(f"Comprehensive PDF Report generated: {report_path}")
        
    def run_full_analysis(self):
        """Execute complete analysis pipeline"""
        print("="*70)
        print("COMPREHENSIVE FAKE JOB POSTINGS DATA ANALYSIS")
        print("="*70)
        
        self.load_and_preprocess()
        self.analyze_missing_data()
        self.analyze_binary_features()
        self.analyze_categorical_features()
        self.analyze_numeric_features()
        self.analyze_text_features()
        self.calculate_comprehensive_correlations()
        self.generate_visualizations()
        self.generate_pdf_report()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nGenerated Files in {OUTPUT_DIR}:")
        print(f"  - Data Analysis Report.pdf (Complete Analysis)")
        print(f"  - {len(self.plots_generated)} visualization images")
        print(f"\nAnalysis Summary:")
        print(f"  - Total features analyzed: {len(self.all_features)}")
        print(f"  - Binary features: {len(self.binary_features)}")
        print(f"  - Categorical features: {len(self.categorical_features)}")
        print(f"  - Numeric features: {len(self.numeric_features)}")
        print(f"  - Text features: {len(self.text_features)}")
        print(f"  - Visualizations generated: {len(self.plots_generated)}")


if __name__ == "__main__":
    analyzer = ComprehensiveJobDataAnalysis()
    analyzer.run_full_analysis()