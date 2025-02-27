import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from scipy.stats import zscore

if len(sys.argv) < 2:
    print("Usage: uv run autolysis.py dataset.csv")
    sys.exit(1)

filename = sys.argv[1]

def load_data(file):
    data_path = os.path.join("data", file) 
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        print(f"Error loading dataset from {data_path}: {e}")
        sys.exit(1)

def analyze_data(df):
    summary = df.describe(include='all').to_string()
    missing_values = df.isnull().sum().to_string()
    
    numeric_df = df.select_dtypes(include=['number'])
    correlations = numeric_df.corr()
    
    outliers = detect_outliers(numeric_df)
    return summary, missing_values, correlations, outliers

def detect_outliers(df):
    numeric_cols = df.select_dtypes(include=['number'])
    z_scores = numeric_cols.apply(zscore)
    outliers = (z_scores.abs() > 3).sum()
    return outliers.to_string()

def visualize_data(df, dataset_name):
    os.makedirs(dataset_name, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm")
    plt.savefig(f"{dataset_name}/correlation_matrix.png")
    
    for col in df.select_dtypes(include=['number']).columns[:2]:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{dataset_name}/{col}_distribution.png")
    
    categorical_cols = df.select_dtypes(include=['object']).columns[:2]
    for col in categorical_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f"Category distribution of {col}")
        plt.savefig(f"{dataset_name}/{col}_countplot.png")

def get_ai_insights(summary, missing_values, outliers, sample_data):
    client = openai.OpenAI(api_key=os.getenv("sk-1234efgh5678ijkl1234efgh5678ijkl1234efgh"))
    prompt = (
        f"Analyze this dataset:\n\n"
        f"Column Names: {sample_data.columns.tolist()}\n\n"
        f"Summary Stats:\n{summary}\n\n"
        f"Missing Values:\n{missing_values}\n\n"
        f"Outliers Detected:\n{outliers}\n\n"
        f"First Few Rows:\n{sample_data.head().to_string()}\n\n"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_readme(analysis):
    with open("README.md", "w") as f:
        f.write("# Data Analysis Report\n\n")
        f.write(analysis)

def save_results(dataset_name):
    os.makedirs(dataset_name, exist_ok=True)
    os.rename("README.md", f"{dataset_name}/README.md")

def main():
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    df = load_data(filename)
    summary, missing_values, correlations, outliers = analyze_data(df)
    visualize_data(df, dataset_name)
    insights = get_ai_insights(summary, missing_values, outliers, df)
    generate_readme(insights)
    save_results(dataset_name)
    print(f"Analysis complete. Check {dataset_name}/README.md and PNG files.")

if __name__ == "__main__":
    main()
