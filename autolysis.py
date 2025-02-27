import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from scipy.stats import zscore

# Get the filename from command-line arguments
if len(sys.argv) < 2:
    print("Usage: uv run autolysis.py dataset.csv")
    sys.exit(1)

filename = sys.argv[1]

def load_data(file):
    """Load the CSV file into a DataFrame"""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def analyze_data(df):
    """Perform basic data analysis"""
    summary = df.describe().to_string()
    missing_values = df.isnull().sum().to_string()
    correlations = df.corr()
    return summary, missing_values, correlations

def detect_outliers(df):
    """Detect outliers using Z-score"""
    numeric_cols = df.select_dtypes(include=['number'])
    z_scores = numeric_cols.apply(zscore)
    outliers = (z_scores.abs() > 3).sum()
    return outliers.to_string()

def visualize_data(df):
    """Generate visualizations and save as PNG"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.savefig("correlation_matrix.png")
    
    for col in df.select_dtypes(include=['number']).columns[:2]:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{col}_distribution.png")
    
    categorical_cols = df.select_dtypes(include=['object']).columns[:2]
    for col in categorical_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f"Category distribution of {col}")
        plt.savefig(f"{col}_countplot.png")

def get_ai_insights(summary, missing_values, outliers, sample_data):
    """Send more meaningful analysis to GPT-4o-Mini"""
    client = openai.OpenAI(api_key=os.getenv("AIPROXY_TOKEN"))
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
    """Save the AI-generated analysis to README.md"""
    with open("README.md", "w") as f:
        f.write("# Data Analysis Report\n\n")
        f.write(analysis)

def save_results(dataset_name):
    """Move README and images to the correct directory"""
    os.makedirs(dataset_name, exist_ok=True)
    os.rename("README.md", f"{dataset_name}/README.md")
    for img in ["correlation_matrix.png", "col1_distribution.png", "col2_distribution.png", "col1_countplot.png", "col2_countplot.png"]:
        if os.path.exists(img):
            os.rename(img, f"{dataset_name}/{img}")

def main():
    df = load_data(filename)
    summary, missing_values, correlations = analyze_data(df)
    outliers = detect_outliers(df)
    visualize_data(df)
    insights = get_ai_insights(summary, missing_values, outliers, df)
    generate_readme(insights)
    
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    save_results(dataset_name)
    
    print("Analysis complete. Check README.md and PNG files.")

if __name__ == "__main__":
    main()
