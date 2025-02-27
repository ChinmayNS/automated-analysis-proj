import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai

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

def get_ai_insights(summary, missing_values):
    """Send analysis to GPT-4o-Mini and get insights"""
    client = openai.OpenAI(api_key=os.getenv("AIPROXY_TOKEN"))
    prompt = f"Analyze this dataset:\nSummary:\n{summary}\nMissing Values:\n{missing_values}"
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

def main():
    df = load_data(filename)
    summary, missing_values, correlations = analyze_data(df)
    visualize_data(df)
    insights = get_ai_insights(summary, missing_values)
    generate_readme(insights)
    print("Analysis complete. Check README.md and PNG files.")

if __name__ == "__main__":
    main()
