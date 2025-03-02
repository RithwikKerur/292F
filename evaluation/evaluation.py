import pandas as pd
import matplotlib.pyplot as plt
import os 
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

def PassRateByModel():
    # Load data from CSV file
    data = pd.read_csv('evaluation/aggregated_topline.csv')

    problems = data['problem'].unique()
    models = data['model'].unique()
    prompt_types = data['prompt_type'].unique()

    # Create a graph for each problem
    for problem in problems:
        plt.figure(figsize=(10, 6))
        problem_data = data[data['problem'] == problem]
        jitter_strength = 0.1
        for i, prompt_type in enumerate(prompt_types):
            subset = problem_data[problem_data['prompt_type'] == prompt_type]
            jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(subset))
            plt.scatter(np.arange(len(subset)) + jitter, subset['pass_rate'], label=prompt_type)
        
        plt.xticks(np.arange(len(models)), models, rotation=45)
        plt.title(f'Pass Rate by Model for {problem}')
        plt.xlabel('Model')
        plt.ylabel('Pass Rate (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'evaluation/Pass_Rate_by_Model_for_{problem}')

def ErrorRateByModel():
# Load data from CSV file
    data = pd.read_csv('evaluation/aggregated_topline.csv')

    problems = data['problem'].unique()
    models = data['model'].unique()
    prompt_types = data['prompt_type'].unique()

    # Create a graph for each problem
    for problem in problems:
        plt.figure(figsize=(10, 6))
        problem_data = data[data['problem'] == problem]
        jitter_strength = 0.1
        for i, prompt_type in enumerate(prompt_types):
            subset = problem_data[problem_data['prompt_type'] == prompt_type]
            jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(subset))
            plt.scatter(np.arange(len(subset)) + jitter, subset['avg_error_count'], label=prompt_type)
        
        plt.xticks(np.arange(len(models)), models, rotation=45)
        plt.title(f'Average Error Count by Model for {problem}')
        plt.xlabel('Model')
        plt.ylabel('Average Error Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'evaluation/Average_Error_Count_by_Model_for_{problem}')

def ConstraintsPassed():
    metric = 'avg_structural_passed'
    title = 'Average Structural Constraints Passed'
    # Get unique problems, models, and prompt types
    data = pd.read_csv('evaluation/aggregated_topline.csv')

    # Group by model and prompt_type and calculate the average pass rate for each combination
    avg_data = data.groupby(['model', 'prompt_type'], as_index=False)[metric].mean()

    # Pivot the data to create a matrix for the heatmap
    heatmap_data = avg_data.pivot(index='model', columns='prompt_type', values=metric)

    # Create the heatmap for the average pass rates
    plt.figure(figsize=(8, 6))
    
    norm = plt.Normalize(vmin=1, vmax=3)  # Narrower range for smaller color variation
    plt.imshow(heatmap_data, cmap='Blues', aspect='auto', norm=norm)  # Light blue to dark blue colormap
    plt.colorbar()

    plt.xticks(ticks=np.arange(len(heatmap_data.columns)), labels=heatmap_data.columns, rotation=45)
    plt.yticks(ticks=np.arange(len(heatmap_data.index)), labels=heatmap_data.index)
    
    # Annotate each cell with the pass rate value
    for (i, j), val in np.ndenumerate(heatmap_data.values):
        plt.text(j, i, f'{val:.1f}', ha='center', va='center', color='black')

    plt.title(title)
    plt.xlabel('Prompt Type')
    plt.ylabel('Model')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"evaluation/{title}")

if __name__ == '__main__':
    ConstraintsPassed()