import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_data(file_path):
    """Reads CSV data into a DataFrame."""
    return pd.read_csv(file_path)

def calculate_row_and_category_averages(df, row_ranges, columns_to_average):
    """Calculates the average score 'k' for each row and the category averages."""
    category_averages = {}
    row_averages_data = []

    for category, (start, end) in row_ranges.items():
        category_data = df.iloc[start:end][columns_to_average]
        row_averages = category_data.mean(axis=1)  # Average across columns for each row
        category_averages[category] = row_averages.mean()  # Average of row averages
        for i in range(start, end):
            k = df.iloc[i][columns_to_average].mean()
            row_averages_data.append((category, i, k))
    return pd.DataFrame(category_averages, index=['Category Average']), pd.DataFrame(row_averages_data, columns=['Category', 'Row Index', 'k'])

def plot_data(category_averages, row_averages_data, categories, columns_to_average):
    """Plots category averages and individual row averages with horizontal lines."""
    plt.figure(figsize=(7, 3.5))

    gate_type_offsets = {'OR Similarity Score': -0.3, 'AND Similarity Score': -0.1, 'NOT XOR Similarity Score': 0.1, 'NOT AND Similarity Score': 0.3}
    category_colors = {'Generic': 'orange', 'Intrinsic': 'lightblue', 'Extrinsic': 'lightgray'}
    category_avg_colors = {'Generic': 'red', 'Intrinsic': 'darkblue', 'Extrinsic': 'darkgray'}

    # Plotting individual row data with horizontal lines
    for _, row in row_averages_data.iterrows():
        row_x_values, row_y_values = [], []
        for gate in columns_to_average:
            x_value = categories.index(row['Category']) + gate_type_offsets[gate]
            y_value = row['k']
            row_x_values.append(x_value)
            row_y_values.append(y_value)
            plt.scatter(x_value, y_value, color=category_colors[row['Category']], alpha=0.5)
        plt.plot(row_x_values, row_y_values, linestyle='-', color=category_colors[row['Category']], alpha=0.5)

    # Plotting category averages
    for category in categories:
        avg_x_values, avg_y_values = [], []
        for gate, offset in gate_type_offsets.items():
            avg_x_values.append(categories.index(category) + offset)
            avg_y_values.append(category_averages[category])
        plt.plot(avg_x_values, avg_y_values, linestyle='-', color=category_avg_colors[category], alpha=1.0)

    # Setting plot details
    plt.ylabel('Average Score')
    plt.xlabel('Prompt Type')
    plt.xticks(ticks=range(len(categories)), labels=categories)
    plt.ylim(bottom=None, top=1)
    plt.grid(True)
    plt.show()

# Constants
file_path = 'logic_scores.csv'
row_ranges = {'Generic': (1, 12), 'Extrinsic': (119, 127), 'Intrinsic': (176, 186)}
columns_to_average = ['OR Similarity Score', 'AND Similarity Score', 'NOT XOR Similarity Score', 'NOT AND Similarity Score']
categories = ["Generic", "Extrinsic", "Intrinsic"]

# Main execution
df = read_data(file_path)
category_averages, row_averages_data = calculate_row_and_category_averages(df, row_ranges, columns_to_average)
plot_data(category_averages.iloc[0], row_averages_data, categories, columns_to_average)

