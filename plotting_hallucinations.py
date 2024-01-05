import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

file_path = 'logic_scores.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Define the columns to calculate the averages for
columns_to_average = ['OR Similarity Score', 'AND Similarity Score', 
                      'NOT XOR Similarity Score', 'NOT AND Similarity Score']

# Calculating the average scores for different sections of the DataFrame
generic_response = df.iloc[1:41][columns_to_average].mean()
extrinsic_hallucinations = df.iloc[119:145][columns_to_average].mean()
intrinsic_hallucinations = df.iloc[179:187][columns_to_average].mean()

# Create a DataFrame to hold these averages for easier plotting
averages_df = pd.DataFrame({
    'Generic Response': generic_response,
    'Extrinsic Hallucinations': extrinsic_hallucinations,
    'Intrinsic Hallucinations': intrinsic_hallucinations
})

# Preparing data for clustering scatter points around each prompt type
scatter_data = averages_df.T.reset_index()
scatter_data = scatter_data.rename(columns={'index': 'Prompt Type'})

# Melting the DataFrame for seaborn plotting
melted_scatter_data = scatter_data.melt(id_vars='Prompt Type', var_name='Gate Type', value_name='Average Score')

# Adding a small offset to x-coordinates for clustering
offset = 0.1  # Offset for each gate type
scatter_data_expanded = melted_scatter_data.copy()
gate_type_offsets = {'OR Similarity Score': -1.5 * offset, 
                     'AND Similarity Score': -0.5 * offset, 
                     'NOT XOR Similarity Score': 0.5 * offset, 
                     'NOT AND Similarity Score': 1.5 * offset}

# Apply offset to each gate type to create clusters within each prompt type
scatter_data_expanded['x_offset'] = scatter_data_expanded['Gate Type'].map(gate_type_offsets)
scatter_data_expanded['Prompt Type'] = pd.Categorical(scatter_data_expanded['Prompt Type'], 
                                                    ["Generic Response", "Extrinsic Hallucinations", "Intrinsic Hallucinations"])
scatter_data_expanded['x_value'] = scatter_data_expanded['Prompt Type'].cat.codes + scatter_data_expanded['x_offset']

# Plotting the averages using a scatter plot with connected lines within each category
plt.figure(figsize=(12, 6))

# Loop through each prompt type to plot and connect points
for prompt_type in scatter_data_expanded['Prompt Type'].cat.categories:
    subset = scatter_data_expanded[scatter_data_expanded['Prompt Type'] == prompt_type]
    sns.scatterplot(data=subset, x='x_value', y='Average Score', hue='Gate Type', style='Gate Type', s=100, legend=False)
    plt.plot(subset['x_value'], subset['Average Score'], linestyle='--', alpha=1)

# Adding custom legend entries
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='OR Similarity Score', markersize=10, markerfacecolor='b'),
    plt.Line2D([0], [0], marker='X', color='w', label='AND Similarity Score', markersize=10, markerfacecolor='orange'),
    plt.Line2D([0], [0], marker='^', color='w', label='NOT XOR Similarity Score', markersize=10, markerfacecolor='g'),
    plt.Line2D([0], [0], marker='s', color='w', label='NOT AND Similarity Score', markersize=10, markerfacecolor='r'),
]

# Adding the legend back, but only once
plt.legend(handles=legend_elements, title='Gate Type', loc='upper left', bbox_to_anchor=(0.75, 1.05))

plt.title('')
plt.ylabel('Average Score')
plt.xlabel('Response Type')
plt.xticks(ticks=[0, 1, 2], labels=["Generic Response", "Extrinsic Hallucinations", "Intrinsic Hallucinations"])
plt.grid(True)
plt.show()
