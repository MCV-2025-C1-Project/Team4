import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your results from the CSV file
df = pd.read_csv('background_removal_grid_search_results.csv')
# df['threshold'] = df['name'].apply(lambda x: x.split('_')[0])
column_to_group = 'threshold'
# column_to_group = 'color_space'

# set str to the threshold column
df['threshold'] = df['threshold'].astype(str)
# Extract the base color space (e.g., 'HSV_all' becomes 'HSV')
# This helps in grouping different channel combinations under one color space

# Group by the color space and find the HIGHEST score achieved for each one
df_grouped = df.groupby(column_to_group)[['miou', 'f1_score']].max().reset_index()
# df_grouped = df.groupby('color_space')[['miou', 'f1_score']].max().reset_index()


# Sort the aggregated results for a cleaner plot
df_grouped_miou = df_grouped.sort_values('miou', ascending=False)
df_grouped_f1 = df_grouped.sort_values('f1_score', ascending=False)


# --- Plotting ---
# Create a figure with two side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f'Best Performance by {column_to_group}', fontsize=16)

# Plot for the best mIoU score in each color space group
sns.barplot(ax=axes[0], x='miou', y=column_to_group, data=df_grouped_miou, palette='YlGn')
axes[0].set_title(f'Best mIoU Score per {column_to_group}')
axes[0].set_xlabel('Mean Intersection over Union (mIoU)')
axes[0].set_ylabel(column_to_group)

# Plot for the best F1-Score in each color space group
sns.barplot(ax=axes[1], x='f1_score', y=column_to_group, data=df_grouped_f1, palette='YlGn')
axes[1].set_title(f'Best F1-Score per {column_to_group}')
axes[1].set_xlabel('F1-Score')
axes[1].set_ylabel(column_to_group) # Hide the y-axis label to avoid repetition


# Final adjustments and saving the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'performance_by_{column_to_group}.png')
plt.show()