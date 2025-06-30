import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# Constants
BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
VIS_OUTPUT_DIR = os.path.join(RESULTS_DIR, "visualizations")

# Ensure output directory exists
os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

def get_latest_results():
    """Get the most recent recall results CSV file"""
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "recall_results_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("No recall evaluation results found. Run recall_eval.py first.")
    
    # Sort by modification time (newest first)
    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"Loading results from: {os.path.basename(latest_file)}")
    
    results_df = pd.read_csv(latest_file)
    
    # Check if Average row exists, if not, calculate and add it
    if 'Average' not in results_df['Category'].values:
        # Get recall columns
        recall_cols = [col for col in results_df.columns if col.startswith('Recall@')]
        
        # Calculate averages
        avg_values = {'Category': 'Average'}
        for col in recall_cols:
            avg_values[col] = results_df[col].mean()
        
        # Add average row to DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([avg_values])], ignore_index=True)
        print("Added Average row to results")
    
    return results_df

def create_recall_line_plot(results_df, output_filename="recall_line_plot.png"):
    """Create line plot showing Recall@k for different categories"""
    
    # Filter out the Average row for category lines
    category_df = results_df[results_df['Category'] != 'Average']
    
    # Get k values from column names (Recall@5, Recall@10, etc.)
    k_columns = [col for col in results_df.columns if col.startswith('Recall@')]
    k_values = [int(col.split('@')[1]) for col in k_columns]
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Plot lines for each category
    for _, row in category_df.iterrows():
        category = row['Category']
        recall_values = [row[f'Recall@{k}'] for k in k_values]
        plt.plot(k_values, recall_values, marker='o', linewidth=2, markersize=8, label=category)
    
    # Check if Average row exists and add it
    avg_rows = results_df[results_df['Category'] == 'Average']
    if not avg_rows.empty:
        avg_row = avg_rows.iloc[0]
        avg_values = [avg_row[f'Recall@{k}'] for k in k_values]
        plt.plot(k_values, avg_values, 'k--', marker='D', linewidth=2.5, markersize=10, label='Average')
    else:
        # Calculate average on the fly
        avg_values = [category_df[f'Recall@{k}'].mean() for k in k_values]
        plt.plot(k_values, avg_values, 'k--', marker='D', linewidth=2.5, markersize=10, label='Average')
    
    # Add labels and styling
    plt.title('Recall@k by Category', fontsize=16)
    plt.xlabel('k (Number of Recommendations)', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.xticks(k_values, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    # Add horizontal lines at 0.25, 0.50, 0.75 intervals
    for y in [0.25, 0.50, 0.75]:
        plt.axhline(y=y, color='gray', linestyle=':', alpha=0.5)
    
    # Set y-axis limits with a little padding
    plt.ylim(-0.02, 1.02)
    
    # Save the plot
    output_path = os.path.join(VIS_OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Line plot saved to {output_path}")
    
    return output_path

def create_recall_bar_chart(results_df, k=10, output_filename=None):
    """Create bar chart comparing Recall@k across categories"""
    if output_filename is None:
        output_filename = f"recall_at_{k}_bar_chart.png"
    
    # Filter out the Average row
    category_df = results_df[results_df['Category'] != 'Average']
    
    # Get recall values for the specified k
    recall_column = f'Recall@{k}'
    if recall_column not in results_df.columns:
        raise ValueError(f"No data for {recall_column}")
    
    # Sort by recall value (descending)
    sorted_df = category_df.sort_values(recall_column, ascending=False)
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Create a custom colormap for bars
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted_df)))
    
    # Create the bar chart
    bars = plt.bar(
        sorted_df['Category'], 
        sorted_df[recall_column], 
        color=colors,
        width=0.6
    )
    
    # Calculate average and add line
    avg_value = sorted_df[recall_column].mean()
    plt.axhline(y=avg_value, color='red', linestyle='--', 
                label=f'Average: {avg_value:.2f}')
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{height:.2f}',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )
    
    # Add labels and styling
    plt.title(f'Recall@{k} by Category', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.xticks(fontsize=12, rotation=30, ha='right')
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Set y-axis limits with padding
    plt.ylim(0, min(1.1, max(sorted_df[recall_column]) * 1.2))
    
    # Save the plot
    output_path = os.path.join(VIS_OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar chart saved to {output_path}")
    
    return output_path

def create_heatmap(results_df, output_filename="recall_heatmap.png"):
    """Create heatmap showing recall across categories and k values"""
    
    # Filter out the Average row
    category_df = results_df[results_df['Category'] != 'Average']
    
    # Get k values from column names
    k_columns = [col for col in results_df.columns if col.startswith('Recall@')]
    
    # Prepare data for heatmap
    heatmap_data = category_df.set_index('Category')[k_columns]
    
    # Set up the plot
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap - blue to yellow to red
    custom_cmap = LinearSegmentedColormap.from_list(
        'blue_yellow_red', 
        ['#2c7bb6', '#ffffbf', '#d7191c']
    )
    
    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap=custom_cmap,
        linewidths=0.5,
        fmt='.2f',
        vmin=0, 
        vmax=1,
        cbar_kws={'label': 'Recall Value'}
    )
    
    # Add labels and styling
    plt.title('Recall Heatmap by Category and k', fontsize=16)
    plt.xlabel('k Value', fontsize=14)
    plt.ylabel('Category', fontsize=14)
    
    # Adjust x-axis labels to show just the k value
    new_labels = [col.split('@')[1] for col in k_columns]
    ax.set_xticklabels(new_labels, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the plot
    output_path = os.path.join(VIS_OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")
    
    return output_path

def create_dashboard(results_df, output_filename="recall_dashboard.png"):
    """Create a comprehensive visualization dashboard"""
    
    # Set up the figure with GridSpec for custom layout
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 0.8], width_ratios=[1.2, 1])
    
    # Get k values from column names
    k_columns = [col for col in results_df.columns if col.startswith('Recall@')]
    k_values = [int(col.split('@')[1]) for col in k_columns]
    
    # Filter for category data (exclude Average)
    category_df = results_df[results_df['Category'] != 'Average']
    
    # 1. Line Plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot lines for each category
    for _, row in category_df.iterrows():
        category = row['Category']
        recall_values = [row[f'Recall@{k}'] for k in k_values]
        ax1.plot(k_values, recall_values, marker='o', linewidth=2, markersize=8, label=category)
    
    # Calculate and plot average
    avg_values = [category_df[f'Recall@{k}'].mean() for k in k_values]
    ax1.plot(k_values, avg_values, 'k--', marker='D', linewidth=2.5, markersize=10, label='Average')
    
    ax1.set_title('Recall@k by Category', fontsize=14)
    ax1.set_xlabel('k (Number of Recommendations)', fontsize=12)
    ax1.set_ylabel('Recall', fontsize=12)
    ax1.set_xticks(k_values)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=10, loc='best')
    
    # 2. Bar Chart (top right) - Recall@10
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Sort by Recall@10 value
    sorted_df = category_df.sort_values('Recall@10', ascending=False)
    
    # Create bar chart
    bars = ax2.bar(sorted_df['Category'], sorted_df['Recall@10'], color='skyblue')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add average line
    avg_value = category_df['Recall@10'].mean()
    ax2.axhline(y=avg_value, color='red', linestyle='--', label=f'Average: {avg_value:.2f}')
    
    ax2.set_title('Recall@10 by Category', fontsize=14)
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.set_ylim(0, min(1.1, max(sorted_df['Recall@10']) * 1.2))
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.set_xticklabels(sorted_df['Category'], rotation=30, ha='right')
    ax2.legend(fontsize=10)
    
    # 3. Heatmap (bottom spanning both columns)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Create heatmap data
    heatmap_data = category_df.set_index('Category')[k_columns]
    
    # Create custom colormap - blue to yellow to red
    custom_cmap = LinearSegmentedColormap.from_list(
        'blue_yellow_red', 
        ['#2c7bb6', '#ffffbf', '#d7191c']
    )
    
    # Create heatmap
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap=custom_cmap,
        linewidths=0.5,
        fmt='.2f',
        vmin=0, 
        vmax=1,
        cbar_kws={'label': 'Recall Value'},
        ax=ax3
    )
    
    ax3.set_title('Recall Heatmap by Category and k', fontsize=14)
    ax3.set_xlabel('k Value', fontsize=12)
    ax3.set_ylabel('Category', fontsize=12)
    
    # Adjust x-axis labels to show just the k value
    new_labels = [col.split('@')[1] for col in k_columns]
    ax3.set_xticklabels(new_labels, fontsize=10)
    
    # Add main title
    plt.suptitle('Recall Evaluation Dashboard', fontsize=20, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the dashboard
    output_path = os.path.join(VIS_OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dashboard saved to {output_path}")
    
    return output_path

def create_summary_table(results_df, output_filename="recall_summary_table.png"):
    """Create a graphical summary table of recall results"""
    
    # Create a copy for manipulation
    table_data = results_df.copy()
    
    # Get k values for column headers
    k_columns = [col for col in table_data.columns if col.startswith('Recall@')]
    
    # Check if Average row exists, if not, calculate and add it
    if 'Average' not in table_data['Category'].values:
        avg_values = {'Category': 'Average'}
        for col in k_columns:
            avg_values[col] = table_data[table_data['Category'] != 'Average'][col].mean()
        table_data = pd.concat([table_data, pd.DataFrame([avg_values])], ignore_index=True)
    
    # Create a figure and axis
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Format all recall values to 3 decimal places
    for col in k_columns:
        table_data[col] = table_data[col].apply(lambda x: f"{x:.3f}")
    
    # Create the table
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.2] + [0.15] * len(k_columns)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)  # Adjust row height
    
    # Style header row
    for j, cell in enumerate(table._cells[(0, j)] for j in range(len(table_data.columns))):
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4472C4')
    
    # Find and highlight the average row with different color
    for i, category in enumerate(table_data['Category']):
        if category == 'Average':
            avg_row_idx = i + 1  # +1 because of the header row
            for j in range(len(table_data.columns)):
                cell = table._cells[(avg_row_idx, j)]
                cell.set_facecolor('#FFC000')
                cell.set_text_props(weight='bold')
            break
    
    # Color coding cells based on recall value
    for i in range(1, len(table_data) + 1):
        if i == avg_row_idx:  # Skip the average row as we already colored it
            continue
            
        for j, col in enumerate(k_columns):
            cell = table._cells[(i, j + 1)]  # +1 because first column is Category
            value = float(table_data.iloc[i-1][col])
            
            # Color gradient based on value
            if value < 0.25:
                color = '#F8696B'  # Red
            elif value < 0.50:
                color = '#FFEB84'  # Yellow
            elif value < 0.75:
                color = '#63BE7B'  # Green
            else:
                color = '#00FF00'  # Bright green
                
            cell.set_facecolor(color)
    
    # Add title
    plt.title('Recall Evaluation Summary', fontsize=16, pad=20)
    
    # Save the table
    output_path = os.path.join(VIS_OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary table saved to {output_path}")
    
    return output_path

def main():
    """Main function to create all visualizations"""
    
    try:
        # Get the results data
        results_df = get_latest_results()
        
        # Create all visualizations
        create_recall_line_plot(results_df)
        
        # Get k values from column names
        k_values = [int(col.split('@')[1]) for col in results_df.columns 
                    if col.startswith('Recall@')]
        
        # Create bar charts for each k value
        for k in k_values:
            create_recall_bar_chart(results_df, k=k)
        
        # Create heatmap
        create_heatmap(results_df)
        
        # Create summary table
        create_summary_table(results_df)
        
        # Create comprehensive dashboard
        create_dashboard(results_df)
        
        print(f"\nAll visualizations saved to {VIS_OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()