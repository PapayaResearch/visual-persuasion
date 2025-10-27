import os
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import defaultdict

class EvaluationAnalyzer:
    """
    Analyzes evaluation results and generates statistics and visualizations.
    """
    def __init__(self, num_previews: int):
        self.num_previews = num_previews
        self.colors = {
            'older': '#7fbf7f',      # Green for older image chosen
            'newer': '#ff7f7f',      # Red for newer image chosen
            'neutral': '#7f9fff'     # Blue for neutral/mixed content
        }

    def plot_class_heatmap(self, class_name: str, class_df: pd.DataFrame, output_dir: str) -> str:
        """Generate a heatmap for a single image class."""
        all_bases = sorted(set(class_df['base1'].unique()) | set(class_df['base2'].unique()))
        n = len(all_bases)

        # Create comparison matrix
        matrix = np.full((n, n), np.nan)

        for _, row in class_df.iterrows():
            base1, base2, choice = row['base1'], row['base2'], row['choice']
            idx1, idx2 = all_bases.index(base1), all_bases.index(base2)

            if idx1 < idx2:
                matrix[idx2, idx1] = 1 if choice == base1 else 0
            else:
                matrix[idx1, idx2] = 1 if choice == base2 else 0

        # Create figure
        fig, ax_heat = plt.subplots(figsize=(max(n * 0.8, 8), max(n * 0.8, 8)))

        cmap = sns.color_palette([self.colors['older'], "#e0e0e0", self.colors['newer']], as_cmap=True)
        mask = np.triu(np.ones_like(matrix, dtype=bool))

        sns.heatmap(matrix, mask=mask, cmap=cmap, linewidths=0.5, vmin=0, vmax=1, square=True,
                    xticklabels=all_bases, yticklabels=all_bases,
                    cbar_kws={'label': 'Choice', 'ticks': [0, 1]}, ax=ax_heat)

        ax_heat.collections[0].colorbar.set_ticklabels(['Newer', 'Older'])
        ax_heat.set_title(f'Comparison Results for {class_name}', fontsize=16, pad=20, fontweight='bold')

        save_path = os.path.join(output_dir, f'{class_name}_heatmap.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_class_preview(self, class_name: str, class_df: pd.DataFrame, image_dir: str, output_dir: str) -> str:
        """Generate a preview image for a single image class."""
        # Collect all unique filenames from class_df
        unique_filenames = set()
        for _, row in class_df.iterrows():
            filename1 = f"{class_name}_{row['base1']}"
            filename2 = f"{class_name}_{row['base2']}"
            unique_filenames.add(filename1)
            unique_filenames.add(filename2)
        
        # Select preview images
        num_preview_images = min(self.num_previews, len(unique_filenames)) if self.num_previews != -1 else len(unique_filenames)
        selected_filenames = sorted(random.sample(sorted(unique_filenames), num_preview_images))
        
        # Create grid layout
        cols = min(5, len(selected_filenames))
        rows = (len(selected_filenames) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.suptitle(f'Image Preview for {class_name}', fontsize=16, fontweight='bold')

        # Ensure 2D array
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        # Fill grid
        for idx, filename in enumerate(selected_filenames):
            row_idx = idx // cols
            col_idx = idx % cols
            ax = axes[row_idx, col_idx]

            # Try common image extensions
            img_loaded = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                img_path = os.path.join(image_dir, filename + ext)
                if os.path.exists(img_path):
                    ax.imshow(Image.open(img_path))
                    ax.set_title('_'.join(filename.split('_')[1:]), fontsize=10)
                    img_loaded = True
                    break
            
            if not img_loaded:
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title('_'.join(filename.split('_')[1:]), fontsize=10)

            ax.axis('off')

        # Hide unused subplots
        for idx in range(len(selected_filenames), rows * cols):
            row_idx = idx // cols
            col_idx = idx % cols
            axes[row_idx, col_idx].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(output_dir, f'{class_name}_preview.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_class_summary(self, class_name: str, class_df: pd.DataFrame, output_dir: str) -> str:
        """Generate summary bar chart showing win rate for a single image class."""
        # Calculate win rates for this class
        base_wins = defaultdict(int)
        base_total = defaultdict(int)
        
        for _, row in class_df.iterrows():
            base1, base2, choice = row['base1'], row['base2'], row['choice']
            
            base_total[base1] += 1
            base_total[base2] += 1
            
            if choice == base1:
                base_wins[base1] += 1
            elif choice == base2:
                base_wins[base2] += 1
        
        # Calculate win rates
        win_rates = {base: (base_wins[base] / base_total[base] * 100) if base_total[base] > 0 else 0 
                     for base in base_total}
        
        win_rate_series = pd.Series(win_rates).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(range(len(win_rate_series)), win_rate_series.values, color=self.colors['neutral'])

        ax.set_title(f'Win Rates for {class_name}', fontsize=16, pad=20, fontweight='bold')
        ax.set_ylabel('Win Rate (%)', fontsize=14)
        ax.set_xticks(range(len(win_rate_series)))
        ax.set_xticklabels(win_rate_series.index, rotation=45, ha='right')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, win_rate_series.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 2,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=11)

        save_path = os.path.join(output_dir, f'{class_name}_summary.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        return save_path

    def run(self, csv_path: str) -> str:
        """Generate analysis dashboard from CSV file."""
        # Load CSV data
        df = pd.read_csv(csv_path, encoding='utf-8')
        logging.info(f"Loaded {len(df)} evaluation records from CSV\n")

        # Determine image directory (parent of CSV location)
        output_dir = os.path.dirname(csv_path)
        image_dir = os.path.dirname(os.path.dirname(os.path.dirname(output_dir)))

        # Generate per-class plots
        for class_name in df['image_class'].unique():
            class_df = df[df['image_class'] == class_name]
            self.plot_class_heatmap(class_name, class_df, output_dir)
            self.plot_class_preview(class_name, class_df, image_dir, output_dir)
            self.plot_class_summary(class_name, class_df, output_dir)
            logging.info(f"Generated analysis plots for class: {class_name}\n")