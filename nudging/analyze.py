import os
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Dict
from collections import defaultdict

class EvaluationAnalyzer:
    """
    Analyzes evaluation results and generates statistics and visualizations.
    """
    def __init__(self, num_previews: int):
        self.df = None
        self.num_previews = num_previews
        self.image_dir = None
        self.colors = {
            'older': '#7fbf7f',      # Green for older image chosen
            'newer': '#ff7f7f',      # Red for newer image chosen
            'neutral': '#7f9fff'     # Blue for neutral/mixed content
        }
    
    def generate_statistics(self) -> Dict:
        """Generate key statistics from the evaluation data."""
        stats = {}
        # Get all unique bases
        all_bases = sorted(set(self.df['base1'].unique()) | set(self.df['base2'].unique()))
        # Count wins and comparisons for each base
        base_wins = defaultdict(int)
        base_comparisons = defaultdict(int)
        
        for _, row in self.df.iterrows():
            base1, base2, choice = row['base1'], row['base2'], row['choice']
            base_comparisons[base1] += 1
            base_comparisons[base2] += 1
            if choice == base1:
                base_wins[base1] += 1
            elif choice == base2:
                base_wins[base2] += 1
        
        # Calculate win rates
        base_win_rates = {
            base: (base_wins[base] / base_comparisons[base] * 100) if base_comparisons[base] > 0 else 0
            for base in all_bases
        }
        
        stats['total_comparisons'] = len(self.df)
        stats['base_win_rates'] = base_win_rates
        stats['base_wins'] = dict(base_wins)
        stats['base_comparisons'] = dict(base_comparisons)
        
        # Per-class statistics
        class_stats = {}
        for class_name in self.df['image_class'].unique():
            class_df = self.df[self.df['image_class'] == class_name]
            class_base_wins = defaultdict(int)
            for _, row in class_df.iterrows():
                if row['choice'] == row['base1']:
                    class_base_wins[row['base1']] += 1
                elif row['choice'] == row['base2']:
                    class_base_wins[row['base2']] += 1
            class_stats[class_name] = dict(class_base_wins)
        
        stats['per_class'] = class_stats
        return stats
    
    def plot_class_heatmap(self, class_name: str, class_df: pd.DataFrame, output_dir: str) -> str:
        """Generate a heatmap for a single image class with preview images."""
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
        
        # Get image files for this class
        image_files = self._get_image_files_for_class(class_name)
        
        # Create figure
        fig_width = max(len(all_bases) * 3, 12)
        fig = plt.figure(figsize=(fig_width, fig_width * 0.8))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Heatmap
        heatmap_ratio = n / len(all_bases)
        margin = (1 - heatmap_ratio) / 2
        heatmap_gs = gs[0].subgridspec(1, 3, width_ratios=[margin, heatmap_ratio, margin])
        ax_heat = fig.add_subplot(heatmap_gs[1])
        
        cmap = sns.color_palette([self.colors['older'], "#e0e0e0", self.colors['newer']], as_cmap=True)
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        
        sns.heatmap(matrix, mask=mask, cmap=cmap, linewidths=0.5, vmin=0, vmax=1, square=True,
                    xticklabels=all_bases, yticklabels=all_bases,
                    cbar_kws={'label': 'Choice', 'ticks': [0, 1]}, ax=ax_heat)
        
        ax_heat.collections[0].colorbar.set_ticklabels(['Newer', 'Older'])
        ax_heat.set_title(f'Comparison Results for {class_name}', fontsize=14, pad=20)
        ax_heat.set_xlabel('Base', fontsize=12)
        ax_heat.set_ylabel('Base', fontsize=12)
        
        # Preview images
        preview_gs = gs[1].subgridspec(1, len(all_bases), wspace=0.1)
        
        for idx, base_type in enumerate(all_bases):
            ax_img = fig.add_subplot(preview_gs[idx])
            
            if base_type in image_files:
                img_path = os.path.join(self.image_dir, image_files[base_type])
                ax_img.imshow(Image.open(img_path))
            else:
                ax_img.text(0.5, 0.5, 'Missing', ha='center', va='center',
                           transform=ax_img.transAxes, fontsize=10, color='gray')
            
            ax_img.set_title(base_type, fontsize=10, pad=5)
            ax_img.axis('off')
        
        save_path = os.path.join(output_dir, f'{class_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _get_image_files_for_class(self, class_name: str) -> Dict[str, str]:
        """Get mapping of base type to filename for a specific class."""
        image_files = {}
        
        for file in os.listdir(self.image_dir):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                continue
            name_without_ext = os.path.splitext(file)[0]
            split = name_without_ext.split('_')
            img_class = '_'.join(split[:-1])
            base = split[-1]
            if img_class == class_name:
                image_files[base] = file
        
        return image_files
    
    def generate_preview(self, output_dir: str) -> str:
        """Generate a preview image showing sample images from different classes and base types."""
        analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        all_classes = sorted(self.df['image_class'].unique())
        all_bases = sorted(set(self.df['base1'].unique()) | set(self.df['base2'].unique()))
        
        num_preview_classes = min(self.num_previews, len(all_classes)) if self.num_previews != -1 else len(all_classes)
        
        random.shuffle(all_classes)
        selected_classes = sorted(all_classes[:num_preview_classes])
        
        rows, cols = len(selected_classes), len(all_bases)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.suptitle('Image Preview by Class and Base Type', fontsize=16, fontweight='bold')
        
        # Ensure 2D array
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Get all image files
        image_files = defaultdict(dict)
        for file in os.listdir(self.image_dir):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                continue
            
            name_without_ext = os.path.splitext(file)[0]
            split = name_without_ext.split('_')
            image_class = '_'.join(split[:-1])
            base = split[-1]
            image_files[image_class][base] = file
        
        # Fill grid
        for row_idx, class_name in enumerate(selected_classes):
            for col_idx, base_type in enumerate(all_bases):
                ax = axes[row_idx, col_idx]
                
                if class_name in image_files and base_type in image_files[class_name]:
                    img_path = os.path.join(self.image_dir, image_files[class_name][base_type])
                    ax.imshow(Image.open(img_path))
                    ax.set_title(f"{base_type}", fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'Missing', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f"{base_type}", fontsize=10)
                
                ax.axis('off')
                
                if col_idx == 0:
                    ax.set_ylabel(f"{class_name}", fontsize=10, rotation=90, labelpad=20, va='center')
        
        # Column labels
        for col_idx, base_type in enumerate(all_bases):
            axes[0, col_idx].set_xlabel(f"{base_type}", fontsize=12, labelpad=10)
            axes[0, col_idx].xaxis.set_label_position('top')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(analysis_dir, 'preview.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path
    
    def plot_heatmap(self, output_dir: str, stats: Dict) -> str:
        """Generate heatmap showing wins by class and base type with percentages."""
        analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        all_bases = sorted(stats['base_win_rates'].keys())
        class_names = sorted(stats['per_class'].keys())
        
        # Create annotation data with percentages
        annot_data = []
        for class_name in class_names:
            class_df = self.df[self.df['image_class'] == class_name]
            row = []
            for base in all_bases:
                wins = stats['per_class'][class_name].get(base, 0)
                total = len(class_df[(class_df['base1'] == base) | (class_df['base2'] == base)])
                percentage = (wins / total * 100) if total > 0 else 0
                row.append(f"{percentage:.1f}% ({wins})")
            annot_data.append(row)
        
        annot_df = pd.DataFrame(annot_data, index=class_names, columns=all_bases)
        
        # Create win count data for heatmap coloring
        heatmap_data = [[stats['per_class'][cls].get(base, 0) for base in all_bases] for cls in class_names]
        heatmap_df = pd.DataFrame(heatmap_data, index=class_names, columns=all_bases)
        
        fig, ax = plt.subplots(figsize=(max(10, len(all_bases) * 1.5), max(8, len(class_names) * 0.5)))
        sns.heatmap(heatmap_df, cmap='YlGn', linewidths=0.5, cbar_kws={'label': 'Wins'},
                    ax=ax, annot=annot_df, fmt='')
        
        ax.set_title('Wins by Image Class and Base Type', fontsize=18, pad=20)
        ax.set_xlabel('Base Type', fontsize=14)
        ax.set_ylabel('Image Class', fontsize=14)
        
        save_path = os.path.join(analysis_dir, 'heatmap.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return save_path
    
    def plot_summary(self, output_dir: str, stats: Dict) -> str:
        """Generate summary bar chart showing win rate by base type."""
        analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        win_rate_series = pd.Series(stats['base_win_rates']).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(range(len(win_rate_series)), win_rate_series.values, color=self.colors['neutral'])
        
        ax.set_title('Win Rate by Base Type', fontsize=18, pad=20)
        ax.set_xlabel('Base Type', fontsize=14)
        ax.set_ylabel('Win Rate (%)', fontsize=14)
        ax.set_xticks(range(len(win_rate_series)))
        ax.set_xticklabels(win_rate_series.index, rotation=45, ha='right')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, win_rate_series.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 2,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
        
        save_path = os.path.join(analysis_dir, 'summary.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return save_path
    
    def run(self, csv_path: str) -> str:
        """Generate analysis dashboard from CSV file."""
        # Load CSV data
        self.df = pd.read_csv(csv_path, encoding='utf-8')
        logging.info(f"Loaded {len(self.df)} evaluation records from CSV\n")
        
        # Determine image directory (parent of CSV location)
        output_dir = os.path.dirname(csv_path)
        self.image_dir = os.path.dirname(os.path.dirname(os.path.dirname(output_dir)))
        
        # Generate statistics
        stats = self.generate_statistics()
        
        # Generate per-class heatmaps
        for class_name in self.df['image_class'].unique():
            class_df = self.df[self.df['image_class'] == class_name]
            plot_path = self.plot_class_heatmap(class_name, class_df, output_dir)
            logging.info(f"Generated heatmap for {class_name}\n")
        
        # Generate preview
        logging.info("Generating overall image preview...\n")
        self.generate_preview(output_dir)
        
        # Generate heatmap
        logging.info("Generating wins heatmap...\n")
        self.plot_heatmap(output_dir, stats)
        
        # Generate summary
        logging.info("Generating win rate summary...\n")
        self.plot_summary(output_dir, stats)