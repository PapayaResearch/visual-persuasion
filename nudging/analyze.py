import os
import re
import glob
import logging
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from collections import defaultdict
import numpy as np
from PIL import Image

class EvaluationAnalyzer:
    """
    Analyzes evaluation results and generates statistics and visualizations.
    """
    def __init__(self, num_previews: int):
        # DataFrame to store all evaluation results
        self.df = None
        self.num_previews = num_previews
        self.image_dir = None

        self.colors = {
            'older': '#7fbf7f',      # Green for older image chosen
            'newer': '#ff7f7f',      # Red for newer image chosen
            'neutral': '#7f9fff'     # Blue for neutral/mixed content
        }
    
    def parse_logs(self, results_dir: str) -> pd.DataFrame:
        """Parse all evaluation log files in the directory."""
        data = []
        log_files = glob.glob(os.path.join(results_dir, "*.log"))
        
        if not log_files:
            logging.warning(f"No log files found in {results_dir}\n")
            self.df = pd.DataFrame()
            return self.df
            
        logging.info(f"Analyzing {len(log_files)} log files from {results_dir}\n")
        
        for log_file in log_files:
            image_class = os.path.basename(log_file).replace('.log', '')
            
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Split by separator lines
            sections = content.split('-' * 40)
            
            for section in sections:
                if 'VLM Choice:' not in section:
                    continue
                
                # Extract the comparison pair (e.g., "Evaluating best vs bestcontext")
                eval_match = re.search(r'Evaluating (\S+) vs (\S+)', section)
                if not eval_match:
                    continue
                base1 = eval_match.group(1)
                base2 = eval_match.group(2)
                
                # Extract VLM choice
                choice_match = re.search(r'VLM Choice:\s*(\S+)', section)
                if not choice_match:
                    continue
                choice = choice_match.group(1)
                
                # Extract reason
                reason_match = re.search(r'Reason[^:]*:[^\n]*\n(.*?)(?=-{40}|\Z)', section, re.DOTALL)
                reason = reason_match.group(1).strip() if reason_match else ""
                
                data.append({
                    'image_class': image_class,
                    'base1': base1,
                    'base2': base2,
                    'choice': choice,
                    'reason': reason
                })
        
        self.df = pd.DataFrame(data)
        logging.info(f"Parsed {len(self.df)} evaluation records\n")
        return self.df
    
    def plot_class_heatmap(self, class_name: str, class_df: pd.DataFrame, output_dir: str) -> str:
        """Generate a heatmap for a single image class with preview images."""
        # Get all unique base names and sort them
        all_bases = sorted(set(class_df['base1'].unique()) | set(class_df['base2'].unique()))
        n = len(all_bases)
        
        # Create matrix: 1 if older (row) chosen, 0 if newer (col) chosen, NaN if no comparison
        matrix = np.full((n, n), np.nan)
        
        for _, row in class_df.iterrows():
            base1, base2, choice = row['base1'], row['base2'], row['choice']
            
            # Find indices
            idx1 = all_bases.index(base1)
            idx2 = all_bases.index(base2)
            
            # Ensure we're filling below diagonal (older vs newer)
            if idx1 < idx2:  # base1 is older (earlier in sorted order)
                # Check if older (base1) was chosen
                matrix[idx2, idx1] = 1 if choice == base1 else 0
            else:  # base2 is older
                # Check if older (base2) was chosen
                matrix[idx1, idx2] = 1 if choice == base2 else 0
        
        # Get image files for this class
        image_files = {}
        if self.image_dir:
            for file in os.listdir(self.image_dir):
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    continue
                
                name_without_ext = os.path.splitext(file)[0]
                split = name_without_ext.split('_')
                img_class = '_'.join(split[:-1])
                base = split[-1]
                
                if img_class == class_name:
                    image_files[base] = file
        
        # Calculate figure width based on number of base types (preview determines width)
        preview_width = len(all_bases) * 3  # 3 inches per base type
        fig_width = max(preview_width, 12)  # Minimum 12 inches
        
        # Create figure with two rows: heatmap on top, preview images on bottom
        fig = plt.figure(figsize=(fig_width, fig_width * 0.8))  # Height proportional to width
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Top: Heatmap - center it by adding margins on left and right
        # Calculate the position to center the heatmap
        heatmap_ratio = n / len(all_bases)  # Ratio of heatmap width to full width
        left_margin = (1 - heatmap_ratio) / 2
        right_margin = left_margin
        
        # Create a sub-gridspec for the heatmap row with margins
        heatmap_gs = gs[0].subgridspec(1, 3, width_ratios=[left_margin, heatmap_ratio, right_margin])
        ax_heat = fig.add_subplot(heatmap_gs[1])  # Use the middle cell
        
        # Custom colormap: green=older chosen, gray=missing, red=newer chosen
        cmap = sns.color_palette([self.colors['older'], "#e0e0e0", self.colors['newer']], as_cmap=True)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        
        sns.heatmap(matrix, mask=mask, cmap=cmap, linewidths=0.5, 
                    vmin=0, vmax=1, square=True,
                    xticklabels=all_bases, yticklabels=all_bases,
                    cbar_kws={'label': 'Choice', 'ticks': [0, 1]}, ax=ax_heat)
        
        # Fix colorbar labels
        colorbar = ax_heat.collections[0].colorbar
        colorbar.set_ticklabels(['Newer', 'Older'])
        
        ax_heat.set_title(f'Comparison Results for {class_name}', fontsize=14, pad=20)
        ax_heat.set_xlabel('Base', fontsize=12)
        ax_heat.set_ylabel('Base', fontsize=12)
        
        # Bottom: Preview images
        ax_preview = fig.add_subplot(gs[1])
        ax_preview.axis('off')
        
        # Create sub-gridspec for preview images
        preview_gs = gs[1].subgridspec(1, len(all_bases), wspace=0.1)
        
        for idx, base_type in enumerate(all_bases):
            ax_img = fig.add_subplot(preview_gs[idx])
            
            if base_type in image_files:
                img_path = os.path.join(self.image_dir, image_files[base_type])
                try:
                    img = Image.open(img_path)
                    ax_img.imshow(img)
                except Exception as e:
                    logging.error(f"Failed to load image {img_path}: {e}\n")
                    ax_img.text(0.5, 0.5, 'Error\nloading\nimage', 
                            ha='center', va='center', transform=ax_img.transAxes, 
                            fontsize=9, color='red')
            else:
                # Missing image
                ax_img.text(0.5, 0.5, 'Missing', 
                        ha='center', va='center', transform=ax_img.transAxes, 
                        fontsize=10, color='gray')
            
            ax_img.set_title(base_type, fontsize=10, pad=5)
            ax_img.axis('off')
        
        # Save
        save_path = os.path.join(output_dir, f'{class_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_preview(self, output_dir: str) -> str:
        """Generate a preview image showing sample images from different classes and base types."""
        if self.df.empty or self.image_dir is None:
            logging.warning("No data or image directory available for preview generation.\n")
            return ""
        
        # Create analysis subdirectory
        analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Get all unique classes and base types
        all_classes = sorted(self.df['image_class'].unique())
        all_bases = sorted(set(self.df['base1'].unique()) | set(self.df['base2'].unique()))
        
        # Select random classes for preview
        num_preview_classes = min(self.num_previews, len(all_classes))
        if num_preview_classes == 0:
            logging.warning("No classes available for preview.\n")
            return ""
        
        random.shuffle(all_classes)
        selected_classes = sorted(all_classes[:num_preview_classes])
        
        # Create figure with rows=classes, cols=base_types
        rows = len(selected_classes)
        cols = len(all_bases)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        
        # Add title
        fig.suptitle('Image Preview by Class and Base Type', fontsize=16, fontweight='bold')
        
        # Ensure axes is always 2D array for consistent indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Get all image files from image directory
        image_files = {}
        for file in os.listdir(self.image_dir):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                continue
            
            name_without_ext = os.path.splitext(file)[0]
            split = name_without_ext.split('_')
            image_class = '_'.join(split[:-1])
            base = split[-1]
            
            if image_class not in image_files:
                image_files[image_class] = {}
            image_files[image_class][base] = file
        
        # Fill the grid
        for row_idx, class_name in enumerate(selected_classes):
            for col_idx, base_type in enumerate(all_bases):
                ax = axes[row_idx, col_idx]
                
                # Check if this class-base combination exists
                if class_name in image_files and base_type in image_files[class_name]:
                    img_file = image_files[class_name][base_type]
                    img_path = os.path.join(self.image_dir, img_file)
                    
                    try:
                        img = Image.open(img_path)
                        ax.imshow(img)
                        ax.set_title(f"{base_type}", fontsize=10)
                    except Exception as e:
                        logging.error(f"Failed to load image {img_path}: {e}\n")
                        ax.text(0.5, 0.5, 'Error loading\nimage', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=10)
                        ax.set_title(f"{base_type}", fontsize=10)
                else:
                    # Missing image
                    ax.text(0.5, 0.5, 'Missing', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(f"{base_type}", fontsize=10)
                
                ax.axis('off')
                
                # Add class label to the leftmost column
                if col_idx == 0:
                    ax.set_ylabel(f"{class_name}", fontsize=10, rotation=90, 
                                 labelpad=20, va='center')
        
        # Add column labels at the top
        for col_idx, base_type in enumerate(all_bases):
            axes[0, col_idx].set_xlabel(f"{base_type}", fontsize=12, labelpad=10)
            axes[0, col_idx].xaxis.set_label_position('top')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        
        # Save
        save_path = os.path.join(analysis_dir, 'preview.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Generated preview with {num_preview_classes} classes and {len(all_bases)} base types: {save_path}\n")
        
        return save_path
    
    def generate_statistics(self) -> Dict:
        """Generate key statistics from the evaluation data."""
        if self.df.empty:
            return {}
            
        stats = {}
        
        # Get all unique bases across all comparisons
        all_bases = sorted(set(self.df['base1'].unique()) | set(self.df['base2'].unique()))
        
        # Count wins for each base (how many times it was chosen)
        base_wins = defaultdict(int)
        base_comparisons = defaultdict(int)
        
        for _, row in self.df.iterrows():
            base1, base2, choice = row['base1'], row['base2'], row['choice']
            
            # Track comparisons
            base_comparisons[base1] += 1
            base_comparisons[base2] += 1
            
            # Track wins
            if choice == base1:
                base_wins[base1] += 1
            elif choice == base2:
                base_wins[base2] += 1
        
        # Calculate win rates
        base_win_rates = {}
        for base in all_bases:
            total = base_comparisons.get(base, 0)
            wins = base_wins.get(base, 0)
            base_win_rates[base] = (wins / total * 100) if total > 0 else 0
        
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
    
    def plot_heatmap(self, output_dir: str, stats: Dict) -> str:
        """Generate heatmap showing wins by class and base type."""
        if self.df.empty:
            return ""
        
        # Create analysis subdirectory
        analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        base_win_rates = stats['base_win_rates']
        all_bases = sorted(base_win_rates.keys())
        per_class_stats = stats['per_class']
        class_names = sorted(per_class_stats.keys())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, len(all_bases) * 1.5), max(8, len(class_names) * 0.5)))
        
        # Create pivot: rows=class, cols=base, values=wins
        heatmap_data = []
        for class_name in class_names:
            row = []
            for base in all_bases:
                row.append(per_class_stats[class_name].get(base, 0))
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, index=class_names, columns=all_bases)
        
        # Calculate total comparisons per class per base for percentage
        total_comparisons = {}
        for class_name in class_names:
            class_df = self.df[self.df['image_class'] == class_name]
            total_comparisons[class_name] = {}
            for base in all_bases:
                # Count how many times this base appeared in comparisons for this class
                count = len(class_df[(class_df['base1'] == base) | (class_df['base2'] == base)])
                total_comparisons[class_name][base] = count
        
        # Create annotation text with percentages and counts
        annot_data = []
        for class_name in class_names:
            row = []
            for base in all_bases:
                wins = per_class_stats[class_name].get(base, 0)
                total = total_comparisons[class_name].get(base, 0)
                if total > 0:
                    percentage = (wins / total) * 100
                    row.append(f"{percentage:.1f}% ({wins})")
                else:
                    row.append(f"0% (0)")
            annot_data.append(row)
        
        annot_df = pd.DataFrame(annot_data, index=class_names, columns=all_bases)
        
        sns.heatmap(heatmap_df, cmap='YlGn', linewidths=0.5, 
                    cbar_kws={'label': 'Wins'}, ax=ax, annot=annot_df, fmt='')
        
        ax.set_title('Wins by Image Class and Base Type', fontsize=18, pad=20)
        ax.set_xlabel('Base Type', fontsize=14)
        ax.set_ylabel('Image Class', fontsize=14)
        
        # Save
        save_path = os.path.join(analysis_dir, 'heatmap.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return save_path
    
    def plot_summary(self, output_dir: str, stats: Dict) -> str:
        """Generate summary bar chart showing win rate by base type."""
        if self.df.empty:
            return ""
        
        # Create analysis subdirectory
        analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        base_win_rates = stats['base_win_rates']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        win_rate_series = pd.Series(base_win_rates).sort_values(ascending=False)
        
        bars = ax.bar(range(len(win_rate_series)), win_rate_series.values, 
                      color=self.colors['neutral'])
        
        ax.set_title('Win Rate by Base Type', fontsize=18, pad=20)
        ax.set_xlabel('Base Type', fontsize=14)
        ax.set_ylabel('Win Rate (%)', fontsize=14)
        ax.set_xticks(range(len(win_rate_series)))
        ax.set_xticklabels(win_rate_series.index, rotation=45, ha='right')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, win_rate_series.values)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 2, 
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
        
        # Save
        save_path = os.path.join(analysis_dir, 'summary.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return save_path
    
    def run(self, analysis_dir: str) -> str:
        """Generate analysis dashboard."""
        # Get image directory (parent of analysis_dir, which is typically results_dir/evaluation/model/)
        # analysis_dir structure: .../results_dir/evaluation/model/
        # image_dir should be: .../results_dir/
        self.image_dir = os.path.dirname(os.path.dirname(analysis_dir))
        
        self.parse_logs(analysis_dir)

        if self.df.empty:
            logging.warning("No data available for analysis.\n")
            return analysis_dir
        
        # Generate per-class heatmaps with previews
        logging.info("Generating per-class heatmaps with previews...\n")
        for class_name in self.df['image_class'].unique():
            class_df = self.df[self.df['image_class'] == class_name]
            plot_path = self.plot_class_heatmap(class_name, class_df, analysis_dir)
            logging.info(f"Generated heatmap with preview for {class_name}: {plot_path}\n")
        
        # Generate statistics
        stats = self.generate_statistics()
        
        # Generate preview
        logging.info("Generating overall image preview...\n")
        preview_path = self.generate_preview(analysis_dir)
        if preview_path:
            logging.info(f"Preview image generated: {preview_path}\n")
        
        # Generate heatmap
        logging.info("Generating wins heatmap...\n")
        heatmap_path = self.plot_heatmap(analysis_dir, stats)
        if heatmap_path:
            logging.info(f"Heatmap generated: {heatmap_path}\n")
        
        # Generate summary
        logging.info("Generating win rate summary...\n")
        summary_path = self.plot_summary(analysis_dir, stats)
        if summary_path:
            logging.info(f"Summary generated: {summary_path}\n")
        
        return analysis_dir