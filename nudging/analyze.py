import os
import re
import glob
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

class EvaluationAnalyzer:
    """
    Analyzes evaluation results and generates statistics and visualizations.
    """
    def __init__(self):
        # DataFrame to store all evaluation results
        self.df = None

        self.colors = {
            'original': '#ff7f7f',    # Red for original
            'edited': '#7fbf7f',      # Green for edited
            'neutral': '#7f9fff'      # Blue for neutral/mixed content
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
            image_id = os.path.basename(log_file).replace('.log', '')
            
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Split by separator lines
            sections = content.split('-' * 40)
            
            for section in sections:
                if 'VLM Choice:' not in section:
                    continue
                    
                # Extract iteration number
                iter_match = re.search(r'iteration (\d+)', section)
                if not iter_match:
                    continue
                iteration = int(iter_match.group(1))
                
                # Extract VLM choice
                choice_match = re.search(r'VLM Choice:\s*(original|edited)', section)
                if not choice_match:
                    continue
                choice = choice_match.group(1)
                
                # Extract reason
                reason_match = re.search(r'Reason[^:]*:[^\n]*\n(.*?)(?=-{40}|\Z)', section, re.DOTALL)
                reason = reason_match.group(1).strip() if reason_match else ""
                
                data.append({
                    'image_id': image_id,
                    'iteration': iteration,
                    'choice': choice,
                    'reason': reason
                })
        
        self.df = pd.DataFrame(data)
        logging.info(f"Parsed {len(self.df)} evaluation records\n")
        return self.df
    
    def generate_statistics(self) -> Dict:
        """Generate key statistics from the evaluation data."""
        if self.df.empty:
            return {}
            
        stats = {}
        
        # Overall statistics
        total = len(self.df)
        edited_preferred = len(self.df[self.df['choice'] == 'edited'])
        original_preferred = len(self.df[self.df['choice'] == 'original'])
        
        stats['total_comparisons'] = total
        stats['edited_preferred'] = edited_preferred
        stats['original_preferred'] = original_preferred
        stats['edited_percentage'] = (edited_preferred / total) * 100 if total > 0 else 0
        
        # Per-iteration statistics
        iter_stats = self.df.groupby('iteration')['choice'].apply(
            lambda x: (x == 'edited').mean() * 100
        ).to_dict()
        stats['per_iteration'] = iter_stats
        
        # Most successful iterations
        iteration_success = self.df.groupby('iteration')['choice'].apply(
            lambda x: (x == 'edited').sum()
        ).sort_values(ascending=False)
        stats['most_successful_iteration'] = iteration_success.index[0] if not iteration_success.empty else None
        
        # Per-image statistics
        image_stats = self.df.groupby('image_id')['choice'].apply(
            lambda x: (x == 'edited').mean() * 100
        ).to_dict()
        stats['per_image'] = image_stats
        
        # Most improved images
        image_success = self.df.groupby('image_id')['choice'].apply(
            lambda x: (x == 'edited').sum()
        ).sort_values(ascending=False)
        stats['most_improved_images'] = image_success.index[:5].tolist() if len(image_success) >= 5 else image_success.index.tolist()
        
        return stats
    
    def plot_analysis_dashboard(self, output_dir: str) -> str:
        """Generate a comprehensive dashboard with multiple plots."""
        if self.df.empty:
            return ""
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Overall preference pie chart (top-left)
        counts = self.df['choice'].value_counts()
        colors = [self.colors['original'] if choice == 'original' else self.colors['edited'] 
                 for choice in counts.index]
        wedges, texts, autotexts = ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors)
        
        # Make percentage annotations larger
        for autotext in autotexts:
            autotext.set_fontsize(16)
        
        ax1.set_title('Overall Preference Distribution', fontsize=16)
        
        # 2. Image heatmap (top-right)
        # Create pivot table: rows=image_id, cols=iteration, values=1 for edited, 0 for original
        pivot_data = self.df.pivot_table(
            index='image_id', 
            columns='iteration',
            values='choice',
            aggfunc=lambda x: (x == 'edited').mean()  # 1.0 if all edited, 0.0 if all original
        ).fillna(-0.1)  # Fill NaN with -0.1 to distinguish from 0.0 (original)
        
        # Custom colormap: red=original, gray=missing, green=edited
        cmap = sns.color_palette([self.colors['original'], "#e0e0e0", self.colors['edited']], as_cmap=True)
        
        sns.heatmap(pivot_data, cmap=cmap, linewidths=0.5, 
                    vmin=-0.1, vmax=1.0, 
                    cbar_kws={'label': 'Preference'}, ax=ax2)
        
        # Fix colorbar labels - set proper tick positions and labels
        colorbar = ax2.collections[0].colorbar
        colorbar.set_ticks([-0.05, 0.0, 1.0])
        colorbar.set_ticklabels(['N/A', 'Original', 'Edited'])
        
        ax2.set_title('Evaluation Results by Image and Iteration', fontsize=16)
        ax2.set_xlabel('Iteration', fontsize=14)
        ax2.set_ylabel('Image ID', fontsize=14)
        
        # 3. Preference by iteration (bottom-left)
        iter_data = self.df.groupby('iteration')['choice'].apply(
            lambda x: (x == 'edited').mean() * 100
        ).reset_index()
        ax3.plot(iter_data['iteration'], iter_data['choice'], 
            marker='o', linewidth=2, markersize=8, color=self.colors['neutral'])
        
        # Add annotations for each data point
        for i, row in iter_data.iterrows():
            ax3.annotate(f'{row["choice"]:.1f}%', 
                (row['iteration'], row['choice']),
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', 
                    edgecolor='gray', 
                    alpha=0.8))
        
        ax3.set_title('Preference for Edited Images by Iteration', fontsize=16)
        ax3.set_xlabel('Iteration', fontsize=14)
        ax3.set_ylabel('Percentage Preferring Edited (%)', fontsize=14)
        ax3.set_ylim(0, 110)
        ax3.grid(True, alpha=0.3)
        
        # Force integer x-axis ticks
        ax3.set_xticks(iter_data['iteration'].unique())
        ax3.set_xticklabels([str(int(x)) for x in iter_data['iteration'].unique()])
        
        # 4. All images sorted by success rate (bottom-right)
        all_images = self.df.groupby('image_id')['choice'].apply(
            lambda x: (x == 'edited').mean() * 100
        ).sort_values(ascending=False).reset_index()
        
        # Use consistent neutral blue color for all bars
        all_images['image_id'] = all_images['image_id'].astype(str)
        bars4 = ax4.bar(range(len(all_images)), all_images['choice'], color=self.colors['neutral'])
        ax4.set_title('Success Rate by Image', fontsize=16)
        ax4.set_xlabel('Image ID', fontsize=14)
        ax4.set_ylabel('Success Rate (%)', fontsize=14)
        ax4.set_xticks(range(len(all_images)))
        ax4.set_xticklabels(all_images['image_id'], rotation=90)
        ax4.set_ylim(0, 110)
        
        plt.suptitle('Evaluation Results Analysis', fontsize=20)
        
        # Save and return the path
        save_path = os.path.join(output_dir, 'analysis.png')
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(save_path, dpi=120)
        plt.close()
        return save_path
    
    def run(self, analysis_dir: str) -> str:
        """Generate analysis dashboard."""
        output_dir = os.path.join(analysis_dir, "analysis")
        os.makedirs(output_dir, exist_ok=True)

        self.parse_logs(analysis_dir)

        if self.df.empty:
            logging.warning("No data available for analysis.\n")
            return output_dir
            
        # Generate only the dashboard plot
        plot_path = self.plot_analysis_dashboard(output_dir)
        
        if plot_path:
            logging.info(f"Analysis dashboard generated: {plot_path}\n")
        
        return output_dir