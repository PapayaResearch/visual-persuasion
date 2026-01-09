import os
import re
import argparse
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory", "-d",
        type=str,
        help="Directory containing the result files (images and prompts)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results_viewer.html",
        help="Output HTML filename (default: results_viewer.html)"
    )

    args = parser.parse_args()

    # Validate directory
    directory = Path(args.directory)
    organized_data = organize_files(directory)

    print(f"Found {len(organized_data)} image set(s)")

    # Generate HTML
    output_path = Path(args.output)
    generate_html(organized_data, output_path)

    print(f"\n✅ Successfully generated: {output_path.absolute()}")
    print(f"Open the file in your browser to view the results!")

    return 0


def parse_filename(filename: str) -> tuple[str, int, str, str]:
    """
    Parse a filename to extract base_name, iteration, stage, and file_type.

    Examples:
        BACKPACK_9742702d_iter-0-original.jpg -> ('BACKPACK_9742702d', 0, 'original', 'jpg')
        BACKPACK_9742702d_iter-1-a_edited.jpg -> ('BACKPACK_9742702d', 1, 'edited', 'jpg')
        BACKPACK_9742702d_iter-1-b_best.jpg -> ('BACKPACK_9742702d', 1, 'best', 'jpg')
    """
    # Match pattern: basename_iter-N-stage.ext
    pattern = r'(.+)_iter-(\d+|n)-([ab]_)?(.+)\.(jpg|txt)'
    match = re.match(pattern, filename)

    base_name = match.group(1)
    iteration = match.group(2)
    iteration = -1 if iteration == 'n' else int(iteration)
    stage_prefix = match.group(3) or ''
    stage = match.group(4)
    file_type = match.group(5)

    # Clean up stage name
    if stage_prefix:
        if 'a_' in stage_prefix:
            stage = 'edited' if 'edited' in stage else stage
        elif 'b_' in stage_prefix:
            stage = 'best' if 'best' in stage else stage

    return (base_name, iteration, stage, file_type)


def organize_files(directory: Path) -> dict[str, dict]:
    """
    Organize files by base name and iteration.

    Returns a nested dictionary structure:
    {
        'BACKPACK_9742702d': {
            0: {'original': 'path/to/original.jpg'},
            1: {'edited': 'path/to/edited.jpg', 'edited_prompt': 'prompt text', 'best': 'path/to/best.jpg', ...},
            ...
        },
        ...
    }
    """
    organized = defaultdict(lambda: defaultdict(dict))

    for filename in os.listdir(directory):
        filepath = directory / filename
        if not filepath.is_file():
            continue

        parsed = parse_filename(filename)
        if not parsed:
            continue

        base_name, iteration, stage, file_type = parsed

        # Store file paths and read text content
        if file_type == 'jpg':
            organized[base_name][iteration][stage] = str(filepath)
        elif file_type == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            organized[base_name][iteration][f'{stage}_prompt'] = content

    return organized


def generate_html(organized_data: dict[str, dict], output_path: Path) -> None:
    """
    Generate an HTML webpage displaying all results in a structured format.
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual Nudging Results Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }

        .container {
            max-width: 1800px;
            margin: 0 auto;
        }

        h1 {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .image-set {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        .image-set-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
        }

        .image-set-title {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .iteration {
            margin-bottom: 40px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }

        .iteration-header {
            background: linear-gradient(to right, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 15px 20px;
            font-size: 1.3em;
            font-weight: bold;
        }

        .iteration-content {
            padding: 20px;
            background: #fafafa;
        }

        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .image-card {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.2);
        }

        .image-card-header {
            background: #667eea;
            color: white;
            padding: 10px 15px;
            font-weight: bold;
            text-align: center;
        }

        .image-card-header.edited {
            background: #f5576c;
        }

        .image-card-header.best {
            background: #4CAF50;
        }

        .image-card-header.original {
            background: #2196F3;
        }

        .image-card-header.final {
            background: #FFD700;
            color: #333;
        }

        .image-wrapper {
            position: relative;
            width: 100%;
            padding-bottom: 75%; /* 4:3 aspect ratio */
            overflow: hidden;
            background: #f5f5f5;
        }

        .image-wrapper img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            cursor: pointer;
        }

        .prompt-section {
            padding: 15px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }

        .prompt-title {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
            font-size: 0.9em;
            text-transform: uppercase;
        }

        .prompt-text {
            font-size: 0.9em;
            line-height: 1.6;
            color: #555;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 200px;
            overflow-y: auto;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }

        .stats {
            background: linear-gradient(to right, #e0c3fc 0%, #8ec5fc 100%);
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }

        .stat-item {
            text-align: center;
            padding: 10px;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }

        .stat-label {
            font-size: 0.9em;
            color: #555;
            margin-top: 5px;
        }

        /* Modal for full-size images */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            cursor: pointer;
        }

        .modal-content {
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        .close {
            position: absolute;
            top: 30px;
            right: 50px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }

        @media (max-width: 768px) {
            .image-grid {
                grid-template-columns: 1fr;
            }

            h1 {
                font-size: 1.8em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Visual Nudging Results Viewer</h1>
"""

    # Process each image set
    for base_name in sorted(organized_data.keys()):
        iterations = organized_data[base_name]
        max_iteration = max(k for k in iterations.keys() if k != -1)

        html_content += f"""
        <div class="image-set">
            <div class="image-set-header">
                <div class="image-set-title">{base_name}</div>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-value">{max_iteration}</div>
                        <div class="stat-label">Iterations</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{len(iterations)}</div>
                        <div class="stat-label">Total Stages</div>
                    </div>
                </div>
            </div>
"""

        # Process each iteration
        for iter_num in sorted(k for k in iterations.keys() if k != -1):
            iter_data = iterations[iter_num]

            if iter_num == 0:
                # Original image
                html_content += f"""
            <div class="iteration">
                <div class="iteration-header">Initial State (Iteration 0)</div>
                <div class="iteration-content">
                    <div class="image-grid">
                        <div class="image-card">
                            <div class="image-card-header original">Original Image</div>
                            <div class="image-wrapper">
                                <img src="{iter_data.get('original', '')}" alt="Original" onclick="openModal(this.src)">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
"""
            else:
                # Optimization iterations
                html_content += f"""
            <div class="iteration">
                <div class="iteration-header">Iteration {iter_num}</div>
                <div class="iteration-content">
                    <div class="image-grid">
"""

                # Edited image
                if 'edited' in iter_data:
                    html_content += f"""
                        <div class="image-card">
                            <div class="image-card-header edited">Edited Image</div>
                            <div class="image-wrapper">
                                <img src="{iter_data['edited']}" alt="Edited" onclick="openModal(this.src)">
                            </div>
"""
                    if 'edited_prompt' in iter_data:
                        html_content += f"""
                            <div class="prompt-section">
                                <div class="prompt-title">Editing Prompt</div>
                                <div class="prompt-text">{iter_data['edited_prompt']}</div>
                            </div>
"""
                    html_content += """
                        </div>
"""

                # Best image
                if 'best' in iter_data:
                    html_content += f"""
                        <div class="image-card">
                            <div class="image-card-header best">Best Image (Selected)</div>
                            <div class="image-wrapper">
                                <img src="{iter_data['best']}" alt="Best" onclick="openModal(this.src)">
                            </div>
"""
                    if 'best_prompt' in iter_data:
                        html_content += f"""
                            <div class="prompt-section">
                                <div class="prompt-title">Prompt Leading to Best</div>
                                <div class="prompt-text">{iter_data['best_prompt']}</div>
                            </div>
"""
                    html_content += """
                        </div>
"""

                html_content += """
                    </div>
                </div>
            </div>
"""

        # Final result
        if -1 in iterations:
            final_data = iterations[-1]
            html_content += f"""
            <div class="iteration">
                <div class="iteration-header">🏆 Final Result</div>
                <div class="iteration-content">
                    <div class="image-grid">
                        <div class="image-card">
                            <div class="image-card-header final">Final Optimized Image</div>
                            <div class="image-wrapper">
                                <img src="{final_data.get('edit', '')}" alt="Final" onclick="openModal(this.src)">
                            </div>
"""
            if 'prompt' in final_data:
                html_content += f"""
                            <div class="prompt-section">
                                <div class="prompt-title">Final Prompt</div>
                                <div class="prompt-text">{final_data['prompt']}</div>
                            </div>
"""
            html_content += """
                        </div>
                    </div>
                </div>
            </div>
"""

        html_content += """
        </div>
"""

    # Close HTML and add modal
    html_content += """
    </div>

    <!-- Modal for full-size images -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        function openModal(src) {
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = src;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>
"""

    # Write HTML to file
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(html_content)


if __name__ == "__main__":
    main()
