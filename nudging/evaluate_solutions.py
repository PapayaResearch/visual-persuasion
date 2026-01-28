import os
import logging
import random
import itertools
import threading
import pandas as pd
import io
from typing import List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from utils.wrappers import LanguageModel, ImageModel

class EvaluationPipeline:
    """
    Evaluation pipeline to assess the visual nudges.
    """
    def __init__(
        self,
        image_editing_model: ImageModel,
        context_removal_model: LanguageModel,
        evaluator_model: LanguageModel,
        strategy_name: str,
        valid_statuses: List[str],
        name: str,
        metadata_template: str,
        cache_dir: str,
        n_evaluations: int = 1,
        max_comparisons: int = -1,
        sampling_seed: int = 42
    ):
        self.image_editing_model = image_editing_model
        self.context_removal_model = context_removal_model
        self.evaluator_model = evaluator_model
        self.evaluator_model.return_usage_data = True
        self.strategy_name = strategy_name
        self.valid_statuses = valid_statuses
        self.name = name
        self.metadata_template = metadata_template
        self.cache_dir = cache_dir
        self.n_evaluations = n_evaluations
        self.max_comparisons = max_comparisons
        self.sampling_seed = sampling_seed

    def _parse_filename_competition(self, filename: str):
        """
        Parse competition filename: CATEGORY_ID_STATUS.jpg or pair-X_..._CATEGORY_ID_STATUS.jpg
        Returns (category, image_id, status) or None if should skip.
        """
        if not filename.endswith('.jpg'):
            return None

        for status in self.valid_statuses:
            if filename.endswith(f'_{status}.jpg'):
                base = filename[:-4]
                parts = base.split('_')
                status_parts = status.split('_')
                num_status_parts = len(status_parts)
                image_id = parts[-(num_status_parts + 1)]
                category = parts[-(num_status_parts + 2)]
                return category, image_id, status

        return None

    def _load_completed_comparisons(self, csv_path: str) -> Set[Tuple[str, str, str]]:
        """
        Load completed comparisons from existing CSV.
        Returns a set of (image_class, base1, base2) tuples.
        """
        if not os.path.exists(csv_path):
            return set()

        completed = set()
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            completed.add((row['image_class'], row['base1'], row['base2']))

        logging.info(f"Loaded {len(completed)} completed comparisons from existing CSV\n")
        return completed

    def _generate_removal_prompt(
        self,
        img_bytes_1: bytes,
        img_bytes_2: bytes,
        image_class: str
    ) -> tuple[str, str]:
        """
        Generate prompts to remove contextual elements from both images.
        Returns two image editing instructions, one for each image.
        """
        logging.info(f"Generating context removal prompts for {image_class}\n")

        # Format metadata template with category if it contains placeholder
        if '{category}' in self.metadata_template:
            metadata = self.metadata_template.format(category=image_class)
        else:
            metadata = self.metadata_template

        removal_prompts = self.context_removal_model.get_response(
            images=[img_bytes_1, img_bytes_2],
            metadata=metadata
        )

        logging.info(f"Generated removal prompt 1: {removal_prompts.editing_instruction_1}\n")
        logging.info(f"Generated removal prompt 2: {removal_prompts.editing_instruction_2}\n")
        return removal_prompts.editing_instruction_1, removal_prompts.editing_instruction_2

    def _debias_image(
        self,
        img_bytes: bytes,
        removal_prompt: str,
        save_path: str = None
    ) -> bytes:
        """
        Apply the removal prompt to an image to create a debiased version.
        Returns the edited image bytes.
        """
        edited_image, edited_image_bytes = self.image_editing_model.edit(
            removal_prompt,
            img_bytes
        )

        if save_path:
            edited_image.save(save_path)
            logging.info(f"Saved debiased image to {save_path}\n")

        return edited_image_bytes

    def _create_comparison_grid(
        self,
        img_bytes_1: bytes,
        img_bytes_2: bytes,
        debiased_bytes_1: bytes,
        debiased_bytes_2: bytes,
        save_path: str
    ):
        """
        Create a 2x2 grid showing original and debiased images side by side.
        Top row: original images
        Bottom row: debiased images
        """
        # Load images
        img1 = Image.open(io.BytesIO(img_bytes_1))
        img2 = Image.open(io.BytesIO(img_bytes_2))
        debiased1 = Image.open(io.BytesIO(debiased_bytes_1))
        debiased2 = Image.open(io.BytesIO(debiased_bytes_2))

        # Resize all to same size (use max dimensions)
        max_width = max(img1.width, img2.width, debiased1.width, debiased2.width)
        max_height = max(img1.height, img2.height, debiased1.height, debiased2.height)

        img1 = img1.resize((max_width, max_height))
        img2 = img2.resize((max_width, max_height))
        debiased1 = debiased1.resize((max_width, max_height))
        debiased2 = debiased2.resize((max_width, max_height))

        # Create 2x2 grid
        grid_width = max_width * 2
        grid_height = max_height * 2
        grid = Image.new('RGB', (grid_width, grid_height))

        # Paste images: top row (originals), bottom row (debiased)
        grid.paste(img1, (0, 0))
        grid.paste(img2, (max_width, 0))
        grid.paste(debiased1, (0, max_height))
        grid.paste(debiased2, (max_width, max_height))

        # Save grid
        grid.save(save_path)
        logging.info(f"Saved comparison grid to {save_path}\n")

    def _evaluate_comparison(
        self,
        image_class: str,
        image_id_1: str,
        edit_type_1: str,
        img_bytes_1: bytes,
        image_id_2: str,
        edit_type_2: str,
        img_bytes_2: bytes,
        results_dir: str
    ) -> dict:
        """
        Evaluate a single comparison between two images using multiple judges.
        Applies debiasing preprocessing to remove contextual elements.
        Returns a dictionary with the evaluation result based on majority vote.
        """
        base_1 = f"{image_id_1}_{edit_type_1}"
        base_2 = f"{image_id_2}_{edit_type_2}"

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Check cache for debiased images (pair-specific since editing instructions depend on both images)
        debiased_cache_path_1 = os.path.join(self.cache_dir, f"{image_class}_{base_1}_vs_{base_2}_debiased_1.jpg")
        debiased_cache_path_2 = os.path.join(self.cache_dir, f"{image_class}_{base_1}_vs_{base_2}_debiased_2.jpg")

        both_cached = os.path.exists(debiased_cache_path_1) and os.path.exists(debiased_cache_path_2)

        if both_cached:
            # Load both debiased images from cache
            logging.info(f"[CACHE HIT] Loading cached debiased images for {base_1} and {base_2}\n")
            with open(debiased_cache_path_1, 'rb') as f:
                debiased_img_bytes_1 = f.read()
            with open(debiased_cache_path_2, 'rb') as f:
                debiased_img_bytes_2 = f.read()
        else:
            # Generate editing prompts and debias both images
            logging.info(f"[CACHE MISS] Generating editing prompts and debiasing {base_1} and {base_2}\n")
            removal_prompt_1, removal_prompt_2 = self._generate_removal_prompt(
                img_bytes_1, img_bytes_2, image_class
            )

            # Debias and save both images
            debiased_img_bytes_1 = self._debias_image(img_bytes_1, removal_prompt_1, save_path=debiased_cache_path_1)
            debiased_img_bytes_2 = self._debias_image(img_bytes_2, removal_prompt_2, save_path=debiased_cache_path_2)

            # Save prompts for reference
            prompts_path = os.path.join(self.cache_dir, f"{image_class}_{base_1}_vs_{base_2}_prompts.txt")
            with open(prompts_path, 'w') as f:
                f.write(f"EDITING PROMPT FOR {base_1}:\n")
                f.write(removal_prompt_1)
                f.write(f"\n\n{'='*80}\n\n")
                f.write(f"EDITING PROMPT FOR {base_2}:\n")
                f.write(removal_prompt_2)

        logging.info(f"Debiasing complete for {base_1} vs {base_2}\n")

        # Create and save comparison grid (original + debiased side by side)
        grid_path = os.path.join(results_dir, f"{base_1}_vs_{base_2}_comparison.jpg")
        self._create_comparison_grid(
            img_bytes_1, img_bytes_2,
            debiased_img_bytes_1, debiased_img_bytes_2,
            grid_path
        )

        def evaluate_single(is_1_first: bool):
            """Single judge evaluation"""
            images = [debiased_img_bytes_1, debiased_img_bytes_2] if is_1_first else [debiased_img_bytes_2, debiased_img_bytes_1]
            choice_map = {
                "first": base_1 if is_1_first else base_2,
                "second": base_2 if is_1_first else base_1
            }

            logging.info(f"Evaluating with {base_1} as {'first' if is_1_first else 'second'} image.\n")

            # Format metadata template with category if it contains placeholder
            if '{category}' in self.metadata_template:
                metadata = self.metadata_template.format(category=image_class)
            else:
                metadata = self.metadata_template

            evaluation, usage = self.evaluator_model.get_response(
                images=images,
                metadata=metadata
            )

            real_choice = choice_map.get(evaluation.choice.lower())

            logging.info(f"Judge chose {real_choice} - {evaluation.reason}\n")
            return (real_choice, evaluation.reason, usage)


        choice_1_first, reason_1_first, usage_1_first = evaluate_single(is_1_first=True)
        choice_2_first, reason_2_first, usage_2_first = evaluate_single(is_1_first=False)

        choice = None
        if choice_1_first == choice_2_first:
            choice = choice_1_first
            vlm_reason = reason_1_first if choice_1_first == base_1 else reason_2_first

        else:
            logging.warning(f"Inconsistent judge results for {base_1} vs {base_2}: {choice_1_first} vs {choice_2_first}\n")
            choice = "inconsistent"
            vlm_reason = " ".join([f"Judge 1: {reason_1_first}", f"Judge 2: {reason_2_first}"])

        completion_tokens = usage_1_first.completion_tokens + usage_2_first.completion_tokens
        prompt_tokens = usage_1_first.prompt_tokens + usage_2_first.prompt_tokens
        total_tokens = usage_1_first.total_tokens + usage_2_first.total_tokens

        reasoning_tokens = 0
        if usage_1_first.completion_tokens_details and usage_2_first.completion_tokens_details:
            reasoning_tokens = usage_1_first.completion_tokens_details.reasoning_tokens + usage_2_first.completion_tokens_details.reasoning_tokens

        return {
            'image_class': image_class,
            'base1': base_1,
            'base2': base_2,
            'choice': choice,
            'reason': vlm_reason,
            'completion_tokens': completion_tokens,
            'prompt_tokens': prompt_tokens,
            'total_tokens': total_tokens,
            'reasoning_tokens': reasoning_tokens
        }

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Runs the evaluation pipeline for each image comparison in parallel.
        Supports resumption by skipping already completed comparisons.
        """
        csv_save_path = os.path.join(results_dir, 'results_solutions.csv')

        # Load completed comparisons from existing CSV
        completed_comparisons = self._load_completed_comparisons(csv_save_path)

        # Group images by class
        class_groups = defaultdict(set)

        for img_path in image_paths:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            filename = os.path.basename(img_path)

            if self.strategy_name == 'competition':
                parsed = self._parse_filename_competition(filename)
                if parsed is None:
                    continue
                category, image_id, status = parsed
                class_groups[category].add((image_id, status, img_bytes))

        logging.info(f"Found {len(class_groups)} image classes to evaluate\n")

        # Collect all comparison tasks
        all_comparison_tasks = []
        for image_class in sorted(class_groups.keys()):
            comparable_images = sorted(class_groups[image_class], key=lambda x: '_'.join(x[:2]))

            for (image_id_1, edit_type_1, img_bytes_1), (image_id_2, edit_type_2, img_bytes_2) in \
                    itertools.combinations(comparable_images, 2):

                # Skip same image comparisons
                if image_id_1 == image_id_2:
                    continue
                # Skip same status comparisons
                if edit_type_1 == edit_type_2:
                    continue

                all_comparison_tasks.append((
                    image_class, image_id_1, edit_type_1, img_bytes_1,
                    image_id_2, edit_type_2, img_bytes_2
                ))

        # Sample from all tasks first
        if self.max_comparisons > 0 and len(all_comparison_tasks) > self.max_comparisons:
            logging.info(f"Sampling {self.max_comparisons} out of {len(all_comparison_tasks)} total\n")

            # Seed the random number generator for reproducibility
            random.seed(self.sampling_seed)

            # Group by comparison type and sample equally
            comparison_groups = defaultdict(list)
            for task in all_comparison_tasks:
                comp_type = tuple(sorted([task[2], task[5]]))
                comparison_groups[comp_type].append(task)

            per_group = self.max_comparisons // len(comparison_groups)
            comparison_tasks = []
            for comp_type, tasks in comparison_groups.items():
                sample_size = min(per_group, len(tasks))
                comparison_tasks.extend(random.sample(tasks, sample_size))
                logging.info(f"  {comp_type}: sampled {sample_size}/{len(tasks)}")
        else:
            comparison_tasks = all_comparison_tasks

        # Filter out completed comparisons
        comparison_tasks = [
            task for task in comparison_tasks
            if (task[0], f"{task[1]}_{task[2]}", f"{task[4]}_{task[5]}") not in completed_comparisons
        ]

        comparison_tasks = list(comparison_tasks) * self.n_evaluations  # Repeat tasks for multiple evaluations if >1 specified
        total_comparisons = len(comparison_tasks)
        logging.info(f"Total comparisons to evaluate: {total_comparisons}\n")

        # Prepare CSV file for incremental writing
        csv_lock = threading.Lock()
        file_exists = os.path.exists(csv_save_path)

        # Prepare results directory
        debiased_images_dir = os.path.join(results_dir, 'debiased_images')
        os.makedirs(debiased_images_dir, exist_ok=True)

        # Open CSV in append mode for incremental writing
        with open(csv_save_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'image_class', 'base1', 'base2', 'choice', 'reason',
                'completion_tokens', 'prompt_tokens', 'total_tokens',
                'reasoning_tokens'
            ]
            writer = pd.DataFrame(columns=fieldnames)

            # Write header if file is new
            if not file_exists:
                writer.to_csv(csvfile, index=False, header=True, mode='a')
                csvfile.flush()

            # Execute comparisons in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_comparison,
                        *task,
                        debiased_images_dir
                    ): task for task in comparison_tasks
                }

                for future in tqdm(
                        as_completed(futures),
                        total=total_comparisons,
                        desc="Evaluating comparisons",
                        unit="comparison"
                ):
                    result = future.result()

                    # Write result immediately to CSV with thread safety
                    with csv_lock:
                        result_df = pd.DataFrame([result])
                        result_df.to_csv(csvfile, index=False, header=False, mode='a')
                        csvfile.flush()  # Force write to disk

        logging.info(f"Evaluation completed. Results saved to: {csv_save_path}\n")
