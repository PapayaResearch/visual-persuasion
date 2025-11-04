import os
import io
import logging
import dataclasses
import time
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import ImageModel, LanguageModel
from utils.features import compute_ssim, count_objects, estimate_mean_depth_entropy


@dataclasses.dataclass
class VisualNudgeMAPElites:
    """
    Orchestrates the visual nudging pipeline with MAP-Elites.
    Hydra instantiates this class and its model dependencies automatically.
    """
    iterations: int
    editing_context_prompt: str
    initial_prompt: str
    background_state_prompt: str
    image_editing_model: ImageModel
    num_judges: int
    evaluator_prompt: str
    use_history_of_prompts: bool
    evaluator_model: LanguageModel

    # Optimizer
    num_proposals: int
    proposer_sees_current_prompt: bool
    proposer_sees_history: bool
    proposer_model: LanguageModel

    # MAP-Elites specific
    map_elites_bins: int = 5  # Number of bins per dimension
    map_elites_variations: int = 5  # Number of variations to generate per iteration


    def __post_init__(self):
        """Initialize the MAP-Elites archive."""
        # Archive structure: {(ssim_bin, depth_bin, objects_bin): {'prompt': str, 'image_bytes': bytes, 'fitness': float, 'image': Image}}
        self.archive = {}
        self.feature_bins = self.map_elites_bins
        
        self._total_num_images_generated = 0
        self._cost_per_image_generated = 0.039

    # def _get_fitness(self, edited_image_bytes: bytes, original_image_bytes: bytes, context_image_bytes: bytes) -> float:
    #     """
    #     Compute fitness score for an image based on evaluator/judge preference.
    #     Returns a score between 0 and 1, where 1 means strongly preferred by judges.
    #     """
        # selected_choice, _ = self._evaluate(
        #     edited_image_bytes=edited_image_bytes,
        #     original_image_bytes=original_image_bytes,
        #     context_image_bytes=context_image_bytes
        # )
        
        # # Fitness is binary: 1.0 if judges prefer edited, 0.0 if they prefer original
        # return 1.0 if selected_choice.lower() == "edited" else 0.0
    
    def _get_fitness(self, edited_image_bytes: bytes, original_image_bytes: bytes, context_image_bytes: bytes) -> float:
        """
        Compute fitness score for an image based on evaluator/judge preference.
        Returns a score between 0 and 1, where 1 means strongly preferred by judges.
        """
        eval_total_start = time.time()
        comparison_image_bytes = context_image_bytes if context_image_bytes else original_image_bytes

        def evaluate_single(judge_id: int, is_edited_first: bool):
            """Single evaluation task"""
            judge_start = time.time()
            logging.info(f"Judge {judge_id}: Evaluating with edited image as {'first' if is_edited_first else 'second'} image.\n")

            images = [edited_image_bytes, comparison_image_bytes] if is_edited_first else [comparison_image_bytes, edited_image_bytes]
            choice_map = {
                "first": "edited" if is_edited_first else "original",
                "second": "original" if is_edited_first else "edited",
            }

            logging.info(f"Judge {judge_id} choice map: {choice_map}\n")

            model_call_start = time.time()
            evaluation = self.evaluator_model.get_response(
                task=self.evaluator_prompt,
                images=images
            )
            model_call_duration = time.time() - model_call_start

            real_choice = choice_map.get(evaluation.choice.lower())

            if not real_choice:
                logging.warning(f"Judge {judge_id}: Evaluation failed. Skipping.\n")
                return None

            judge_duration = time.time() - judge_start
            print(f"  ⏱️  Judge {judge_id} ({'edited 1st' if is_edited_first else 'orig 1st'}): {judge_duration:.2f}s (model: {model_call_duration:.2f}s)")
            logging.info(f"Judge {judge_id}: {evaluation}\n")
            return (real_choice, evaluation.reason)

        # Run all evaluations in parallel
        parallel_start = time.time()
        judge_results = {}  # judge_id -> {True: result, False: result}

        with ThreadPoolExecutor(max_workers=min(self.num_judges * 2, 8)) as eval_executor:
            # Submit all evaluation tasks
            submit_start = time.time()
            future_to_judge = {}
            for judge_id in range(self.num_judges):
                for is_edited_first in [True, False]:
                    future = eval_executor.submit(evaluate_single, judge_id, is_edited_first)
                    future_to_judge[future] = (judge_id, is_edited_first)
            submit_duration = time.time() - submit_start
            print(f"  ⏱️  Submitted {len(future_to_judge)} judge tasks: {submit_duration:.2f}s")

            # Collect results organized by judge
            collect_start = time.time()
            for future in as_completed(future_to_judge):
                judge_id, is_edited_first = future_to_judge[future]
                result = future.result()

                if judge_id not in judge_results:
                    judge_results[judge_id] = {}
                judge_results[judge_id][is_edited_first] = result
            collect_duration = time.time() - collect_start
            print(f"  ⏱️  Collected all judge results: {collect_duration:.2f}s")

        parallel_duration = time.time() - parallel_start
        print(f"  ⏱️  Parallel evaluation total: {parallel_duration:.2f}s")

        # Count consistent judges and compute fitness as proportion preferring edited
        aggregation_start = time.time()
        consistent_edited_votes = 0
        consistent_original_votes = 0
        total_consistent_judges = 0

        for judge_id, results in judge_results.items():
            result_edited_first = results.get(True)
            result_original_first = results.get(False)

            # Both evaluations must succeed
            if result_edited_first is None or result_original_first is None:
                logging.warning(f"Judge {judge_id}: One or both evaluations failed. Skipping judge.\n")
                continue

            choice_edited_first, _ = result_edited_first
            choice_original_first, _ = result_original_first

            # Only count if both orderings agree
            if choice_edited_first == choice_original_first:
                logging.info(f"Judge {judge_id}: Consistent choice '{choice_edited_first}' across both orderings.\n")
                total_consistent_judges += 1
                if choice_edited_first == "edited":
                    consistent_edited_votes += 1
                else:
                    consistent_original_votes += 1
            else:
                logging.warning(f"Judge {judge_id}: Inconsistent - chose '{choice_edited_first}' when edited first, '{choice_original_first}' when original first. Skipping judge.\n")

        # Compute fitness as proportion of consistent judges preferring edited
        if total_consistent_judges == 0:
            # No consistent judges - return 0.5 (neutral)
            logging.warning("No judges were consistent across both orderings. Returning neutral fitness of 0.5.\n")
            fitness = 0.5
        else:
            fitness = consistent_edited_votes / total_consistent_judges
            logging.info(f"Fitness: {fitness:.2f} ({consistent_edited_votes}/{total_consistent_judges} consistent judges preferred edited).\n")

        aggregation_duration = time.time() - aggregation_start
        print(f"  ⏱️  Fitness computation: {aggregation_duration:.2f}s")

        eval_total_duration = time.time() - eval_total_start
        print(f"  ⏱️  TOTAL _get_fitness(): {eval_total_duration:.2f}s")

        return fitness

    def _get_local_elites_context(self, feature_bin: tuple[int, int, int], k: int = 3) -> str:
        """
        Get the k nearest elites in feature space to provide context for optimization.
        Returns a string describing the prompts and features of nearby elites.
        """
        if not self.archive:
            return "No elites in archive yet."
        
        # Compute distances to all archived elites
        distances = []
        for archived_bin, elite in self.archive.items():
            # Euclidean distance in feature space
            dist = sum((a - b) ** 2 for a, b in zip(feature_bin, archived_bin)) ** 0.5
            distances.append((dist, archived_bin, elite))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        nearest = distances[:k]
        
        # Format context
        context_parts = [f"Found {len(nearest)} nearby elite(s) in archive:"]
        for i, (dist, bin_coords, elite) in enumerate(nearest, 1):
            context_parts.append(
                f"\nElite {i} (distance {dist:.2f}, bins {bin_coords}):\n"
                f"  Prompt: {elite['prompt']}\n"
                f"  Fitness: {elite['fitness']:.2f}"
            )
        
        return "\n".join(context_parts)

    def _process_single_image(
        self,
        image_path: str,
        img_idx: int,
        total_images: int,
        results_dir: str,
        pbar: tqdm
    ) -> None:
        """
        Processes a single image through the MAP-Elites optimization loop.
        """
        image_start_time = time.time()
        base_filename, _ = os.path.splitext(os.path.basename(image_path))
        logging.info(f"\n===== Processing Image {img_idx + 1}/{total_images}: {base_filename} =====\n")

        # Load and save original image
        load_start = time.time()
        with open(image_path, "rb") as f:
            original_image_bytes = f.read()

            original_image = Image.open(io.BytesIO(original_image_bytes))
            original_save_path = os.path.join(results_dir, f"{base_filename}_iter-0-original.jpg")
            original_image.save(original_save_path)
            logging.info(f"Saved original image to: {original_save_path}\n")
        load_duration = time.time() - load_start
        print(f"⏱️  Image load & save: {load_duration:.2f}s")

        current_prompt = self.initial_prompt
        best_prompt = current_prompt
        best_image = original_image
        best_fitness = 0.0
        context_image_bytes = original_image_bytes
        
        history_of_prompts = [current_prompt] if self.use_history_of_prompts else []

        for iter in range(self.iterations):
            iter_start_time = time.time()
            logging.info(f"\n>> ITERATION {iter + 1}/{self.iterations} <<\n")

            # Generate multiple prompt variations using proposer
            proposer_start = time.time()
            
            # Get context from local elites if archive is not empty
            local_elites_context = ""
            if self.archive:
                # Use best prompt's features as reference point
                local_elites_context = f"\n\nLocal elites for context:\n{self._get_local_elites_context((0, 0, 0), k=3)}"
            
            proposer_response = self.proposer_model.get_response(
                reason=f"Generate diverse variations to explore the feature space.{local_elites_context}",
                current_prompt=best_prompt if self.proposer_sees_current_prompt else "",
                history_of_prompts=("\n".join([f"{i}. {p}" for i, p in enumerate(history_of_prompts)]) if self.use_history_of_prompts and self.proposer_sees_history else ""),
                current_iteration=iter + 1,
                total_iterations=self.iterations,
                num_proposals=self.map_elites_variations
            )
            proposer_duration = time.time() - proposer_start
            print(f"⏱️  [Iter {iter + 1}] Proposer model: {proposer_duration:.2f}s")

            candidate_prompts = proposer_response.candidate_prompts
            logging.info(f"PROPOSER generated {len(candidate_prompts)} candidate prompts:\n")
            for i, prompt in enumerate(candidate_prompts):
                logging.info(f"  Candidate {i+1}: {prompt}\n")

            # Generate images for all candidate prompts IN PARALLEL
            edit_start = time.time()
            
            def generate_single_image(i: int, prompt: str):
                """Generate a single image from a prompt"""
                logging.info(f"Generating image {i+1}/{len(candidate_prompts)} for prompt: {prompt}\n")
                editing_prompt = f"{prompt}\n{self.editing_context_prompt}"
                edited_image, edited_image_bytes = self.image_editing_model.edit(
                    f"{editing_prompt}\n{self.background_state_prompt}",
                    original_image_bytes,
                    context_image_bytes
                )
                self._total_num_images_generated += 1
                
                # Save candidate image
                candidate_save_path = os.path.join(results_dir, f"{base_filename}_iter-{iter + 1}-candidate-{i+1}.jpg")
                edited_image.save(candidate_save_path)
                
                return {
                    'prompt': prompt,
                    'image': edited_image,
                    'image_bytes': edited_image_bytes,
                    'save_path': candidate_save_path
                }
            
            # Parallel image generation
            candidate_images = []
            with ThreadPoolExecutor(max_workers=min(len(candidate_prompts), 4)) as img_executor:
                futures = [img_executor.submit(generate_single_image, i, prompt) 
                          for i, prompt in enumerate(candidate_prompts)]
                for future in as_completed(futures):
                    candidate_images.append(future.result())
            
            edit_duration = time.time() - edit_start
            print(f"⏱️  [Iter {iter + 1}] Generated {len(candidate_images)} images (parallel): {edit_duration:.2f}s")

            # Compute features and fitness for each candidate IN PARALLEL, update archive
            eval_start = time.time()
            
            def evaluate_candidate(i: int, candidate: dict):
                """Evaluate features and fitness for a single candidate"""
                logging.info(f"\nEvaluating candidate {i+1}/{len(candidate_images)}\n")
                
                edited_image = Image.open(io.BytesIO(candidate['image_bytes']))
                
                # Compute features (for binning/diversity only)
                ssim_value = compute_ssim(original_image, edited_image)
                depth_value = estimate_mean_depth_entropy(edited_image)
                objects_count = count_objects(edited_image)
                
                # Discretize into bins
                ssim_bin = min(int(ssim_value * self.feature_bins), self.feature_bins - 1)
                depth_normalized = depth_value / 255.0
                depth_bin = min(int(depth_normalized * self.feature_bins), self.feature_bins - 1)
                objects_bin = min(objects_count, self.feature_bins - 1)
                
                feature_bin = (ssim_bin, depth_bin, objects_bin)
                
                # Compute fitness from JUDGES (actual quality measure!)
                fitness = self._get_fitness(
                    edited_image_bytes=candidate['image_bytes'],
                    original_image_bytes=original_image_bytes,
                    context_image_bytes=context_image_bytes
                )
                
                logging.info(f"Candidate {i+1}: SSIM={ssim_value:.3f}, Depth={depth_value:.1f}, Objects={objects_count}\n")
                logging.info(f"Candidate {i+1}: Fitness={fitness:.2f}, Feature bin={feature_bin}\n")
                
                
                # Log current total cost
                total_cost = self._total_num_images_generated * self._cost_per_image_generated
                logging.info(f"Total images generated so far: {self._total_num_images_generated}, Estimated cost: ${total_cost:.2f}\n")
                print(f"Total images generated so far: {self._total_num_images_generated}, Estimated cost: ${total_cost:.2f}\n")

                return {
                    'index': i,
                    'feature_bin': feature_bin,
                    'fitness': fitness,
                    'ssim': ssim_value,
                    'depth': depth_value,
                    'objects': objects_count
                }
            
            # Parallel feature computation AND fitness evaluation (with judges)
            evaluations = []
            with ThreadPoolExecutor(max_workers=min(len(candidate_images), 4)) as eval_executor:
                futures = [eval_executor.submit(evaluate_candidate, i, candidate) for i, candidate in enumerate(candidate_images)]
                for future in as_completed(futures):
                    evaluations.append(future.result())
            
            # Update candidates with evaluation results and update archive
            iteration_best_fitness = 0.0
            iteration_best_candidate = None
            
            for eval_result in evaluations:
                i = eval_result['index']
                candidate = candidate_images[i]
                candidate['feature_bin'] = eval_result['feature_bin']
                candidate['fitness'] = eval_result['fitness']
                
                feature_bin = eval_result['feature_bin']
                fitness = eval_result['fitness']
                
                # Update archive if this is a new bin or better than existing elite
                if feature_bin not in self.archive or fitness > self.archive[feature_bin]['fitness']:
                    logging.info(f"🎯 NEW ELITE for bin {feature_bin}!\n")
                    self.archive[feature_bin] = {
                        'prompt': candidate['prompt'],
                        'image_bytes': candidate['image_bytes'],
                        'image': candidate['image'],
                        'fitness': fitness
                    }
                    
                    # Save elite
                    elite_save_path = os.path.join(results_dir, f"{base_filename}_elite_bin-{feature_bin}.jpg")
                    candidate['image'].save(elite_save_path)
                    with open(elite_save_path.replace(".jpg", ".txt"), "w") as f:
                        f.write(f"Feature bin: {feature_bin}\nFitness: {fitness:.2f}\n")
                        f.write(f"SSIM: {eval_result['ssim']:.3f}, Depth: {eval_result['depth']:.1f}, Objects: {eval_result['objects']}\n")
                        f.write(f"\nPrompt:\n{candidate['prompt']}")
                
                # Track best of this iteration
                if fitness > iteration_best_fitness:
                    iteration_best_fitness = fitness
                    iteration_best_candidate = candidate
            
            eval_duration = time.time() - eval_start
            print(f"⏱️  [Iter {iter + 1}] Evaluated all candidates: {eval_duration:.2f}s")
            
            logging.info(f"\n📊 Archive size: {len(self.archive)} elites\n")
            
            # Update best overall if this iteration produced something better
            if iteration_best_candidate and iteration_best_fitness > best_fitness:
                best_fitness = iteration_best_fitness
                best_prompt = iteration_best_candidate['prompt']
                best_image = iteration_best_candidate['image']
                context_image_bytes = iteration_best_candidate['image_bytes']
                logging.info(f"✨ NEW OVERALL BEST from iteration {iter + 1}!\n")
                logging.info(f"Best prompt: {best_prompt}\n")
                logging.info(f"Best fitness: {best_fitness:.2f}\n")
            
            # Save current best
            best_save_path = os.path.join(results_dir, f"{base_filename}_iter-{iter + 1}-best.jpg")
            best_image.save(best_save_path)
            with open(best_save_path.replace(".jpg", ".txt"), "w") as f:
                f.write(f"Best fitness: {best_fitness:.2f}\n\nPrompt:\n{best_prompt}")
            
            # Update history
            if self.use_history_of_prompts:
                history_of_prompts.append(best_prompt)

            iter_duration = time.time() - iter_start_time
            print(f"⏱️  [Iter {iter + 1}] TOTAL iteration time: {iter_duration:.2f}s")
            print(f"{'=' * 60}")

            # Update progress at end of iteration
            pbar.update(1)

        # Final save
        final_save_start = time.time()
        final_best_path = os.path.join(results_dir, f"{base_filename}_final-best.jpg")
        best_image.save(final_best_path)
        
        final_prompt_path = os.path.join(results_dir, f"{base_filename}_final-prompt.txt")
        with open(final_prompt_path, "w") as f:
            f.write(f"Best fitness: {best_fitness:.2f}\n\nBest prompt:\n{best_prompt}")
        
        # Save archive summary
        archive_summary_path = os.path.join(results_dir, f"{base_filename}_archive-summary.txt")
        with open(archive_summary_path, "w") as f:
            f.write(f"MAP-Elites Archive Summary\n")
            f.write(f"Total elites: {len(self.archive)}\n\n")
            for bin_coords, elite in sorted(self.archive.items()):
                f.write(f"Bin {bin_coords}:\n")
                f.write(f"  Fitness: {elite['fitness']:.2f}\n")
                f.write(f"  Prompt: {elite['prompt']}\n\n")
        
        logging.info(f"Saved final best image to: {final_best_path}\n")
        logging.info(f"Best prompt:\n{best_prompt}\n")
        logging.info(f"Best fitness: {best_fitness:.2f}\n")
        logging.info(f"Archive size: {len(self.archive)} elites\n")
        
        final_save_duration = time.time() - final_save_start
        print(f"⏱️  Final save: {final_save_duration:.2f}s")

        image_duration = time.time() - image_start_time
        print(f"\n⏱️  ✅ TOTAL time for {base_filename}: {image_duration:.2f}s ({image_duration/60:.2f}m)\n")

    def _evaluate(self, edited_image_bytes: bytes, original_image_bytes: bytes, context_image_bytes: bytes) -> tuple[str, str]:
        """
        Evaluates the edited image against the original (or context) image using multiple judges.
        Only counts judges where both orderings agree (order-independent judgments).
        Returns majority vote from consistent judges and aggregated reasons.
        """
        eval_total_start = time.time()
        comparison_image_bytes = context_image_bytes if context_image_bytes else original_image_bytes

        def evaluate_single(judge_id: int, is_edited_first: bool):
            """Single evaluation task"""
            judge_start = time.time()
            logging.info(f"Judge {judge_id}: Evaluating with edited image as {'first' if is_edited_first else 'second'} image.\n")

            images = [edited_image_bytes, comparison_image_bytes] if is_edited_first else [comparison_image_bytes, edited_image_bytes]
            choice_map = {
                "first": "edited" if is_edited_first else "original",
                "second": "original" if is_edited_first else "edited",
            }

            logging.info(f"Judge {judge_id} choice map: {choice_map}\n")

            model_call_start = time.time()
            evaluation = self.evaluator_model.get_response(
                task=self.evaluator_prompt,
                images=images
            )
            model_call_duration = time.time() - model_call_start

            real_choice = choice_map.get(evaluation.choice.lower())

            if not real_choice:
                logging.warning(f"Judge {judge_id}: Evaluation failed. Skipping.\n")
                return None

            judge_duration = time.time() - judge_start
            print(f"  ⏱️  Judge {judge_id} ({'edited 1st' if is_edited_first else 'orig 1st'}): {judge_duration:.2f}s (model: {model_call_duration:.2f}s)")
            logging.info(f"Judge {judge_id}: {evaluation}\n")
            return (real_choice, evaluation.reason)

        # Run all evaluations in parallel
        parallel_start = time.time()
        judge_results = {}  # judge_id -> {True: result, False: result}

        with ThreadPoolExecutor(max_workers=min(self.num_judges * 2, 8)) as eval_executor:
            # Submit all evaluation tasks
            submit_start = time.time()
            future_to_judge = {}
            for judge_id in range(self.num_judges):
                for is_edited_first in [True, False]:
                    future = eval_executor.submit(evaluate_single, judge_id, is_edited_first)
                    future_to_judge[future] = (judge_id, is_edited_first)
            submit_duration = time.time() - submit_start
            print(f"  ⏱️  Submitted {len(future_to_judge)} judge tasks: {submit_duration:.2f}s")

            # Collect results organized by judge
            collect_start = time.time()
            for future in as_completed(future_to_judge):
                judge_id, is_edited_first = future_to_judge[future]
                result = future.result()

                if judge_id not in judge_results:
                    judge_results[judge_id] = {}
                judge_results[judge_id][is_edited_first] = result
            collect_duration = time.time() - collect_start
            print(f"  ⏱️  Collected all judge results: {collect_duration:.2f}s")

        parallel_duration = time.time() - parallel_start
        print(f"  ⏱️  Parallel evaluation total: {parallel_duration:.2f}s")

        # Only count judges where both orderings agree (order-independent)
        aggregation_start = time.time()
        consistent_choices = {}
        consistent_reasons = {}

        for judge_id, results in judge_results.items():
            result_edited_first = results.get(True)
            result_original_first = results.get(False)

            # Both evaluations must succeed
            if result_edited_first is None or result_original_first is None:
                logging.warning(f"Judge {judge_id}: One or both evaluations failed. Skipping judge.\n")
                continue

            choice_edited_first, reason_edited_first = result_edited_first
            choice_original_first, reason_original_first = result_original_first

            # Only count if both orderings agree
            if choice_edited_first == choice_original_first:
                logging.info(f"Judge {judge_id}: Consistent choice '{choice_edited_first}' across both orderings.\n")
                consistent_choices[choice_edited_first] = consistent_choices.get(choice_edited_first, 0) + 1
                if choice_edited_first not in consistent_reasons:
                    consistent_reasons[choice_edited_first] = []
                consistent_reasons[choice_edited_first].append(reason_edited_first)
            else:
                logging.warning(f"Judge {judge_id}: Inconsistent - chose '{choice_edited_first}' when edited first, '{choice_original_first}' when original first. Skipping judge.\n")

        # Determine final choice
        if not consistent_choices:
            # No consistent judges - default to original (conservative: don't change if truly indistinguishable)
            logging.warning("No judges were consistent across both orderings. Defaulting to 'original'.\n")
            choice = "original"
            reason = "No order-independent preference detected."
        else:
            # Use majority vote from consistent judges
            choice = max(consistent_choices, key=consistent_choices.get)
            reason = "\n".join(consistent_reasons[choice])
            logging.info(f"Final decision: '{choice}' (based on {consistent_choices[choice]} consistent judge(s)).\n")
        
        reasoning_log_path = os.path.join("evaluation_logs", f"evaluation_reasoning_{int(time.time())}.txt")
        os.makedirs("evaluation_logs", exist_ok=True)
        with open(reasoning_log_path, "w") as f:
            f.write(f"Final choice: {choice}\n\nReasons from consistent judges:\n{reason}\n")
        logging.info(f"Saved evaluation reasoning log to: {reasoning_log_path}\n")

        aggregation_duration = time.time() - aggregation_start
        print(f"  ⏱️  Judge aggregation: {aggregation_duration:.2f}s")

        eval_total_duration = time.time() - eval_total_start
        print(f"  ⏱️  TOTAL _evaluate(): {eval_total_duration:.2f}s")

        return choice, reason

    def run(self, image_paths: list[str], results_dir: str, max_workers: int = 1):
        """
        Runs the MAP-Elites optimization loop for each image, optionally in parallel.
        """
        image_paths = image_paths[:3]
        run_start_time = time.time()
        print(f"\n{'=' * 60}")
        print(f"🚀 Starting VisualNudge MAP-Elites pipeline")
        print(f"   Images: {len(image_paths)}")
        print(f"   Iterations per image: {self.iterations}")
        print(f"   Variations per iteration: {self.map_elites_variations}")
        print(f"   Feature bins per dimension: {self.feature_bins}")
        print(f"   Max workers: {max_workers}")
        print(f"   Total iterations: {len(image_paths) * self.iterations}")
        print(f"{'=' * 60}\n")

        total_iterations = len(image_paths) * self.iterations

        with tqdm(total=total_iterations, desc="Total progress", unit="iter", position=0) as pbar_total:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_single_image, image_path, idx, len(image_paths), results_dir, pbar_total): image_path
                          for idx, image_path in enumerate(image_paths)}

                for future in tqdm(as_completed(futures), total=len(image_paths), desc="Images completed", unit="image", position=1, leave=False):
                    future.result()

        run_duration = time.time() - run_start_time
        print(f"\n{'=' * 60}")
        print(f"✅ MAP-Elites pipeline complete!")
        print(f"⏱️  TOTAL RUNTIME: {run_duration:.2f}s ({run_duration/60:.2f}m)")
        print(f"   Average per image: {run_duration/len(image_paths):.2f}s")
        print(f"   Average per iteration: {run_duration/total_iterations:.2f}s")
        print(f"   Final archive size: {len(self.archive)} elites")
        print(f"{'=' * 60}\n")
