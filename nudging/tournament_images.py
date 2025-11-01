import os
import io
import logging
import dataclasses
from PIL import Image
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import ImageModel, LanguageModel

@dataclasses.dataclass
class VisualNudge:
    """
    Orchestrates the visual nudging pipeline.
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

    # Two-stage optimizer configuration
    num_proposals: int
    proposer_sees_current_prompt: bool
    proposer_sees_history: bool
    selector_sees_current_prompt: bool
    selector_sees_history: bool
    proposer_model: LanguageModel
    selector_model: LanguageModel

    # Legacy support: keep optimizer_model for backwards compatibility
    optimizer_model: LanguageModel = None

    def _process_single_image(
        self,
        image_path: str,
        img_idx: int,
        total_images: int,
        results_dir: str,
        pbar: tqdm
    ) -> None:
        """
        Processes a single image through the optimization loop.
        """
        base_filename, _ = os.path.splitext(os.path.basename(image_path))
        logging.info(f"\n===== Processing Image {img_idx + 1}/{total_images}: {base_filename} =====\n")

        with open(image_path, "rb") as f:
            original_image_bytes = f.read()

            original_image = Image.open(io.BytesIO(original_image_bytes))
            original_save_path = os.path.join(results_dir, f"{base_filename}_iter-0-original.jpg")
            original_image.save(original_save_path)
            logging.info(f"Saved original image to: {original_save_path}\n")

            current_prompt = self.initial_prompt
            best_prompt = current_prompt
            best_image = original_image
            context_image_bytes = original_image_bytes
            
            history_of_prompts = [current_prompt] if self.use_history_of_prompts else []

            for iter in range(self.iterations + 1):
                logging.info("\n>> ITERATION " + ("BEST" if iter == self.iterations else f"{iter + 1}/{self.iterations}") + " <<\n")
                logging.info(f"PROMPT:\n{current_prompt}\n")

                # Edit image (using the original image for the first iteration)
                editing_prompt = f"{current_prompt}\n{self.editing_context_prompt}"
                edited_image, edited_image_bytes = self.image_editing_model.edit(
                    f"{editing_prompt}\n{self.background_state_prompt}",
                    original_image_bytes,
                    context_image_bytes
                )

                edited_image_save_path = os.path.join(results_dir, f"{base_filename}_iter-{iter + 1}-a_edited.jpg")
                edited_image.save(edited_image_save_path)
                logging.info(f"Saved edited image to: {edited_image_save_path}\n")
                
                with open(edited_image_save_path.replace(".jpg", ".txt"), "w") as f:
                    f.write(editing_prompt)

                # Evaluate edit
                selected_choice, aggregated_reason = self._evaluate(
                    edited_image_bytes=edited_image_bytes,
                    original_image_bytes=original_image_bytes,
                    context_image_bytes=context_image_bytes
                )

                if selected_choice.lower() == "edited":
                    logging.info("VLM preferred the new image. Updating context for next iteration.\n")
                    logging.info("New best prompt: %s\n" % current_prompt)
                    context_image_bytes = edited_image_bytes
                    best_prompt = current_prompt
                    best_image = edited_image
                elif selected_choice.lower() == "original":
                    best_image = Image.open(io.BytesIO(context_image_bytes))
                    logging.info("VLM preferred the old image. Retaining previous context for next iteration.\n")
                else:
                    raise ValueError("Unexpected choice from evaluator. Got %s" % selected_choice)
                
                best_image_save_path = os.path.join(results_dir, f"{base_filename}_iter-{iter + 1}-b_best.jpg")
                best_image.save(best_image_save_path)
                logging.info(f"Saved best image to: {best_image_save_path}\n")
                with open(best_image_save_path.replace(".jpg", ".txt"), "w") as f:
                    f.write(f"Prompt leading to best image:\n{best_prompt}")

                if iter == self.iterations:
                    best_image_save_path = os.path.join(results_dir, f"{base_filename}_iter-n-edit.jpg")
                    best_image.save(best_image_save_path)
                    best_prompt_save_path = os.path.join(results_dir, f"{base_filename}_iter-n-prompt.txt")
                    with open(best_prompt_save_path, "w") as f:
                        f.write(best_prompt)
                    logging.info(f"Saved best image to: {best_image_save_path}\n")
                    logging.info(f"Best prompt:\n{best_prompt}\n")
                    break
                
                # Get new prompt using two-stage optimizer (proposer + selector)
                # Stage 1: Proposer generates k candidate prompts
                proposer_response = self.proposer_model.get_response(
                    reason=aggregated_reason,
                    current_prompt=best_prompt if self.proposer_sees_current_prompt else "",
                    history_of_prompts=("\n".join([f"{i}. {p}" for i, p in enumerate(history_of_prompts)])
                                      if self.use_history_of_prompts and self.proposer_sees_history else ""),
                    current_iteration=iter + 1,
                    total_iterations=self.iterations,
                    num_proposals=self.num_proposals
                )

                candidate_prompts = proposer_response.candidate_prompts
                logging.info(f"PROPOSER generated {len(candidate_prompts)} candidates:\n")
                for i, prompt in enumerate(candidate_prompts):
                    logging.info(f"  Candidate {i+1}: {prompt}\n")

                # Stage 2: Selector chooses the best candidate
                selector_response = self.selector_model.get_response(
                    candidate_prompts=candidate_prompts,
                    reason=aggregated_reason,
                    current_prompt=best_prompt if self.selector_sees_current_prompt else "",
                    history_of_prompts=("\n".join([f"{i}. {p}" for i, p in enumerate(history_of_prompts)])
                                      if self.use_history_of_prompts and self.selector_sees_history else ""),
                    current_iteration=iter + 1,
                    total_iterations=self.iterations
                )

                # Update the prompt for the next iteration
                current_prompt = selector_response.selected_prompt

                logging.info(f"SELECTOR chose prompt:\n{current_prompt}\n")


                # Update progress at end of iteration
                pbar.update(1)

    def _evaluate(self, edited_image_bytes: bytes, original_image_bytes: bytes, context_image_bytes: bytes) -> tuple[str, str]:
        """
        Evaluates the edited image against the original (or context) image using multiple judges.
        Only counts judges where both orderings agree (order-independent judgments).
        Returns majority vote from consistent judges and aggregated reasons.
        """
        comparison_image_bytes = context_image_bytes if context_image_bytes else original_image_bytes

        def evaluate_single(judge_id: int, is_edited_first: bool):
            """Single evaluation task"""
            logging.info(f"Judge {judge_id}: Evaluating with edited image as {'first' if is_edited_first else 'second'} image.\n")

            images = [edited_image_bytes, comparison_image_bytes] if is_edited_first else [comparison_image_bytes, edited_image_bytes]
            choice_map = {
                "first": "edited" if is_edited_first else "original",
                "second": "original" if is_edited_first else "edited",
            }

            logging.info(f"Judge {judge_id} choice map: {choice_map}\n")

            evaluation = self.evaluator_model.get_response(
                task=self.evaluator_prompt,
                images=images
            )

            real_choice = choice_map.get(evaluation.choice.lower())

            if not real_choice:
                logging.warning(f"Judge {judge_id}: Evaluation failed. Skipping.\n")
                return None

            logging.info(f"Judge {judge_id}: {evaluation}\n")
            return (real_choice, evaluation.reason)

        # Run all evaluations in parallel
        judge_results = {}  # judge_id -> {True: result, False: result}

        with ThreadPoolExecutor(max_workers=min(self.num_judges * 2, 8)) as eval_executor:
            # Submit all evaluation tasks
            future_to_judge = {}
            for judge_id in range(self.num_judges):
                for is_edited_first in [True, False]:
                    future = eval_executor.submit(evaluate_single, judge_id, is_edited_first)
                    future_to_judge[future] = (judge_id, is_edited_first)

            # Collect results organized by judge
            for future in as_completed(future_to_judge):
                judge_id, is_edited_first = future_to_judge[future]
                result = future.result()

                if judge_id not in judge_results:
                    judge_results[judge_id] = {}
                judge_results[judge_id][is_edited_first] = result

        # Only count judges where both orderings agree (order-independent)
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

        return choice, reason

    def run(self, image_paths: List[str], results_dir: str, max_workers: int = 1):
        """
        Runs the optimization loop for each image, optionally in parallel.
        """
        total_iterations = len(image_paths) * (self.iterations + 1)

        with tqdm(total=total_iterations, desc="Total progress", unit="iter", position=0) as pbar_total:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_single_image, image_path, idx, len(image_paths), results_dir, pbar_total): image_path
                          for idx, image_path in enumerate(image_paths)}

                for future in tqdm(as_completed(futures), total=len(image_paths), desc="Images completed", unit="image", position=1, leave=False):
                    future.result()
