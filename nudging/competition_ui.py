import os
import io
import time
import logging
import threading
import dataclasses
from typing import Optional
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import ImageModel, LanguageModel


@dataclasses.dataclass
class VisualNudgeUICompetition:
    """
    Optimizes UI-only variations of the same product image.
    Generates two zero-shot UI placements, then iteratively improves the loser
    via proposer/selector while keeping the product constant.
    """
    name: str
    ui_zero_shot_prompts: list[str]
    base_prior: str
    editing_context_prompt: str
    background_state_prompt: str
    image_editing_model: ImageModel
    evaluator_model: LanguageModel
    proposer_model: LanguageModel
    selector_model: LanguageModel
    num_improvement_proposals: int
    n_evaluations: int = 1
    equilibrium_threshold: float = 0.52
    min_rounds_before_equilibrium: int = 5
    max_rounds_per_image: int = 10
    use_last_winner_as_base: bool = True

    def __post_init__(self):
        if len(self.ui_zero_shot_prompts) < 2:
            raise ValueError("ui_zero_shot_prompts must contain at least two prompts.")

        self._total_num_images_generated = 0
        self._cost_per_image_generated = 0.039
        self._counter_lock = threading.Lock()

    def _compose_prompt(self, ui_instruction: Optional[str] = None) -> str:
        """Merge base prior with UI-specific instruction."""
        base = (self.base_prior or "").strip()
        ui_instruction = (ui_instruction or "").strip()

        if base and ui_instruction:
            return f"{base}\n\n{ui_instruction}"
        return ui_instruction or base

    def _build_full_prompt(self, ui_instruction: Optional[str]) -> str:
        """Append editing context and background guidance."""
        parts = [self._compose_prompt(ui_instruction)]
        if self.editing_context_prompt:
            parts.append(self.editing_context_prompt.strip())
        if self.background_state_prompt:
            parts.append(self.background_state_prompt.strip())
        return "\n".join(part for part in parts if part).strip()

    def _conduct_contest(
        self,
        image_a_bytes: bytes,
        image_b_bytes: bytes,
        image_a_name: str,
        image_b_name: str
    ) -> tuple[str, float, str]:
        """Run multi-judge evaluation between two UI variants."""
        contest_start = time.time()
        logging.info(f"\n🖥️ UI CONTEST: {image_a_name} vs {image_b_name}\n")

        def evaluate_single(judge_id: int, is_a_first: bool):
            judge_start = time.time()

            images = [image_a_bytes, image_b_bytes] if is_a_first else [image_b_bytes, image_a_bytes]
            choice_map = {
                "first": image_a_name if is_a_first else image_b_name,
                "second": image_b_name if is_a_first else image_a_name,
            }

            evaluation = self.evaluator_model.get_response(
                images=images,
                metadata=""
            )

            real_choice = choice_map.get(evaluation.choice.lower())

            logging.info(
                f"Judge {judge_id} ({'A first' if is_a_first else 'B first'}): "
                f"Chose {real_choice} - {evaluation.reason}\n"
            )
            logging.debug(f"  ⏱️ Judge {judge_id}: {time.time() - judge_start:.2f}s")
            return real_choice, evaluation.reason

        judge_results = {}
        num_judges = self.n_evaluations
        with ThreadPoolExecutor(max_workers=min(num_judges * 2, 8)) as executor:
            futures = {}
            for judge_id in range(num_judges):
                for is_a_first in [True, False]:
                    future = executor.submit(evaluate_single, judge_id, is_a_first)
                    futures[future] = (judge_id, is_a_first)

            for future in as_completed(futures):
                judge_id, is_a_first = futures[future]
                result = future.result()
                judge_results.setdefault(judge_id, {})
                judge_results[judge_id][is_a_first] = result

        votes = {image_a_name: 0, image_b_name: 0}
        feedback_by_choice = {image_a_name: [], image_b_name: []}
        consistent = 0

        for judge_id, outcomes in judge_results.items():
            if len(outcomes) < 2:
                continue
            result_a = outcomes.get(True)
            result_b = outcomes.get(False)
            if not result_a or not result_b:
                continue

            choice_a, reason_a = result_a
            choice_b, reason_b = result_b
            if choice_a == choice_b:
                consistent += 1
                votes[choice_a] += 1
                feedback = reason_a if choice_a == image_a_name else reason_b
                feedback_by_choice[choice_a].append(feedback)
            else:
                logging.warning(f"Judge {judge_id} inconsistent ({choice_a} vs {choice_b})\n")

        if consistent == 0:
            logging.warning("No consistent judges. Treating as draw.\n")
            return image_a_name, 0.5, "No consistent preference."

        winner = max(votes, key=votes.get)
        winner_score = votes[winner] / consistent
        feedback = "\n".join(feedback_by_choice[winner]) or "No feedback."

        logging.info(f"🏆 UI WINNER: {winner} ({votes[winner]}/{consistent} = {winner_score:.1%})\n")
        logging.debug(f"⏱️ Contest duration: {time.time() - contest_start:.2f}s")

        return winner, winner_score, feedback

    def _select_best_candidate(self, candidates: list[dict], feedback: str) -> dict:
        """Pick best UI candidate using selector model."""
        image_bytes_list = [c["image_bytes"] for c in candidates]
        descriptions = [f"Candidate {i+1}: {c['prompt']}" for i, c in enumerate(candidates)]

        selector_response = self.selector_model.get_response(
            images=image_bytes_list,
            candidate_descriptions="\n".join(descriptions),
            num_candidates=len(candidates),
            judge_feedback=feedback
        )

        selected_idx = int(selector_response.choice) - 1
        best = candidates[selected_idx]
        logging.info(f"✅ Selector chose candidate {selected_idx+1}: {best['prompt']}\n")
        return best

    def _improve_variant(
        self,
        variant_label: str,
        variant_state: dict,
        feedback: str,
        round_num: int,
        pair_name: str,
        results_dir: str,
        original_image_bytes: bytes
    ) -> None:
        """Run proposer/selector improvement on the provided variant."""
        history_text = "\n".join([f"  - {p}" for p in variant_state["edit_history"]]) or "None"
        logging.info(f"Edit history for {variant_label}:\n{history_text}\n")

        proposer_response = self.proposer_model.get_response(
            current_prompt=variant_state["prompt"],
            history_of_prompts=history_text,
            current_iteration=round_num,
            judge_feedback=feedback,
            total_iterations=self.min_rounds_before_equilibrium,
            num_proposals=self.num_improvement_proposals,
            metadata="Optimize UI for the same product without changing the product itself."
        )

        candidate_prompts = proposer_response.candidate_prompts
        logging.info(f"Generated {len(candidate_prompts)} UI improvement candidates.\n")

        def generate_candidate(idx: int, prompt: str):
            logging.info(f"[{variant_label}] Rendering candidate {idx+1}/{len(candidate_prompts)}\n")
            full_prompt = self._build_full_prompt(prompt)
            edited_image, edited_bytes = self.image_editing_model.edit(
                full_prompt,
                variant_state["bytes"],
                original_image_bytes
            )

            if edited_image is None or edited_bytes is None:
                logging.warning(f"Candidate {idx+1} failed for {variant_label}.\n")
                return None

            with self._counter_lock:
                self._total_num_images_generated += 1

            candidate_path = os.path.join(
                results_dir,
                f"{pair_name}_{variant_label}_round-{round_num}_candidate-{idx+1}.jpg"
            )
            edited_image.save(candidate_path)
            return {
                "prompt": prompt,
                "image": edited_image,
                "image_bytes": edited_bytes,
                "full_prompt": full_prompt,
                "save_path": candidate_path
            }

        candidates = []
        with ThreadPoolExecutor(max_workers=min(len(candidate_prompts), 4)) as executor:
            futures = [executor.submit(generate_candidate, idx, prompt)
                       for idx, prompt in enumerate(candidate_prompts)]
            for future in as_completed(futures):
                candidate = future.result()
                if candidate:
                    candidates.append(candidate)

        if not candidates:
            logging.warning(f"No successful candidates for {variant_label}; keeping previous state.\n")
            return

        best_candidate = self._select_best_candidate(candidates, feedback)

        variant_state["bytes"] = best_candidate["image_bytes"]
        variant_state["prompt"] = best_candidate["prompt"]
        variant_state["image"] = best_candidate["image"]
        variant_state["edit_history"].append(best_candidate["prompt"])

        optimized_path = os.path.join(
            results_dir,
            f"{pair_name}_{variant_label}_round-{round_num}_optimized.jpg"
        )
        best_candidate["image"].save(optimized_path)
        prompt_path = optimized_path.replace(".jpg", ".txt")
        with open(prompt_path, "w") as outfile:
            outfile.write(best_candidate["full_prompt"])

        with self._counter_lock:
            total_cost = self._total_num_images_generated * self._cost_per_image_generated
        logging.info(
            f"Updated {variant_label}. Total images: {self._total_num_images_generated}, "
            f"Estimated cost: ${total_cost:.2f}\n"
        )

    def _generate_zero_shot_variants(
        self,
        original_bytes: bytes,
        original_image: Image.Image,
        base_name: str,
        results_dir: str,
        pair_name: str
    ) -> list[dict]:
        """Create the initial UI contestants via zero-shot prompts."""
        variants = []

        for idx, prompt in enumerate(self.ui_zero_shot_prompts[:2]):
            logging.info(f"Rendering zero-shot UI variant {idx+1}: {prompt}\n")
            full_prompt = self._build_full_prompt(prompt)
            edited_image, edited_bytes = self.image_editing_model.edit(
                full_prompt,
                original_bytes,
                original_bytes
            )

            if edited_image is None or edited_bytes is None:
                logging.warning(f"Zero-shot generation {idx+1} failed for {base_name}.\n")
                continue

            with self._counter_lock:
                self._total_num_images_generated += 1

            label = f"{base_name}_ui-{idx+1}"
            save_path = os.path.join(results_dir, f"{label}_zero-shot.jpg")
            edited_image.save(save_path)
            with open(save_path.replace(".jpg", ".txt"), "w") as prompt_file:
                prompt_file.write(full_prompt)

            variant_state = {
                "label": label,
                "bytes": edited_bytes,
                "prompt": prompt,
                "image": edited_image,
                "edit_history": [prompt],
                "champion_bytes": edited_bytes,
                "champion_prompt": prompt,
                "champion_image": edited_image.copy()
            }
            variants.append(variant_state)

        return variants

    def _process_single_image(
        self,
        image_path: str,
        index: int,
        total: int,
        results_dir: str
    ) -> Optional[dict]:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        pair_name = f"ui-{base_name}"
        logging.info(f"\n{'='*80}\nUI COMPETITION {index}/{total}: {base_name}\n{'='*80}\n")

        with open(image_path, "rb") as f:
            original_bytes = f.read()
        original_image = Image.open(io.BytesIO(original_bytes))

        original_path = os.path.join(results_dir, f"{base_name}_original.jpg")
        if not os.path.isfile(original_path):
            original_image.save(original_path)

        variants = self._generate_zero_shot_variants(
            original_bytes,
            original_image,
            base_name,
            results_dir,
            pair_name
        )

        if len(variants) < 2:
            logging.warning(f"Insufficient zero-shot UI variants for {base_name}; skipping.\n")
            return None

        state_a, state_b = variants[:2]

        round_num = 0
        equilibrium = False
        while round_num < self.max_rounds_per_image and not equilibrium:
            round_num += 1
            logging.info(f"\n-- UI ROUND {round_num}/{self.max_rounds_per_image} --\n")

            winner_name, winner_score, feedback = self._conduct_contest(
                state_a["bytes"], state_b["bytes"],
                state_a["label"], state_b["label"]
            )

            if winner_name == state_a["label"]:
                winner_state, loser_state = state_a, state_b
            else:
                winner_state, loser_state = state_b, state_a

            winner_state["image"].save(os.path.join(
                results_dir, f"{pair_name}_{winner_state['label']}_round-{round_num}_WINNER.jpg"
            ))
            loser_state["image"].save(os.path.join(
                results_dir, f"{pair_name}_{loser_state['label']}_round-{round_num}.jpg"
            ))

            winner_state["champion_bytes"] = winner_state["bytes"]
            winner_state["champion_prompt"] = winner_state["prompt"]
            winner_state["champion_image"] = winner_state["image"].copy()

            if self.use_last_winner_as_base:
                loser_state["bytes"] = loser_state["champion_bytes"]
                loser_state["prompt"] = loser_state["champion_prompt"]
                loser_state["image"] = loser_state["champion_image"].copy()

            if round_num >= self.min_rounds_before_equilibrium and winner_score < self.equilibrium_threshold:
                logging.info(f"🎯 UI equilibrium reached at round {round_num} (score {winner_score:.1%}).\n")
                equilibrium = True
                break

            logging.info(f"Improving UI variant: {loser_state['label']} (loser of round {round_num})\n")
            self._improve_variant(
                loser_state["label"],
                loser_state,
                feedback,
                round_num,
                pair_name,
                results_dir,
                original_bytes
            )

        final_a_path = os.path.join(results_dir, f"{pair_name}_{state_a['label']}_final.jpg")
        final_b_path = os.path.join(results_dir, f"{pair_name}_{state_b['label']}_final.jpg")
        state_a["image"].save(final_a_path)
        state_b["image"].save(final_b_path)

        summary_path = os.path.join(results_dir, f"{pair_name}_summary.txt")
        with open(summary_path, "w") as summary:
            summary.write(f"UI Competition Summary for {base_name}\n")
            summary.write(f"{'='*60}\n")
            summary.write(f"Rounds executed: {round_num}\n")
            summary.write(f"Equilibrium reached: {equilibrium}\n\n")
            summary.write(f"{state_a['label']} history ({len(state_a['edit_history'])} edits):\n")
            for idx, prompt in enumerate(state_a["edit_history"], 1):
                summary.write(f"  {idx}. {prompt}\n")
            summary.write(f"\n{state_b['label']} history ({len(state_b['edit_history'])} edits):\n")
            for idx, prompt in enumerate(state_b["edit_history"], 1):
                summary.write(f"  {idx}. {prompt}\n")

        return {
            "image": base_name,
            "rounds": round_num,
            "equilibrium": equilibrium
        }

    def run(self, image_paths: list[str], results_dir: str, max_workers: int = 1):
        """Run UI competitions in parallel across products."""
        os.makedirs(results_dir, exist_ok=True)
        logging.info(f"Starting UI competition: {len(image_paths)} product images\n")

        results = []
        total = len(image_paths)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_single_image, path, idx + 1, total, results_dir): path
                for idx, path in enumerate(image_paths)
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        completed = sum(1 for r in results if r["equilibrium"])
        logging.info(
            f"\nUI competition finished for {len(results)} images "
            f"({completed} reached equilibrium). Total images generated: {self._total_num_images_generated} "
            f"(~${self._total_num_images_generated * self._cost_per_image_generated:.2f}).\n"
        )
