import os
import io
import time
import logging
import dataclasses
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.wrappers import ImageModel, LanguageModel
from itertools import combinations


@dataclasses.dataclass
class VisualNudgeCompetition:
    """
    Orchestrates paired contests between images, iteratively improving losers until equilibrium.
    Each pair of images competes, and the loser is improved based on judge feedback.
    Process continues until neither image can be improved further (equilibrium).
    """
    name: str
    editing_context_prompt: str
    initial_prompt: str
    background_state_prompt: str
    image_editing_model: ImageModel
    num_judges: int
    evaluator_model: LanguageModel

    # Optimizer for improving losers
    num_improvement_proposals: int
    proposer_model: LanguageModel
    selector_model: LanguageModel
    
    # Equilibrium detection
    equilibrium_threshold: float = 0.51  # Win rate below this (close to 0.5) indicates equilibrium
    min_rounds_before_equilibrium: int = 10  # Minimum rounds before checking for equilibrium
    max_rounds_per_pair: int = 10  # Maximum improvement rounds per pair before declaring equilibrium

    def __post_init__(self):
        """Initialize tracking variables."""
        self._total_num_images_generated = 0
        self._cost_per_image_generated = 0.039
        
        self.contest_history = {}

    def _conduct_contest(
        self, 
        image_a_bytes: bytes, 
        image_b_bytes: bytes,
        image_a_name: str,
        image_b_name: str
    ) -> tuple[str, float, str]:
        """
        Conduct a multi-judge contest between two images.
        Returns: (winner_name, winner_score, aggregated_feedback)
        winner_score is the proportion of consistent judges preferring the winner (0-1)
        """
        contest_start = time.time()
        logging.info(f"\n🥊 CONTEST: {image_a_name} vs {image_b_name}\n")

        def evaluate_single(judge_id: int, is_a_first: bool):
            """Single judge evaluation"""
            judge_start = time.time()
            
            images = [image_a_bytes, image_b_bytes] if is_a_first else [image_b_bytes, image_a_bytes]
            choice_map = {
                "first": image_a_name if is_a_first else image_b_name,
                "second": image_b_name if is_a_first else image_a_name,
            }

            logging.info(f"Judge {judge_id}: Evaluating with {image_a_name} as {'first' if is_a_first else 'second'} image.\n")

            evaluation = self.evaluator_model.get_response(
                images=images
            )

            real_choice = choice_map.get(evaluation.choice.lower())

            judge_duration = time.time() - judge_start
            print(f"  ⏱️  Judge {judge_id} ({image_a_name} {'1st' if is_a_first else '2nd'}): {judge_duration:.2f}s")
            logging.info(f"Judge {judge_id}: Chose {real_choice} - {evaluation.reason}\n")
            return (real_choice, evaluation.reason)

        # Run all evaluations in parallel
        judge_results = {}  # judge_id -> {True: result, False: result}

        with ThreadPoolExecutor(max_workers=min(self.num_judges * 2, 8)) as eval_executor:
            future_to_judge = {}
            for judge_id in range(self.num_judges):
                for is_a_first in [True, False]:
                    future = eval_executor.submit(evaluate_single, judge_id, is_a_first)
                    future_to_judge[future] = (judge_id, is_a_first)

            for future in as_completed(future_to_judge):
                judge_id, is_a_first = future_to_judge[future]
                result = future.result()

                if judge_id not in judge_results:
                    judge_results[judge_id] = {}
                judge_results[judge_id][is_a_first] = result

        # Aggregate consistent judges
        votes = {image_a_name: 0, image_b_name: 0}
        feedback_by_choice = {image_a_name: [], image_b_name: []}
        total_consistent_judges = 0

        for judge_id, results in judge_results.items():
            result_a_first = results.get(True)
            result_b_first = results.get(False)

            choice_a_first, reason_a_first = result_a_first
            choice_b_first, reason_b_first = result_b_first

            # Only count consistent judges
            if choice_a_first == choice_b_first:
                logging.info(f"Judge {judge_id}: Consistent - chose '{choice_a_first}'\n")
                total_consistent_judges += 1
                votes[choice_a_first] += 1
                feedback_by_choice[choice_a_first].append(reason_a_first)
            else:
                logging.warning(f"Judge {judge_id}: Inconsistent - skipping.\n")

        # Determine winner
        if total_consistent_judges == 0:
            logging.warning("No consistent judges. Defaulting to draw.\n")
            winner = image_a_name  # Arbitrary - treat as draw
            winner_score = 0.5
            feedback = "No consistent preference detected."
        else:
            winner = max(votes, key=votes.get)
            winner_score = votes[winner] / total_consistent_judges
            feedback = "\n".join(feedback_by_choice[winner])
            
            logging.info(f"🏆 WINNER: {winner} ({votes[winner]}/{total_consistent_judges} = {winner_score:.2%})\n")

        contest_duration = time.time() - contest_start
        print(f"⏱️  Contest completed: {contest_duration:.2f}s")

        return winner, winner_score, feedback

    def _select_best_proposal(
        self,
        candidate_images: list[dict]
    ) -> dict:
        """
        Use selector model to choose the best proposal from candidates.
        Returns: The best candidate dict
        """
        logging.info(f"\n🎯 SELECTING BEST PROPOSAL from {len(candidate_images)} candidates\n")
        
        # Prepare images and descriptions for selector
        image_bytes_list = [c["image_bytes"] for c in candidate_images]
        descriptions = [f"Candidate {i+1}: {c['prompt']}" for i, c in enumerate(candidate_images)]
        
        selector_response = self.selector_model.get_response(
            images=image_bytes_list,
            candidate_descriptions="\n".join(descriptions),
            num_candidates=len(candidate_images)
        )
        
        selected_idx = int(selector_response.choice) - 1  # Assuming 1-indexed
        if selected_idx < 0 or selected_idx >= len(candidate_images):
            logging.warning(f"Invalid selection {selected_idx+1}, defaulting to first candidate\n")
            selected_idx = 0
        
        best_candidate = candidate_images[selected_idx]
        logging.info(f"✅ Selected candidate {selected_idx+1}: {best_candidate['prompt']}\n")
        
        return best_candidate

    def _improve_loser(
        self,
        loser_image_bytes: bytes,
        loser_prompt: str,
        feedback: str,
        round_num: int,
        results_dir: str,
        pair_name: str,
        history_of_prompts: list[str],
        original_image_bytes: bytes
    ) -> tuple[bytes, str, Image.Image]:
        """
        Generate improved versions of the losing image based on judge feedback.
        Uses selector model to choose best proposal (not tested against winner).
        Returns: (best_improved_bytes, best_improved_prompt, best_improved_image)
        """
        improve_start = time.time()
        logging.info(f"\n🔧 IMPROVING LOSER (Round {round_num})\n")
        logging.info(f"Previous prompt: {loser_prompt}\n")
        logging.info(f"Judge feedback:\n{feedback}\n")
        
        # Format history for proposer
        history_text = "\n".join([f"  - {p}" for p in history_of_prompts]) if history_of_prompts else "None"
        logging.info(f"Edit history:\n{history_text}\n")

        # Generate improvement proposals
        proposer_response = self.proposer_model.get_response(
            current_prompt=loser_prompt,
            judge_feedback=feedback,
            history_of_prompts=history_text,
            current_iteration=round_num,
            total_iterations=self.min_rounds_before_equilibrium,
            num_proposals=self.num_improvement_proposals,
            metadata="The product here is a(n) %s." % pair_name.split("_")[1]
        )
        print(f"⏱️  Proposal category is: {pair_name.split('_')[1]}")

        candidate_prompts = proposer_response.candidate_prompts
        logging.info(f"Generated {len(candidate_prompts)} improvement candidates:\n")
        for i, prompt in enumerate(candidate_prompts):
            logging.info(f"  Candidate {i+1}: {prompt}\n")

        # Generate images for all candidates in parallel
        def generate_single_image(i: int, prompt: str):
            """Generate a single improved image"""
            logging.info(f"Generating improved image {i+1}/{len(candidate_prompts)}\n")
            editing_prompt = f"{prompt}\n{self.editing_context_prompt}"
            edited_image, edited_image_bytes = self.image_editing_model.edit(
                f"{editing_prompt}\n{self.background_state_prompt}",
                loser_image_bytes,
                original_image_bytes
            )
            self._total_num_images_generated += 1
            
            # Save candidate
            candidate_path = os.path.join(results_dir, f"{pair_name}_round-{round_num}_candidate-{i+1}.jpg")
            edited_image.save(candidate_path)
            
            return {
                "prompt": prompt,
                "image": edited_image,
                "image_bytes": edited_image_bytes,
                "save_path": candidate_path
            }

        candidate_images = []
        with ThreadPoolExecutor(max_workers=min(len(candidate_prompts), 4)) as img_executor:
            futures = [img_executor.submit(generate_single_image, i, prompt) for i, prompt in enumerate(candidate_prompts)]
            for future in as_completed(futures):
                candidate_images.append(future.result())

        # Select best candidate using selector model (NOT by testing against winner)
        best_candidate = self._select_best_proposal(candidate_images)
        
        # Log cost
        total_cost = self._total_num_images_generated * self._cost_per_image_generated
        logging.info(f"Total images generated: {self._total_num_images_generated}, Cost: ${total_cost:.2f}\n")
        print(f"Total images: {self._total_num_images_generated}, Cost: ${total_cost:.2f}")

        improve_duration = time.time() - improve_start
        print(f"⏱️  Improvement phase: {improve_duration:.2f}s")

        return (
            best_candidate["image_bytes"],
            best_candidate["prompt"],
            best_candidate["image"]
        )

    def _add_winner_label(self, image: Image.Image, label: str = "WINNER") -> Image.Image:
        """Add a winner label to an image."""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Try to use a nice font, fall back to default if not available
        font = ImageFont.load_default()
        
        # Calculate text size and position
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position at top-center
        x = (img_copy.width - text_width) // 2
        y = 20
        
        # Draw background rectangle
        padding = 10
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill="gold",
            outline="black",
            width=3
        )
        
        # Draw text
        draw.text((x, y), label, fill="black", font=font)
        
        return img_copy

    def _visualize_competition(
        self,
        pair_name: str,
        base_a: str,
        base_b: str,
        round_history: list[dict],
        results_dir: str
    ):
        """
        Create grid display of competition progress.
        Columns: Round number, Contestant A, Contestant B
        Rows: Each round of competition
        """
        num_rounds = len(round_history)
        if num_rounds == 0:
            return
        
        # Create figure with 3 columns (round label, image A, image B)
        fig, axes = plt.subplots(num_rounds, 3, figsize=(15, 5 * num_rounds))
        
        # Handle single round case
        if num_rounds == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f"Competition Progress: {base_a} vs {base_b}", fontsize=16, fontweight="bold")
        
        for round_idx, round_data in enumerate(round_history):
            round_num = round_data["round"]
            winner = round_data["winner"]
            score = round_data["score"]
            
            # Round label column
            ax_label = axes[round_idx, 0]
            ax_label.text(0.5, 0.5, f"Round {round_num}\n{winner} wins\n{score:.1%}", 
                         ha="center", va="center", fontsize=14, fontweight="bold")
            ax_label.axis("off")
            
            # Image A column
            ax_a = axes[round_idx, 1]
            img_a = round_data["image_a"]
            if winner == base_a:
                img_a = self._add_winner_label(img_a, "WINNER")
            ax_a.imshow(img_a)
            ax_a.set_title(f"{base_a}", fontsize=12, fontweight="bold" if winner == base_a else "normal")
            ax_a.axis("off")
            
            # Image B column
            ax_b = axes[round_idx, 2]
            img_b = round_data["image_b"]
            if winner == base_b:
                img_b = self._add_winner_label(img_b, "WINNER")
            ax_b.imshow(img_b)
            ax_b.set_title(f"{base_b}", fontsize=12, fontweight="bold" if winner == base_b else "normal")
            ax_b.axis("off")
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(results_dir, f"{pair_name}_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logging.info(f"📊 Visualization saved to {viz_path}\n")

    def _run_paired_contest(
        self,
        image_a_path: str,
        image_b_path: str,
        results_dir: str,
        pair_idx: int,
        total_pairs: int
    ) -> dict:
        """
        Run a paired contest between two images until equilibrium is reached.
        Only the loser is improved each round.
        Returns final state of both images.
        """
        pair_start = time.time()
        
        base_a = os.path.splitext(os.path.basename(image_a_path))[0]
        base_b = os.path.splitext(os.path.basename(image_b_path))[0]
        pair_name = f"pair-{pair_idx+1}_{base_a}_vs_{base_b}"
        
        logging.info(f"\n{'='*80}\n")
        logging.info(f"PAIRED CONTEST {pair_idx+1}/{total_pairs}: {base_a} vs {base_b}\n")
        logging.info(f"{'='*80}\n")

        # Load initial images
        with open(image_a_path, "rb") as f:
            image_a_bytes = f.read()
        with open(image_b_path, "rb") as f:
            image_b_bytes = f.read()

        # Initialize state with edit history
        state_a = {
            "bytes": image_a_bytes,
            "prompt": self.initial_prompt,
            "image": Image.open(io.BytesIO(image_a_bytes)),
            "name": base_a,
            "edit_history": []  # Track all prompts used
        }
        state_b = {
            "bytes": image_b_bytes,
            "prompt": self.initial_prompt,
            "image": Image.open(io.BytesIO(image_b_bytes)),
            "name": base_b,
            "edit_history": []
        }

        # Save originals
        state_a["image"].save(os.path.join(results_dir, f"{pair_name}_{base_a}_original.jpg"))
        state_b["image"].save(os.path.join(results_dir, f"{pair_name}_{base_b}_original.jpg"))
        
        image_a_bytes_original = state_a["bytes"]
        image_b_bytes_original = state_b["bytes"]

        # Contest loop
        round_num = 0
        equilibrium_reached = False
        contest_history = []
        visualization_history = []  # For matplotlib visualization

        while round_num < self.max_rounds_per_pair and not equilibrium_reached:
            round_num += 1
            logging.info(f"\n{'='*60}\n")
            logging.info(f"ROUND {round_num}/{self.max_rounds_per_pair}\n")
            logging.info(f"{'='*60}\n")

            # Conduct contest
            winner_name, winner_score, feedback = self._conduct_contest(
                image_a_bytes=state_a["bytes"],
                image_b_bytes=state_b["bytes"],
                image_a_name=base_a,
                image_b_name=base_b
            )

            # Record history
            contest_history.append({
                "round": round_num,
                "winner": winner_name,
                "score": winner_score
            })
            
            # Store images for visualization
            visualization_history.append({
                "round": round_num,
                "winner": winner_name,
                "score": winner_score,
                "image_a": state_a["image"].copy(),
                "image_b": state_b["image"].copy()
            })

            # Determine winner and loser
            if winner_name == base_a:
                winner_state = state_a
                loser_state = state_b
                loser_name = base_b
            else:
                winner_state = state_b
                loser_state = state_a
                loser_name = base_a

            # Save round results with winner marked
            winner_img = self._add_winner_label(winner_state["image"], "WINNER")
            winner_img.save(os.path.join(results_dir, f"{pair_name}_{winner_name}_round-{round_num}_WINNER.jpg"))
            loser_state["image"].save(os.path.join(results_dir, f"{pair_name}_{loser_name}_round-{round_num}.jpg"))

            # Check for equilibrium (score close to 0.5 = no clear winner)
            if round_num >= self.min_rounds_before_equilibrium:
                if winner_score < self.equilibrium_threshold:
                    logging.info(f"\n🎯 EQUILIBRIUM REACHED! (score: {winner_score:.2%} < {self.equilibrium_threshold:.2%})\n")
                    equilibrium_reached = True
                    break

            # Improve ONLY the loser
            logging.info(f"\n🔄 Improving {loser_name} (loser of round {round_num})\n")
            
            improved_bytes, improved_prompt, improved_image = self._improve_loser(
                loser_image_bytes=loser_state["bytes"],
                loser_prompt=loser_state["prompt"],
                feedback=feedback,
                round_num=round_num,
                results_dir=results_dir,
                pair_name=pair_name,
                history_of_prompts=loser_state["edit_history"],
                original_image_bytes=image_a_bytes_original if loser_name == base_a else image_b_bytes_original
            )

            # Update ONLY loser's state
            loser_state["bytes"] = improved_bytes
            loser_state["prompt"] = improved_prompt
            loser_state["image"] = improved_image
            loser_state["edit_history"].append(improved_prompt)

        # Generate final visualization
        self._visualize_competition(
            pair_name=pair_name,
            base_a=base_a,
            base_b=base_b,
            round_history=visualization_history,
            results_dir=results_dir
        )

        # Save final results
        logging.info(f"\n{'='*60}\n")
        if equilibrium_reached:
            logging.info(f"✅ Equilibrium reached after {round_num} rounds\n")
        else:
            logging.info(f"⚠️  Max rounds ({self.max_rounds_per_pair}) reached\n")
        logging.info(f"{'='*60}\n")

        # Save finals
        state_a["image"].save(os.path.join(results_dir, f"{pair_name}_{base_a}_final.jpg"))
        state_b["image"].save(os.path.join(results_dir, f"{pair_name}_{base_b}_final.jpg"))

        # Save summary
        summary_path = os.path.join(results_dir, f"{pair_name}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Paired Contest Summary: {base_a} vs {base_b}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total rounds: {round_num}\n")
            f.write(f"Equilibrium reached: {equilibrium_reached}\n\n")
            f.write(f"Contest History:\n")
            for entry in contest_history:
                f.write(f"  Round {entry['round']}: {entry['winner']} won ({entry['score']:.2%})\n")
            f.write(f"\nFinal State:\n")
            f.write(f"\n{base_a}:\n")
            f.write(f"  Prompt: {state_a['prompt']}\n")
            f.write(f"  Edit History: {len(state_a['edit_history'])} edits\n")
            for i, edit in enumerate(state_a["edit_history"], 1):
                f.write(f"    {i}. {edit}\n")
            f.write(f"\n{base_b}:\n")
            f.write(f"  Prompt: {state_b['prompt']}\n")
            f.write(f"  Edit History: {len(state_b['edit_history'])} edits\n")
            for i, edit in enumerate(state_b["edit_history"], 1):
                f.write(f"    {i}. {edit}\n")

        pair_duration = time.time() - pair_start
        print(f"\n⏱️  Pair {pair_idx+1} total: {pair_duration:.2f}s ({pair_duration/60:.2f}m)\n")

        return {
            "pair_name": pair_name,
            "image_a": base_a,
            "image_b": base_b,
            "rounds": round_num,
            "equilibrium": equilibrium_reached,
            "history": contest_history,
            "final_state_a": state_a,
            "final_state_b": state_b
        }

    def run(self, image_paths: list[str], results_dir: str, max_workers: int = 1):
        """
        Run paired contests for all combinations of images.
        Each pair competes until equilibrium is reached.
        """
        run_start = time.time()
        
        # Generate all pairs
        image_categories = {}
        for image_path in image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            category = base_name.split("_")[0]  # Assuming category is prefix before the first underscore
            image_categories.setdefault(category, [])
            image_categories[category].append(image_path)
        
        pairs = []
        for category, paths in image_categories.items():
            category_pairs = list(combinations(paths, 2))
            pairs.extend(category_pairs)
        
        print(f"\n{'='*80}")
        print(f"🥊 Starting Paired Contest Competition")
        print(f"   Images: {len(image_paths)}")
        print(f"   Total pairs: {len(pairs)}")
        print(f"   Max rounds per pair: {self.max_rounds_per_pair}")
        print(f"   Equilibrium threshold: {self.equilibrium_threshold}")
        print(f"   Max workers: {max_workers}")
        print(f"{'='*80}\n")

        os.makedirs(results_dir, exist_ok=True)

        # Process pairs
        results = []
        
        with tqdm(total=len(pairs), desc="Pairs completed", unit="pair") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._run_paired_contest,
                        pair[0], pair[1],
                        results_dir,
                        idx, len(pairs)
                    ): pair
                    for idx, pair in enumerate(pairs)
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

        # Generate global summary
        summary_path = os.path.join(results_dir, "global_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Paired Contest Competition - Global Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total images: {len(image_paths)}\n")
            f.write(f"Total pairs: {len(pairs)}\n")
            f.write(f"Total images generated: {self._total_num_images_generated}\n")
            f.write(f"Estimated cost: ${self._total_num_images_generated * self._cost_per_image_generated:.2f}\n\n")
            
            equilibrium_count = sum(1 for r in results if r["equilibrium"])
            f.write(f"Pairs reaching equilibrium: {equilibrium_count}/{len(pairs)}\n")
            f.write(f"Average rounds per pair: {sum(r['rounds'] for r in results)/len(results):.1f}\n\n")
            
            f.write(f"Individual Pair Results:\n")
            f.write(f"{'-'*80}\n")
            for r in results:
                f.write(f"\n{r['pair_name']}:\n")
                f.write(f"  Rounds: {r['rounds']}\n")
                f.write(f"  Equilibrium: {r['equilibrium']}\n")
                f.write(f"  History: ")
                for h in r["history"]:
                    f.write(f"R{h['round']}:{h['winner']}({h['score']:.0%}) ")
                f.write(f"\n")

        run_duration = time.time() - run_start
        print(f"\n{'='*80}")
        print(f"✅ Paired Contest Competition Complete!")
        print(f"⏱️  TOTAL RUNTIME: {run_duration:.2f}s ({run_duration/60:.2f}m)")
        print(f"   Total pairs: {len(pairs)}")
        print(f"   Pairs with equilibrium: {equilibrium_count}/{len(pairs)}")
        print(f"   Average rounds: {sum(r['rounds'] for r in results)/len(results):.1f}")
        print(f"   Total images generated: {self._total_num_images_generated}")
        print(f"   Estimated cost: ${self._total_num_images_generated * self._cost_per_image_generated:.2f}")
        print(f"{'='*80}\n")

        return results
