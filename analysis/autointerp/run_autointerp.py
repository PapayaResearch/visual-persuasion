import os
import glob
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import litellm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import AgglomerativeClustering
from pydantic import BaseModel
from tqdm import tqdm
from typing import Optional, Union


class Theme(BaseModel):
    name: str
    description: Optional[str] = None


class Summary(BaseModel):
    themes: list[Theme]


TASK_PROMPT = """Summarize a set of visual change descriptions into a concise set of recurring themes (50 words or fewer in total).

Input: Each description characterizes what changed between an original image and an edited version.

Instructions:
1. Identify concrete, specific visual patterns that appear across multiple descriptions
2. Group related changes into distinct thematic categories
3. Name themes precisely using observable visual properties (e.g. "addition of formal attire" not "formalization"; "warm color grading" not "mood shift")
4. Preserve the specificity level of the inputs. If inputs are concrete, themes should be concrete; if inputs are already thematic, themes may be slightly broader categories
5. Use brief noun phrases; no speculation or interpretation beyond what's stated

Output format: A clean list of distinct themes, each with a brief clarifying phrase if needed. Use exactly the following JSON format:
%s

---

## Examples

Input descriptions:
- "A red hat was added to the person's head"
- "The person is now wearing a red scarf"
- "Background changed from indoor office to outdoor park"
- "The setting shifted from a living room to a garden"
- "A red bow was added to the gift box"

Output:
- Addition of red accessories (hat, scarf, bow)
- Indoor-to-outdoor setting changes (office→park, living room→garden)

---

Input descriptions:
- "Lighting shifted to golden hour tones"
- "Warm orange color cast applied"
- "Subject's expression changed from neutral to smiling"
- "Sunset lighting added"
- "Person now appears happy rather than serious"

Output:
- Warm/golden lighting adjustments (golden hour, orange cast, sunset tones)
- Positive expression changes (neutral→smiling, serious→happy)

---

Input descriptions (already thematic):
- "Addition of winter clothing items"
- "Addition of cold-weather accessories"
- "Snow added to outdoor scenes"
- "Bare trees replaced with snow-covered trees"

Output:
- Winter/cold-weather modifications (clothing, accessories, snow, trees)""" % json.dumps(Summary.model_json_schema(), indent=4)


def main():
    args = parse_args()
    if args.linkage == "ward" and args.metric != "euclidean":
        raise ValueError("Ward linkage requires Euclidean metric.")

    assert litellm.supports_response_schema(model=args.summarizer_model), f"Model {args.summarizer_model} deosn't support response schemas."


    df = load_results(args.results_dir, args.infile_pattern, args.include_config)
    df = derive_strategy_column(df, args.strategy_col, args.results_dir)

    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column: {args.text_col}")

    groupby_cols = [c.strip() for c in args.groupby_cols.split(",") if c.strip()]
    all_group_cols = [args.strategy_col] + groupby_cols
    for col in all_group_cols:
        if col not in df.columns:
            raise ValueError(f"Missing grouping column: {col}")

    out_dir = os.path.join(args.out_dir, datetime.now().strftime("run_%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    run_config = {
        "approach": "hierarchical automated interpretability",
        "created_at": datetime.now().isoformat(),
        "results_dir": args.results_dir,
        "infile_pattern": args.infile_pattern,
        "include_config": args.include_config,
        "text_col": args.text_col,
        "embedding_col": args.embedding_col,
        "embedding_model": args.embedding_model,
        "summarizer_model": args.summarizer_model,
        "summarizer_temperature": args.summarizer_temperature,
        "summarizer_max_tokens": args.summarizer_max_tokens,
        "groupby_cols": groupby_cols,
        "strategy_col": args.strategy_col,
        "matryoshka": args.matryoshka,
        "summarize_singletons": args.summarize_singletons,
        "linkage": args.linkage,
        "metric": args.metric,
        "max_workers": args.max_workers,
        "store_inputs": args.store_inputs,
    }
    with open(os.path.join(out_dir, "run_config.yaml"), "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)

    cluster_rows = []
    item_rows = []

    group_iter = tqdm(df.groupby(all_group_cols), desc="Groups")
    for group_key, group_df in group_iter:
        group_df = group_df.reset_index(drop=False)
        text_list = group_df[args.text_col].astype(str).tolist()

        if args.embedding_col:
            if args.embedding_col not in group_df.columns:
                raise ValueError(f"Embedding column not found: {args.embedding_col}")
            embeddings = []
            for cell in group_df[args.embedding_col]:
                parsed = parse_embedding_cell(cell)
                if parsed is None:
                    raise ValueError("Failed to parse embedding cell; ensure JSON list format.")
                embeddings.append(parsed)
            embeddings = np.vstack(embeddings)
        else:
            embeddings = get_embeddings_parallel(text_list, args.embedding_model, args.max_workers)

        if args.save_embeddings:
            key_name = "_".join(map(str, group_key)) if isinstance(group_key, tuple) else str(group_key)
            emb_path = os.path.join(out_dir, f"embeddings_{key_name}.npy")
            np.save(emb_path, embeddings)

        labels_by_level = hierarchical_labels(embeddings, args.linkage, args.metric)
        level_indices = sorted(labels_by_level.keys())
        previous_level_summaries: Optional[list[Summary]] = None

        per_item_levels: dict[int, list[Summary]] = {}

        for level in level_indices:
            labels = labels_by_level[level]
            cluster_to_members: dict[int, list[int]] = {}
            for idx, label in enumerate(labels):
                cluster_to_members.setdefault(int(label), []).append(idx)

            if args.matryoshka and previous_level_summaries is not None:
                source_texts = previous_level_summaries
                source_name = f"themes_level{level - 1}"
            else:
                source_texts = text_list
                source_name = args.text_col

            cluster_texts = {
                cluster_id: [source_texts[i] for i in members]
                for cluster_id, members in cluster_to_members.items()
            }

            cluster_summaries = summarize_clusters_parallel(
                cluster_texts,
                args.summarizer_model,
                args.summarizer_prompt,
                args.summarizer_temperature,
                args.summarizer_max_tokens,
                args.max_workers,
                args.summarize_singletons,
            )

            level_summaries = [None] * len(text_list)
            for cluster_id, members in cluster_to_members.items():
                summary = cluster_summaries[cluster_id]
                for idx in members:
                    level_summaries[idx] = summary

                row = {
                    "level": level,
                    "cluster_id": cluster_id,
                    "cluster_size": len(members),
                    "source_texts_used": source_name,
                    "summary": summary_to_json(summary),
                    "summary_latex": summary_to_latex(summary),
                    "member_row_indices": json.dumps([int(group_df.loc[i, "index"]) for i in members]),
                }
                if "base_id" in group_df.columns:
                    row["member_base_ids"] = json.dumps([str(group_df.loc[i, "base_id"]) for i in members])
                if args.store_inputs:
                    row["member_texts"] = json.dumps(cluster_texts[cluster_id])
                if isinstance(group_key, tuple):
                    for col, value in zip(all_group_cols, group_key):
                        row[col] = value
                else:
                    row[all_group_cols[0]] = group_key
                cluster_rows.append(row)

            per_item_levels[level] = level_summaries
            previous_level_summaries = level_summaries

        for i in range(len(group_df)):
            item_row = {
                "row_index": int(group_df.loc[i, "index"]),
                args.text_col: text_list[i],
            }
            if "base_id" in group_df.columns:
                item_row["base_id"] = str(group_df.loc[i, "base_id"])
            if isinstance(group_key, tuple):
                for col, value in zip(all_group_cols, group_key):
                    item_row[col] = value
            else:
                item_row[all_group_cols[0]] = group_key
            for level in level_indices:
                item_row[f"themes_level{level}"] = summary_to_json(per_item_levels[level][i])
                item_row[f"themes_level{level}_latex"] = summary_to_latex(per_item_levels[level][i])
            item_rows.append(item_row)

    clusters_df = pd.DataFrame(cluster_rows)
    items_df = pd.DataFrame(item_rows)
    clusters_df.to_csv(os.path.join(out_dir, "cluster_summaries.csv"), index=False)
    items_df.to_csv(os.path.join(out_dir, "item_summaries.csv"), index=False)


def load_config_defaults(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping of argument names to values.")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config with argument defaults")
    parser.add_argument("--results_dir", type=str, default="data/results-autointerp", help="Path to the folder containing results")
    parser.add_argument("--infile_pattern", type=str, default="**/results_autointerp.csv", help="Pattern to match result files")
    parser.add_argument("--include_config", action="store_true", help="Whether to include configuration data from YAML files")
    parser.add_argument("--text_col", type=str, default="differences", help="Column containing unit-level differences")
    parser.add_argument("--embedding_col", type=str, default=None, help="Optional column containing precomputed embeddings")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small", help="Embedding model for LiteLLM")
    parser.add_argument("--summarizer_model", type=str, default="gemini/gemini-3-flash-preview", help="Summarizer model for LiteLLM")
    parser.add_argument("--summarizer_temperature", type=float, default=1, help="Temperature for summarizer")
    parser.add_argument("--summarizer_max_tokens", type=int, default=400, help="Max tokens for summarizer")
    parser.add_argument("--summarizer_prompt", type=str, default=TASK_PROMPT, help="System prompt for summarizer")
    parser.add_argument("--groupby_cols", type=str, default="task.name", help="Comma-separated grouping columns (in addition to strategy)")
    parser.add_argument("--strategy_col", type=str, default="evaluate.strategy_name", help="Column used to split strategies")
    parser.add_argument("--matryoshka", action="store_true", help="Use summaries from previous level instead of original differences")
    parser.add_argument("--summarize_singletons", action="store_true", help="Summarize single-item clusters with the LLM")
    parser.add_argument("--max_workers", type=int, default=32, help="Max workers for embedding/summarization calls")
    parser.add_argument("--linkage", type=str, default="ward", choices=["ward", "complete", "average", "single"], help="Linkage for agglomerative clustering")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance metric for clustering")
    parser.add_argument("--out_dir", type=str, default="data/hierarchical_autointerp", help="Output directory for results")
    parser.add_argument("--save_embeddings", action="store_true", help="Save computed embeddings to output directory")
    parser.add_argument("--store_inputs", action="store_true", help="Store the input texts used for each cluster summary")

    pre_args, _ = parser.parse_known_args()
    cfg = load_config_defaults(pre_args.config)
    if cfg:
        parser.set_defaults(**cfg)
    return parser.parse_args()


def load_results(results_dir: str, infile_pattern: str, include_config: bool) -> pd.DataFrame:
    dfs = []
    for filename in glob.glob(os.path.join(results_dir, infile_pattern), recursive=True):
        df = pd.read_csv(filename)
        df["source_file"] = filename
        df["model"] = filename.split("/")[-2]

        if include_config:
            cfg_path = os.path.join(os.path.dirname(filename), "../../../config.yaml")
            with open(cfg_path) as yaml_file:
                cfg = yaml.safe_load(yaml_file) or {}
            cfg_flat = pd.json_normalize(cfg)
            for col in cfg_flat.columns:
                value = cfg_flat.iloc[0][col]
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                df[col] = value

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No files matched {infile_pattern} under {results_dir}")
    return pd.concat(dfs, ignore_index=True)


def derive_strategy_column(df: pd.DataFrame, strategy_col: str, results_dir: str) -> pd.DataFrame:
    if strategy_col in df.columns:
        return df
    if "strategy.name" in df.columns:
        df[strategy_col] = df["strategy.name"]
        return df
    def _from_path(path: str) -> str:
        try:
            rel = os.path.relpath(path, results_dir)
            return rel.split(os.sep)[0]
        except Exception:
            return "unknown"
    df[strategy_col] = df["source_file"].apply(_from_path)
    return df


def parse_embedding_cell(cell) -> Optional[np.ndarray]:
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=float)
    if isinstance(cell, str):
        try:
            parsed = json.loads(cell)
            return np.asarray(parsed, dtype=float)
        except Exception:
            return None
    return None


def get_embedding(text: str, model: str) -> np.ndarray:
    embedding = litellm.embedding(model=model, input=[text])
    return np.asarray(embedding["data"][0]["embedding"], dtype=float)


def get_embeddings_parallel(texts: list[str], model: str, max_workers: int) -> np.ndarray:
    embeddings = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(get_embedding, text, model): i for i, text in enumerate(texts)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Embeddings"):
            idx = futures[future]
            embeddings[idx] = future.result()
    return np.vstack(embeddings)


def summarize_texts(
    texts: list[Union[str, Summary]],
    model: str,
    summarizer_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Summary:
    content_lines = []
    for item in texts:
        if isinstance(item, Summary):
            content_lines.append(f"- {summary_to_prompt_text(item)}")
        else:
            content_lines.append(f"- {item}")
    content = "\n".join(content_lines)
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "user", "content": summarizer_prompt},
            {"role": "user", "content": f"Summarize the following items:\n{content}"},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=Summary
    )
    response = response["choices"][0]["message"]["content"].strip()
    summary_obj = Summary.model_validate_json(response)
    return summary_obj


def summarize_clusters_parallel(
    cluster_texts: dict[int, list[Union[str, Summary]]],
    model: str,
    summarizer_prompt: str,
    temperature: float,
    max_tokens: int,
    max_workers: int,
    summarize_singletons: bool,
) -> dict[int, Summary]:
    results: dict[int, Summary] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for cluster_id, texts in cluster_texts.items():
            if len(texts) == 1 and not summarize_singletons:
                results[cluster_id] = summary_from_singleton(texts[0])
                continue
            futures[ex.submit(summarize_texts, texts, model, summarizer_prompt, temperature, max_tokens)] = cluster_id
        for future in tqdm(as_completed(futures), total=len(futures), desc="Summaries"):
            cluster_id = futures[future]
            results[cluster_id] = future.result()
    return results


def summary_from_singleton(text: Union[str, Summary]) -> Summary:
    if isinstance(text, Summary):
        return text
    return Summary(themes=[Theme(name=str(text))])


def summary_to_prompt_text(summary: Summary) -> str:
    parts = []
    for theme in summary.themes:
        if theme.description:
            parts.append(f"{theme.name} ({theme.description})")
        else:
            parts.append(theme.name)
    return "; ".join(parts)


def summary_to_json(summary: Summary) -> str:
    return json.dumps(summary.model_dump(), ensure_ascii=True)


def summary_to_latex(summary: Summary) -> str:
    lines = []
    for theme in summary.themes:
        if theme.description:
            lines.append(f"\\textbf{{{theme.name}}} — \\textit{{{theme.description}}}")
        else:
            lines.append(f"\\textbf{{{theme.name}}}")
    return "\n".join(lines)


def target_cluster_counts(n_items: int) -> list[int]:
    counts = []
    level = 1
    while True:
        count = max(1, int(np.ceil(n_items / (2 ** level))))
        if not counts or count < counts[-1]:
            counts.append(count)
        if count == 1:
            break
        level += 1
    return counts


def hierarchical_labels(
    embeddings: np.ndarray,
    linkage: str,
    metric: str,
) -> dict[int, np.ndarray]:
    n_items = embeddings.shape[0]
    if n_items == 1:
        return {1: np.zeros(1, dtype=int)}

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0,
        linkage=linkage,
        metric=metric,
        compute_full_tree=True,
    )
    model.fit(embeddings)
    children = model.children_
    desired_counts = target_cluster_counts(n_items)
    remaining_targets = set(desired_counts)
    labels_by_level: dict[int, np.ndarray] = {}

    parent = list(range(n_items))
    rank = [0] * n_items
    cluster_rep = {i: i for i in range(n_items)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if rank[ra] < rank[rb]:
            parent[ra] = rb
            return rb
        if rank[ra] > rank[rb]:
            parent[rb] = ra
            return ra
        parent[rb] = ra
        rank[ra] += 1
        return ra

    cluster_count = n_items
    for merge_idx, (a, b) in enumerate(children):
        rep_a = cluster_rep[int(a)]
        rep_b = cluster_rep[int(b)]
        root = union(rep_a, rep_b)
        new_cluster_id = n_items + merge_idx
        cluster_rep[new_cluster_id] = root
        cluster_count -= 1

        if cluster_count in remaining_targets:
            roots = [find(i) for i in range(n_items)]
            unique_roots = {}
            labels = np.zeros(n_items, dtype=int)
            next_label = 0
            for i, r in enumerate(roots):
                if r not in unique_roots:
                    unique_roots[r] = next_label
                    next_label += 1
                labels[i] = unique_roots[r]
            level = desired_counts.index(cluster_count) + 1
            labels_by_level[level] = labels
            remaining_targets.remove(cluster_count)
            if not remaining_targets:
                break
    return labels_by_level


if __name__ == "__main__":
    main()
