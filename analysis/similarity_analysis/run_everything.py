import os
import argparse
import numpy as np
import pandas as pd
import torch
import clip
import lpips
import scienceplots
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.metrics import structural_similarity


plt.style.use(["science", "no-latex"])

if __package__:
    from . import sim_utils
    from .s3_image_store import S3ImageStore, S3PathContext
else:  # pragma: no cover
    import sim_utils
    from s3_image_store import S3ImageStore, S3PathContext


def _parse_s3_dir(s3_uri: str) -> tuple[str, str]:
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)
    return bucket, prefix


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _compute_clip_embeddings(
    *,
    image_paths: list[str],
    store: S3ImageStore,
    model,
    preprocess,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    desc: str,
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    model.eval()
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        kept_paths = []
        for path in batch_paths:
            img = store.get_pil_rgb(path)
            if img is None:
                continue
            images.append(preprocess(img))
            kept_paths.append(path)
        if not images:
            continue
        image_batch = torch.stack(images, dim=0).to(device=device, dtype=dtype)
        with torch.no_grad():
            batch_emb = model.encode_image(image_batch).cpu().numpy()
        for path, emb in zip(kept_paths, batch_emb):
            embeddings[path] = emb
    return embeddings


def _compute_clip_embeddings_matted(
    *,
    image_paths: list[str],
    store: S3ImageStore,
    model,
    preprocess,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    matting_model: str,
    background: str,
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    model.eval()
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        kept_paths = []
        for path in batch_paths:
            img = store.get_matted_pil_rgb(path, matting_model=matting_model, background=background)
            if img is None:
                continue
            images.append(preprocess(img))
            kept_paths.append(path)
        if not images:
            continue
        image_batch = torch.stack(images, dim=0).to(device=device, dtype=dtype)
        with torch.no_grad():
            batch_emb = model.encode_image(image_batch).cpu().numpy()
        for path, emb in zip(kept_paths, batch_emb):
            embeddings[path] = emb
    return embeddings


def run_ssim(
    *,
    df: pd.DataFrame,
    df_pairs: pd.DataFrame,
    store: S3ImageStore,
    out_dir: str,
    ssim_size: int | None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ssim_cache_path = os.path.join(out_dir, "ssim_cache.npy")
    ssim_cache: dict = {}
    if os.path.exists(ssim_cache_path):
        ssim_cache = np.load(ssim_cache_path, allow_pickle=True).item()
    print("[ssim] cache entries:", len(ssim_cache))

    between_results = []
    between_tasks = []
    for pair_id, group in df_pairs.groupby("pair_id", sort=False):
        if len(group) < 2:
            continue
        group = group.sort_values("image_path")
        path_a = group.iloc[0]["image_path"]
        path_b = group.iloc[1]["image_path"]
        cache_key = ("between", path_a, path_b, ssim_size)
        meta = {"pair_id": pair_id, "debiased_iter": group.iloc[0]["debiased_iter"], "path_a": path_a, "path_b": path_b}
        if cache_key in ssim_cache:
            between_results.append({"similarity": ssim_cache[cache_key], "meta": meta})
        else:
            meta["cache_key"] = cache_key
            between_tasks.append({"path_a": path_a, "path_b": path_b, "meta": meta})

    def _ssim_for_task(task: dict) -> dict:
        if ssim_size is not None:
            img_a = store.get_numpy_rgb_uint8(task["path_a"], ssim_size)
            img_b = store.get_numpy_rgb_uint8(task["path_b"], ssim_size)
        else:
            img_a_pil = store.get_pil_rgb(task["path_a"])
            img_b_pil = store.get_pil_rgb(task["path_b"])
            if img_a_pil is None or img_b_pil is None:
                return {"similarity": float("nan"), "meta": task["meta"]}
            if img_a_pil.size != img_b_pil.size:
                img_b_pil = img_b_pil.resize(img_a_pil.size)
            img_a = np.asarray(img_a_pil, dtype=np.uint8)
            img_b = np.asarray(img_b_pil, dtype=np.uint8)
        if img_a is None or img_b is None:
            return {"similarity": float("nan"), "meta": task["meta"]}
        score = structural_similarity(img_a, img_b, channel_axis=-1, data_range=255)
        return {"similarity": float(score), "meta": task["meta"]}

    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_task = {executor.submit(_ssim_for_task, task): task for task in between_tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as e:
                meta = task.get("meta", {})
                raise RuntimeError(
                    f"SSIM failed for between_pairs: pair_id={meta.get('pair_id')!r} "
                    f"path_a={task.get('path_a')!r} path_b={task.get('path_b')!r}"
                ) from e
            between_results.append(result)
            cache_key = result["meta"].get("cache_key")
            if cache_key is not None:
                ssim_cache[cache_key] = result["similarity"]
    print("[ssim] between_pairs cached:", len(between_results) - len(between_tasks), "computed:", len(between_tasks))

    df_between = pd.DataFrame(
        [
            {
                "pair_id": r["meta"]["pair_id"],
                "debiased_iter": r["meta"]["debiased_iter"],
                "similarity": r["similarity"],
            }
            for r in between_results
            if not np.isnan(r["similarity"])
        ]
    )
    sim_by_k_stats_between_pairs = (
        df_between.groupby("debiased_iter")[["similarity"]]
        .agg(["mean", "std", "count"])
        .apply(lambda row: sim_utils.make_95_confidence_interval(row, metric="similarity"), axis=1)
    )
    print(sim_by_k_stats_between_pairs)

    orig_images = df[df.pair_id.isna()].set_index("current_base_id")["image_path"].to_dict()
    df_pairs = df_pairs.copy()
    df_pairs["is_final"] = df_pairs.current_id.str.endswith("_final")

    original_tasks = []
    original_results = []
    for (base_id, debiased_iter, is_final), group in df_pairs.groupby(["current_base_id", "debiased_iter", "is_final"], sort=False):
        orig_path = orig_images.get(base_id)
        if not orig_path:
            continue
        path_a = group.iloc[0]["image_path"]
        path_b = orig_path
        cache_key = ("original", path_a, path_b, ssim_size)
        meta = {
            "current_base_id": base_id,
            "debiased_iter": debiased_iter,
            "is_final": is_final,
            "path_a": path_a,
            "path_b": path_b,
        }
        if cache_key in ssim_cache:
            original_results.append({"similarity": ssim_cache[cache_key], "meta": meta})
        else:
            meta["cache_key"] = cache_key
            original_tasks.append({"path_a": path_a, "path_b": path_b, "meta": meta})

    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_task = {executor.submit(_ssim_for_task, task): task for task in original_tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as e:
                meta = task.get("meta", {})
                raise RuntimeError(
                    f"SSIM failed for from_original: current_base_id={meta.get('current_base_id')!r} "
                    f"debiased_iter={meta.get('debiased_iter')!r} is_final={meta.get('is_final')!r} "
                    f"path_a={task.get('path_a')!r} path_b={task.get('path_b')!r}"
                ) from e
            original_results.append(result)
            cache_key = result["meta"].get("cache_key")
            if cache_key is not None:
                ssim_cache[cache_key] = result["similarity"]
    print("[ssim] from_original cached:", len(original_results) - len(original_tasks), "computed:", len(original_tasks))

    df_original = pd.DataFrame(
        [
            {
                "current_base_id": r["meta"]["current_base_id"],
                "debiased_iter": r["meta"]["debiased_iter"],
                "is_final": r["meta"]["is_final"],
                "similarity": r["similarity"],
            }
            for r in original_results
            if not np.isnan(r["similarity"])
        ]
    )
    sim_by_k_stats_from_originals = (
        df_original.groupby(["is_final", "debiased_iter"])[["similarity"]]
        .agg(["mean", "std", "count"])
        .apply(lambda row: sim_utils.make_95_confidence_interval(row, metric="similarity"), axis=1)
    )
    print(sim_by_k_stats_from_originals)

    sim_utils.plot_joint_metric(
        plt=plt,
        stats_between_pairs=sim_by_k_stats_between_pairs,
        stats_from_originals=sim_by_k_stats_from_originals,
        out_path=os.path.join(out_dir, "ssim_similarity_joint_plot.pdf"),
        y_label="SSIM (mean ± 95% CI)",
    )

    between_long = pd.DataFrame(
        [
            {
                "metric": "ssim",
                "comparison": "between_pairs",
                "pair_id": r["meta"]["pair_id"],
                "debiased_iter": r["meta"]["debiased_iter"],
                "similarity": r["similarity"],
                "path_a": r["meta"].get("path_a"),
                "path_b": r["meta"].get("path_b"),
                "ssim_size": ssim_size,
            }
            for r in between_results
            if not np.isnan(r["similarity"])
        ]
    )
    from_orig_long = pd.DataFrame(
        [
            {
                "metric": "ssim",
                "comparison": "from_original",
                "current_base_id": r["meta"]["current_base_id"],
                "debiased_iter": r["meta"]["debiased_iter"],
                "is_final": bool(r["meta"]["is_final"]),
                "similarity": r["similarity"],
                "path_a": r["meta"].get("path_a"),
                "path_b": r["meta"].get("path_b"),
                "ssim_size": ssim_size,
            }
            for r in original_results
            if not np.isnan(r["similarity"])
        ]
    )
    sim_utils.write_long_csv(pd.concat([between_long, from_orig_long], axis=0, ignore_index=True), os.path.join(out_dir, "ssim_similarity_long.csv"))
    print("[ssim] long rows:", len(between_long) + len(from_orig_long))

    if ssim_cache:
        np.save(ssim_cache_path, ssim_cache, allow_pickle=True)
        print("[ssim] cache entries (saved):", len(ssim_cache))


def run_lpips(
    *,
    df: pd.DataFrame,
    df_pairs: pd.DataFrame,
    store: S3ImageStore,
    out_dir: str,
    lpips_net: str,
    lpips_size: int | None,
    lpips_batch_size: int,
    amp: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp and device.type == "cuda"
    if lpips_size is not None and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    loss_fn = lpips.LPIPS(net=lpips_net).to(device=device)
    loss_fn.eval()
    if device.type == "cuda":
        loss_fn = loss_fn.to(memory_format=torch.channels_last)

    cache_path = os.path.join(out_dir, "lpips_cache.npy")
    cache: dict = {}
    if os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True).item()
    print("[lpips] cache entries:", len(cache))

    df_pairs = df_pairs.copy()
    df_pairs["is_final"] = df_pairs.current_id.str.endswith("_final")

    between_tasks = []
    between_results = []
    for pair_id, group in df_pairs.groupby("pair_id", sort=False):
        if len(group) < 2:
            continue
        group = group.sort_values("image_path")
        path_a = group.iloc[0]["image_path"]
        path_b = group.iloc[1]["image_path"]
        cache_key = ("between", path_a, path_b, lpips_net, lpips_size)
        meta = {"pair_id": pair_id, "debiased_iter": group.iloc[0]["debiased_iter"], "path_a": path_a, "path_b": path_b}
        if cache_key in cache:
            between_results.append({"distance": cache[cache_key], "meta": meta})
        else:
            meta["cache_key"] = cache_key
            between_tasks.append({"path_a": path_a, "path_b": path_b, "meta": meta})

    def _lpips_batch(batch: list[dict]) -> list[float]:
        tensors_a = []
        tensors_b = []
        valid_idx = []
        for idx, task in enumerate(batch):
            ta = store.get_lpips_tensor(task["path_a"], lpips_size)
            tb = store.get_lpips_tensor(task["path_b"], lpips_size)
            tensors_a.append(ta)
            tensors_b.append(tb)
            if ta is not None and tb is not None:
                valid_idx.append(idx)
        out = [float("nan")] * len(batch)
        if not valid_idx:
            return out
        batch_a = torch.stack([tensors_a[i] for i in valid_idx], dim=0).contiguous(memory_format=torch.channels_last)
        batch_b = torch.stack([tensors_b[i] for i in valid_idx], dim=0).contiguous(memory_format=torch.channels_last)
        batch_a = batch_a.to(device=device)
        batch_b = batch_b.to(device=device)
        with torch.inference_mode():
            if use_amp:
                with torch.cuda.amp.autocast():
                    dist = loss_fn(batch_a, batch_b).flatten().detach().cpu().numpy()
            else:
                dist = loss_fn(batch_a, batch_b).flatten().detach().cpu().numpy()
        for i, d in zip(valid_idx, dist):
            out[i] = float(d)
        return out

    for i in range(0, len(between_tasks), lpips_batch_size):
        batch = between_tasks[i : i + lpips_batch_size]
        dists = _lpips_batch(batch)
        for task, dist in zip(batch, dists):
            between_results.append({"distance": dist, "meta": task["meta"]})
            cache_key = task["meta"].get("cache_key")
            if cache_key is not None:
                cache[cache_key] = dist
    print("[lpips] between_pairs cached:", len(between_results) - len(between_tasks), "computed:", len(between_tasks))

    df_between = pd.DataFrame(
        [
            {"pair_id": r["meta"]["pair_id"], "debiased_iter": r["meta"]["debiased_iter"], "distance": r["distance"]}
            for r in between_results
            if not np.isnan(r["distance"])
        ]
    )
    dist_by_k_stats_between_pairs = (
        df_between.groupby("debiased_iter")[["distance"]]
        .agg(["mean", "std", "count"])
        .apply(lambda row: sim_utils.make_95_confidence_interval(row, metric="distance"), axis=1)
    )
    print(dist_by_k_stats_between_pairs)

    orig_images = df[df.pair_id.isna()].set_index("current_base_id")["image_path"].to_dict()
    original_tasks = []
    original_results = []
    for (base_id, debiased_iter, is_final), group in df_pairs.groupby(["current_base_id", "debiased_iter", "is_final"], sort=False):
        orig_path = orig_images.get(base_id)
        if not orig_path:
            continue
        path_a = group.iloc[0]["image_path"]
        path_b = orig_path
        cache_key = ("original", path_a, path_b, lpips_net, lpips_size)
        meta = {
            "current_base_id": base_id,
            "debiased_iter": debiased_iter,
            "is_final": is_final,
            "path_a": path_a,
            "path_b": path_b,
        }
        if cache_key in cache:
            original_results.append({"distance": cache[cache_key], "meta": meta})
        else:
            meta["cache_key"] = cache_key
            original_tasks.append({"path_a": path_a, "path_b": path_b, "meta": meta})

    for i in range(0, len(original_tasks), lpips_batch_size):
        batch = original_tasks[i : i + lpips_batch_size]
        dists = _lpips_batch(batch)
        for task, dist in zip(batch, dists):
            original_results.append({"distance": dist, "meta": task["meta"]})
            cache_key = task["meta"].get("cache_key")
            if cache_key is not None:
                cache[cache_key] = dist
    print("[lpips] from_original cached:", len(original_results) - len(original_tasks), "computed:", len(original_tasks))

    df_original = pd.DataFrame(
        [
            {
                "current_base_id": r["meta"]["current_base_id"],
                "debiased_iter": r["meta"]["debiased_iter"],
                "is_final": r["meta"]["is_final"],
                "distance": r["distance"],
            }
            for r in original_results
            if not np.isnan(r["distance"])
        ]
    )
    dist_by_k_stats_from_originals = (
        df_original.groupby(["is_final", "debiased_iter"])[["distance"]]
        .agg(["mean", "std", "count"])
        .apply(lambda row: sim_utils.make_95_confidence_interval(row, metric="distance"), axis=1)
    )
    print(dist_by_k_stats_from_originals)

    sim_utils.plot_joint_metric(
        plt=plt,
        stats_between_pairs=dist_by_k_stats_between_pairs,
        stats_from_originals=dist_by_k_stats_from_originals,
        out_path=os.path.join(out_dir, "lpips_distance_joint_plot.pdf"),
        y_label="LPIPS distance (mean ± 95% CI)",
    )

    between_long = pd.DataFrame(
        [
            {
                "metric": f"lpips_{lpips_net}",
                "comparison": "between_pairs",
                "pair_id": r["meta"]["pair_id"],
                "debiased_iter": r["meta"]["debiased_iter"],
                "distance": r["distance"],
                "path_a": r["meta"].get("path_a"),
                "path_b": r["meta"].get("path_b"),
                "lpips_net": lpips_net,
                "lpips_size": lpips_size,
            }
            for r in between_results
            if not np.isnan(r["distance"])
        ]
    )
    from_orig_long = pd.DataFrame(
        [
            {
                "metric": f"lpips_{lpips_net}",
                "comparison": "from_original",
                "current_base_id": r["meta"]["current_base_id"],
                "debiased_iter": r["meta"]["debiased_iter"],
                "is_final": bool(r["meta"]["is_final"]),
                "distance": r["distance"],
                "path_a": r["meta"].get("path_a"),
                "path_b": r["meta"].get("path_b"),
                "lpips_net": lpips_net,
                "lpips_size": lpips_size,
            }
            for r in original_results
            if not np.isnan(r["distance"])
        ]
    )
    sim_utils.write_long_csv(pd.concat([between_long, from_orig_long], axis=0, ignore_index=True), os.path.join(out_dir, "lpips_distance_long.csv"))
    print("[lpips] long rows:", len(between_long) + len(from_orig_long))

    if cache:
        np.save(cache_path, cache, allow_pickle=True)
        print("[lpips] cache entries (saved):", len(cache))


def run_clip_embedding_similarity(
    *,
    df: pd.DataFrame,
    df_pairs: pd.DataFrame,
    store: S3ImageStore,
    out_dir: str,
    embedding_model: str,
    batch_size: int,
    resume: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model, preprocess = clip.load(embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    embeddings_path = os.path.join(out_dir, "image_embeddings.npy")
    embeddings_map: dict[str, np.ndarray] = {}
    if resume and os.path.exists(embeddings_path):
        embeddings_map = np.load(embeddings_path, allow_pickle=True).item()
    print("[embed] embeddings cached:", len(embeddings_map))

    image_paths = df["image_path"].astype(str).tolist()
    missing = [p for p in image_paths if p not in embeddings_map]
    if missing:
        print("[embed] embeddings missing:", len(missing))
        new = _compute_clip_embeddings(
            image_paths=missing,
            store=store,
            model=model,
            preprocess=preprocess,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            desc="Computing CLIP embeddings",
        )
        embeddings_map.update(new)
        np.save(embeddings_path, embeddings_map, allow_pickle=True)
    print("[embed] embeddings total:", len(embeddings_map))

    df = df.copy()
    df["embedding"] = df["image_path"].apply(lambda p: embeddings_map.get(str(p)))
    df = df[df.embedding.notnull()]
    df_pairs = df_pairs.copy()
    df_pairs["embedding"] = df_pairs["image_path"].apply(lambda p: embeddings_map.get(str(p)))
    missing_pair_embeddings = int(df_pairs["embedding"].isna().sum())
    if missing_pair_embeddings:
        print("[embed] pair rows missing embeddings (corrupt/unreadable):", missing_pair_embeddings)
    df_pairs = df_pairs[df_pairs.embedding.notnull()].copy()
    before_pairs = len(df_pairs)
    df_pairs = sim_utils.filter_valid_pair_rows(df_pairs, min_per_pair=2)
    print("[embed] pair rows after filter:", len(df_pairs), "dropped:", before_pairs - len(df_pairs))

    sim_between = (
        df_pairs.groupby("pair_id", sort=False)
        .apply(
            lambda group: pd.Series(
                {
                    "similarity": _cosine_similarity(
                        group.sort_values("image_path").iloc[0]["embedding"],
                        group.sort_values("image_path").iloc[1]["embedding"],
                    ),
                    "debiased_iter": group.iloc[0]["debiased_iter"],
                }
            )
        )
        .groupby("debiased_iter")
        .agg(["mean", "std", "count"])
        .apply(lambda row: sim_utils.make_95_confidence_interval(row, metric="similarity"), axis=1)
    )
    print(sim_between)

    orig_embeddings = df[df.pair_id.isna()].set_index("current_base_id")["embedding"].to_dict()
    missing_originals = int((~df_pairs["current_base_id"].isin(orig_embeddings.keys())).sum())
    if missing_originals:
        print("[embed] pair rows missing original embedding (corrupt/unreadable original):", missing_originals)
    df_pairs = df_pairs[df_pairs["current_base_id"].isin(orig_embeddings.keys())].copy()
    df_pairs["is_final"] = df_pairs.current_id.str.endswith("_final")
    sim_from = (
        df_pairs.groupby(["current_base_id", "debiased_iter", "is_final"], sort=False)
        .apply(
            lambda group: pd.Series(
                {
                    "similarity": _cosine_similarity(
                        group.iloc[0]["embedding"],
                        orig_embeddings[group.name[0]],
                    )
                }
            )
        )
        .groupby(["is_final", "debiased_iter"])
        .agg(["mean", "std", "count"])
        .apply(lambda row: sim_utils.make_95_confidence_interval(row, metric="similarity"), axis=1)
    )
    print(sim_from)

    sim_utils.plot_joint_metric(
        plt=plt,
        stats_between_pairs=sim_between,
        stats_from_originals=sim_from,
        out_path=os.path.join(out_dir, "embedding_similarity_joint_plot.pdf"),
        y_label="Cosine similarity (mean ± 95% CI)",
    )

    df_pairs_sorted = df_pairs.sort_values(["pair_id", "image_path"]).copy()
    between_long = (
        df_pairs_sorted.groupby("pair_id", sort=False)
        .apply(
            lambda group: pd.Series(
                {
                    "metric": "embedding_cosine",
                    "comparison": "between_pairs",
                    "pair_id": group.name,
                    "debiased_iter": group.iloc[0]["debiased_iter"],
                    "similarity": _cosine_similarity(group.iloc[0]["embedding"], group.iloc[1]["embedding"]),
                    "path_a": group.iloc[0]["image_path"],
                    "path_b": group.iloc[1]["image_path"],
                    "embedding_model": embedding_model,
                }
            )
        )
        .reset_index(drop=True)
    )
    orig_paths = df[df.pair_id.isna()].set_index("current_base_id")["image_path"].to_dict()
    from_orig_long = (
        df_pairs.groupby(["current_base_id", "debiased_iter", "is_final"], sort=False)
        .apply(
            lambda group: pd.Series(
                {
                    "metric": "embedding_cosine",
                    "comparison": "from_original",
                    "current_base_id": group.name[0],
                    "debiased_iter": group.name[1],
                    "is_final": bool(group.name[2]),
                    "similarity": _cosine_similarity(group.iloc[0]["embedding"], orig_embeddings[group.name[0]]),
                    "path_a": group.iloc[0]["image_path"],
                    "path_b": orig_paths.get(group.name[0]),
                    "embedding_model": embedding_model,
                }
            )
        )
        .reset_index(drop=True)
    )
    sim_utils.write_long_csv(pd.concat([between_long, from_orig_long], axis=0, ignore_index=True), os.path.join(out_dir, "embedding_similarity_long.csv"))
    print("[embed] long rows:", len(between_long) + len(from_orig_long))


def run_matted_ssim(
    *,
    df: pd.DataFrame,
    df_pairs: pd.DataFrame,
    store: S3ImageStore,
    out_dir: str,
    ssim_size: int | None,
    matting_model: str,
    background: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "matted_ssim_cache.npy")
    cache: dict = {}
    if os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True).item()
    print("[matted_ssim] cache entries:", len(cache))

    between_results = []
    between_tasks = []
    for pair_id, group in df_pairs.groupby("pair_id", sort=False):
        if len(group) < 2:
            continue
        group = group.sort_values("image_path")
        path_a = group.iloc[0]["image_path"]
        path_b = group.iloc[1]["image_path"]
        cache_key = ("between", path_a, path_b, ssim_size, matting_model, background)
        meta = {"pair_id": pair_id, "debiased_iter": group.iloc[0]["debiased_iter"], "path_a": path_a, "path_b": path_b}
        if cache_key in cache:
            between_results.append({"similarity": cache[cache_key], "meta": meta})
        else:
            meta["cache_key"] = cache_key
            between_tasks.append({"path_a": path_a, "path_b": path_b, "meta": meta})

    def _task(task: dict) -> dict:
        img_a = store.get_matted_pil_rgb(task["path_a"], matting_model=matting_model, background=background)
        img_b = store.get_matted_pil_rgb(task["path_b"], matting_model=matting_model, background=background)
        if img_a is None or img_b is None:
            return {"similarity": float("nan"), "meta": task["meta"]}
        if ssim_size is not None:
            img_a = img_a.resize((ssim_size, ssim_size))
            img_b = img_b.resize((ssim_size, ssim_size))
        if img_a.size != img_b.size:
            img_b = img_b.resize(img_a.size)
        arr_a = np.asarray(img_a, dtype=np.uint8)
        arr_b = np.asarray(img_b, dtype=np.uint8)
        score = structural_similarity(arr_a, arr_b, channel_axis=-1, data_range=255)
        return {"similarity": float(score), "meta": task["meta"]}

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_task = {executor.submit(_task, task): task for task in between_tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as e:
                meta = task.get("meta", {})
                raise RuntimeError(
                    f"Matted SSIM failed for between_pairs: pair_id={meta.get('pair_id')!r} "
                    f"path_a={task.get('path_a')!r} path_b={task.get('path_b')!r}"
                ) from e
            between_results.append(result)
            cache_key = result["meta"].get("cache_key")
            if cache_key is not None:
                cache[cache_key] = result["similarity"]
    print("[matted_ssim] between_pairs cached:", len(between_results) - len(between_tasks), "computed:", len(between_tasks))

    df_between = pd.DataFrame(
        [
            {"pair_id": r["meta"]["pair_id"], "debiased_iter": r["meta"]["debiased_iter"], "similarity": r["similarity"]}
            for r in between_results
            if not np.isnan(r["similarity"])
        ]
    )
    sim_between = (
        df_between.groupby("debiased_iter")[["similarity"]]
        .agg(["mean", "std", "count"])
        .apply(lambda row: sim_utils.make_95_confidence_interval(row, metric="similarity"), axis=1)
    )
    print(sim_between)

    orig_images = df[df.pair_id.isna()].set_index("current_base_id")["image_path"].to_dict()
    df_pairs = df_pairs.copy()
    df_pairs["is_final"] = df_pairs.current_id.str.endswith("_final")

    original_tasks = []
    original_results = []
    for (base_id, debiased_iter, is_final), group in df_pairs.groupby(["current_base_id", "debiased_iter", "is_final"], sort=False):
        orig_path = orig_images.get(base_id)
        if not orig_path:
            continue
        path_a = group.iloc[0]["image_path"]
        path_b = orig_path
        cache_key = ("original", path_a, path_b, ssim_size, matting_model, background)
        meta = {
            "current_base_id": base_id,
            "debiased_iter": debiased_iter,
            "is_final": is_final,
            "path_a": path_a,
            "path_b": path_b,
        }
        if cache_key in cache:
            original_results.append({"similarity": cache[cache_key], "meta": meta})
        else:
            meta["cache_key"] = cache_key
            original_tasks.append({"path_a": path_a, "path_b": path_b, "meta": meta})

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_task = {executor.submit(_task, task): task for task in original_tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as e:
                meta = task.get("meta", {})
                raise RuntimeError(
                    f"Matted SSIM failed for from_original: current_base_id={meta.get('current_base_id')!r} "
                    f"debiased_iter={meta.get('debiased_iter')!r} is_final={meta.get('is_final')!r} "
                    f"path_a={task.get('path_a')!r} path_b={task.get('path_b')!r}"
                ) from e
            original_results.append(result)
            cache_key = result["meta"].get("cache_key")
            if cache_key is not None:
                cache[cache_key] = result["similarity"]
    print("[matted_ssim] from_original cached:", len(original_results) - len(original_tasks), "computed:", len(original_tasks))

    df_original = pd.DataFrame(
        [
            {
                "current_base_id": r["meta"]["current_base_id"],
                "debiased_iter": r["meta"]["debiased_iter"],
                "is_final": r["meta"]["is_final"],
                "similarity": r["similarity"],
            }
            for r in original_results
            if not np.isnan(r["similarity"])
        ]
    )
    sim_from = (
        df_original.groupby(["is_final", "debiased_iter"])[["similarity"]]
        .agg(["mean", "std", "count"])
        .apply(lambda row: sim_utils.make_95_confidence_interval(row, metric="similarity"), axis=1)
    )
    print(sim_from)

    sim_utils.plot_joint_metric(
        plt=plt,
        stats_between_pairs=sim_between,
        stats_from_originals=sim_from,
        out_path=os.path.join(out_dir, "matted_ssim_similarity_joint_plot.pdf"),
        y_label="Matted SSIM (mean ± 95% CI)",
    )

    between_long = pd.DataFrame(
        [
            {
                "metric": "matted_ssim",
                "comparison": "between_pairs",
                "pair_id": r["meta"]["pair_id"],
                "debiased_iter": r["meta"]["debiased_iter"],
                "similarity": r["similarity"],
                "path_a": r["meta"].get("path_a"),
                "path_b": r["meta"].get("path_b"),
                "matting_model": matting_model,
                "background": background,
                "ssim_size": ssim_size,
            }
            for r in between_results
            if not np.isnan(r["similarity"])
        ]
    )
    from_orig_long = pd.DataFrame(
        [
            {
                "metric": "matted_ssim",
                "comparison": "from_original",
                "current_base_id": r["meta"]["current_base_id"],
                "debiased_iter": r["meta"]["debiased_iter"],
                "is_final": bool(r["meta"]["is_final"]),
                "similarity": r["similarity"],
                "path_a": r["meta"].get("path_a"),
                "path_b": r["meta"].get("path_b"),
                "matting_model": matting_model,
                "background": background,
                "ssim_size": ssim_size,
            }
            for r in original_results
            if not np.isnan(r["similarity"])
        ]
    )
    sim_utils.write_long_csv(pd.concat([between_long, from_orig_long], axis=0, ignore_index=True), os.path.join(out_dir, "matted_ssim_similarity_long.csv"))
    print("[matted_ssim] long rows:", len(between_long) + len(from_orig_long))

    if cache:
        np.save(cache_path, cache, allow_pickle=True)
        print("[matted_ssim] cache entries (saved):", len(cache))


def run_matted_embedding_similarity(
    *,
    df: pd.DataFrame,
    df_pairs: pd.DataFrame,
    store: S3ImageStore,
    out_dir: str,
    embedding_model: str,
    batch_size: int,
    resume: bool,
    matting_model: str,
    background: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model, preprocess = clip.load(embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    embeddings_path = os.path.join(out_dir, "matted_embeddings.npy")
    embeddings_map: dict[str, np.ndarray] = {}
    if resume and os.path.exists(embeddings_path):
        embeddings_map = np.load(embeddings_path, allow_pickle=True).item()
    print("[matted_embed] embeddings cached:", len(embeddings_map))

    image_paths = df["image_path"].astype(str).unique().tolist()
    missing = [p for p in image_paths if p not in embeddings_map]
    if missing:
        print("[matted_embed] embeddings missing:", len(missing))
        new = _compute_clip_embeddings_matted(
            image_paths=missing,
            store=store,
            model=model,
            preprocess=preprocess,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            matting_model=matting_model,
            background=background,
        )
        embeddings_map.update(new)
        np.save(embeddings_path, embeddings_map, allow_pickle=True)
    print("[matted_embed] embeddings total:", len(embeddings_map))

    df = df.copy()
    df["embedding"] = df["image_path"].apply(lambda p: embeddings_map.get(str(p)))
    df = df[df.embedding.notnull()]
    df_pairs = df_pairs.copy()
    df_pairs["embedding"] = df_pairs["image_path"].apply(lambda p: embeddings_map.get(str(p)))
    missing_pair_embeddings = int(df_pairs["embedding"].isna().sum())
    if missing_pair_embeddings:
        print("[matted_embed] pair rows missing embeddings (corrupt/unreadable):", missing_pair_embeddings)
    df_pairs = df_pairs[df_pairs.embedding.notnull()]
    before_pairs = len(df_pairs)
    df_pairs = sim_utils.filter_valid_pair_rows(df_pairs, min_per_pair=2)
    print("[matted_embed] pair rows after filter:", len(df_pairs), "dropped:", before_pairs - len(df_pairs))

    sim_between = (
        df_pairs.groupby("pair_id", sort=False)
        .apply(
            lambda group: pd.Series(
                {
                    "similarity": _cosine_similarity(
                        group.sort_values("image_path").iloc[0]["embedding"],
                        group.sort_values("image_path").iloc[1]["embedding"],
                    ),
                    "debiased_iter": group.iloc[0]["debiased_iter"],
                }
            )
        )
        .groupby("debiased_iter")
        .agg(["mean", "std", "count"])
        .apply(lambda row: sim_utils.make_95_confidence_interval(row, metric="similarity"), axis=1)
    )
    print(sim_between)

    orig_embeddings = df[df.pair_id.isna()].set_index("current_base_id")["embedding"].to_dict()
    missing_originals = int((~df_pairs["current_base_id"].isin(orig_embeddings.keys())).sum())
    if missing_originals:
        print("[matted_embed] pair rows missing original embedding (corrupt/unreadable original):", missing_originals)
    df_pairs = df_pairs[df_pairs["current_base_id"].isin(orig_embeddings.keys())].copy()
    df_pairs["is_final"] = df_pairs.current_id.str.endswith("_final")
    sim_from = (
        df_pairs.groupby(["current_base_id", "debiased_iter", "is_final"], sort=False)
        .apply(
            lambda group: pd.Series(
                {
                    "similarity": _cosine_similarity(
                        group.iloc[0]["embedding"],
                        orig_embeddings[group.name[0]],
                    )
                }
            )
        )
        .groupby(["is_final", "debiased_iter"])
        .agg(["mean", "std", "count"])
        .apply(lambda row: sim_utils.make_95_confidence_interval(row, metric="similarity"), axis=1)
    )
    print(sim_from)

    sim_utils.plot_joint_metric(
        plt=plt,
        stats_between_pairs=sim_between,
        stats_from_originals=sim_from,
        out_path=os.path.join(out_dir, "matted_embedding_similarity_joint_plot.pdf"),
        y_label="Matted cosine similarity (mean ± 95% CI)",
    )

    df_pairs_sorted = df_pairs.sort_values(["pair_id", "image_path"]).copy()
    between_long = (
        df_pairs_sorted.groupby("pair_id", sort=False)
        .apply(
            lambda group: pd.Series(
                {
                    "metric": "matted_embedding_cosine",
                    "comparison": "between_pairs",
                    "pair_id": group.name,
                    "debiased_iter": group.iloc[0]["debiased_iter"],
                    "similarity": _cosine_similarity(group.iloc[0]["embedding"], group.iloc[1]["embedding"]),
                    "path_a": group.iloc[0]["image_path"],
                    "path_b": group.iloc[1]["image_path"],
                    "matting_model": matting_model,
                    "background": background,
                    "embedding_model": embedding_model,
                }
            )
        )
        .reset_index(drop=True)
    )
    orig_paths = df[df.pair_id.isna()].set_index("current_base_id")["image_path"].to_dict()
    from_orig_long = (
        df_pairs.groupby(["current_base_id", "debiased_iter", "is_final"], sort=False)
        .apply(
            lambda group: pd.Series(
                {
                    "metric": "matted_embedding_cosine",
                    "comparison": "from_original",
                    "current_base_id": group.name[0],
                    "debiased_iter": group.name[1],
                    "is_final": bool(group.name[2]),
                    "similarity": _cosine_similarity(group.iloc[0]["embedding"], orig_embeddings[group.name[0]]),
                    "path_a": group.iloc[0]["image_path"],
                    "path_b": orig_paths.get(group.name[0]),
                    "matting_model": matting_model,
                    "background": background,
                    "embedding_model": embedding_model,
                }
            )
        )
        .reset_index(drop=True)
    )
    sim_utils.write_long_csv(pd.concat([between_long, from_orig_long], axis=0, ignore_index=True), os.path.join(out_dir, "matted_embedding_similarity_long.csv"))
    print("[matted_embed] long rows:", len(between_long) + len(from_orig_long))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all similarity analyses in-process with shared S3/image caches.")
    parser.add_argument("--cache_dir", type=str, default="s3://<bucket-name>/<cache-prefix>/")
    parser.add_argument("--data_dir", type=str, default="s3://<bucket-name>/<data-prefix>/")

    parser.add_argument("--out_dir", type=str, default="similarity_analysis_out", help="Flat output directory for all caches/CSVs/plots")

    parser.add_argument("--ssim_size", type=int, default=224, help="Resize images to this square size before SSIM; set to 0 to disable")

    parser.add_argument("--lpips_net", type=str, default="alex")
    parser.add_argument("--lpips_size", type=int, default=224, help="Resize images to this square size before LPIPS; set to 0 to disable")
    parser.add_argument("--lpips_batch_size", type=int, default=32)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--embedding_model", type=str, default="ViT-B/32")
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--resume", action="store_true", help="Resume from existing caches/embeddings when present")
    parser.add_argument(
        "--cached_metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cached image_metadata.csv if present; disable with --no-cached-metadata",
    )

    parser.add_argument("--matting_model", type=str, default="u2netp")
    parser.add_argument("--background", choices=["white", "black"], default="white")

    parser.add_argument("--no_ssim", action="store_true")
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--no_embedding", action="store_true")
    parser.add_argument("--no_matted_ssim", action="store_true")
    parser.add_argument("--no_matted_embedding", action="store_true")

    args = parser.parse_args()

    if args.ssim_size == 0:
        args.ssim_size = None
    if args.lpips_size == 0:
        args.lpips_size = None

    os.makedirs(args.out_dir, exist_ok=True)
    cache_bucket, cache_prefix = _parse_s3_dir(args.cache_dir)
    data_bucket, data_prefix = _parse_s3_dir(args.data_dir)

    # Load metadata once and reuse across all metrics.
    metadata_path = os.path.join(args.out_dir, "image_metadata.csv")
    use_cached_metadata = args.cached_metadata and os.path.exists(metadata_path)
    df = sim_utils.load_or_build_image_metadata(
        out_dir=args.out_dir,
        cached_metadata=use_cached_metadata,
        cache_bucket=cache_bucket,
        cache_prefix=cache_prefix,
        data_bucket=data_bucket,
        data_prefix=data_prefix,
    )
    df = sim_utils.make_pair_id_columns(df)
    df_pairs = sim_utils.filter_valid_pair_rows(df, min_per_pair=2)
    print("Metadata images:", len(df))
    print("Valid pair rows:", len(df_pairs), "unique pairs:", df_pairs["pair_id"].nunique(dropna=True))

    store = S3ImageStore(S3PathContext(cache_bucket, cache_prefix, data_bucket, data_prefix))

    if not args.no_ssim:
        run_ssim(df=df, df_pairs=df_pairs, store=store, out_dir=args.out_dir, ssim_size=args.ssim_size)
    if not args.no_lpips:
        run_lpips(
            df=df,
            df_pairs=df_pairs,
            store=store,
            out_dir=args.out_dir,
            lpips_net=args.lpips_net,
            lpips_size=args.lpips_size,
            lpips_batch_size=args.lpips_batch_size,
            amp=args.amp,
        )
    if not args.no_embedding:
        run_clip_embedding_similarity(
            df=df,
            df_pairs=df_pairs,
            store=store,
            out_dir=args.out_dir,
            embedding_model=args.embedding_model,
            batch_size=args.embedding_batch_size,
            resume=args.resume,
        )
    if not args.no_matted_ssim:
        run_matted_ssim(
            df=df,
            df_pairs=df_pairs,
            store=store,
            out_dir=args.out_dir,
            ssim_size=args.ssim_size,
            matting_model=args.matting_model,
            background=args.background,
        )
    if not args.no_matted_embedding:
        run_matted_embedding_similarity(
            df=df,
            df_pairs=df_pairs,
            store=store,
            out_dir=args.out_dir,
            embedding_model=args.embedding_model,
            batch_size=args.embedding_batch_size,
            resume=args.resume,
            matting_model=args.matting_model,
            background=args.background,
        )


if __name__ == "__main__":
    main()
