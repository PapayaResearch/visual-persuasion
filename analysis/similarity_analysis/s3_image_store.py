import io
import threading
import boto3
import numpy as np
import torch
from collections import OrderedDict
from dataclasses import dataclass
from PIL import Image, UnidentifiedImageError
from typing import Any, Callable


class LruCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._data: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        if self.max_items <= 0:
            return None
        with self._lock:
            value = self._data.get(key)
            if value is None:
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: Any, value: Any) -> None:
        if self.max_items <= 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)


@dataclass(frozen=True)
class S3PathContext:
    cache_bucket: str
    cache_prefix: str
    data_bucket: str
    data_prefix: str

    def resolve_bucket(self, path: str) -> str:
        if path.startswith(self.data_prefix):
            return self.data_bucket
        if path.startswith(self.cache_prefix):
            return self.cache_bucket
        return self.cache_bucket


class S3ImageStore:
    """
    Shared, in-process cache for S3 objects + decoded image representations.

    This is designed for pipeline-style usage where multiple metrics are computed
    in the same Python process and we want to avoid re-downloading or re-decoding.
    """

    def __init__(
        self,
        ctx: S3PathContext,
        *,
        max_object_items: int = 2048,
        max_pil_items: int = 1024,
        max_array_items: int = 1024,
        max_tensor_items: int = 1024,
        max_matted_items: int = 512,
        s3_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.ctx = ctx
        self._thread_local = threading.local()
        self._s3_client_factory = s3_client_factory

        self._object_cache = LruCache(max_object_items)
        self._pil_cache = LruCache(max_pil_items)
        self._array_cache = LruCache(max_array_items)
        self._tensor_cache = LruCache(max_tensor_items)
        self._matted_cache = LruCache(max_matted_items)

        self._rembg_sessions: dict[str, Any] = {}
        self._rembg_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._logged_decode_failures: set[str] = set()
        self._missing_sentinel = object()

    def _get_s3_client(self):
        client = getattr(self._thread_local, "s3_client", None)
        if client is None:
            if self._s3_client_factory is not None:
                client = self._s3_client_factory()
            else:
                client = boto3.client("s3")
            self._thread_local.s3_client = client
        return client

    def get_bytes(self, path: str) -> bytes:
        cached = self._object_cache.get(path)
        if cached is not None:
            return cached
        bucket = self.ctx.resolve_bucket(path)
        obj = self._get_s3_client().get_object(Bucket=bucket, Key=path)
        data = obj["Body"].read()
        self._object_cache.set(path, data)
        return data

    def get_pil_rgb(self, path: str) -> Image.Image | None:
        cache_key = (path, "RGB")
        cached = self._pil_cache.get(cache_key)
        if cached is not None:
            return None if cached is self._missing_sentinel else cached
        raw = self.get_bytes(path)
        try:
            with Image.open(io.BytesIO(raw)) as image:
                pil = image.convert("RGB")
        except UnidentifiedImageError as e:
            bucket = self.ctx.resolve_bucket(path)
            head = raw[:64]
            with self._log_lock:
                if path not in self._logged_decode_failures:
                    self._logged_decode_failures.add(path)
                    print(
                        f"[PIL decode failed] s3_bucket={bucket!r} s3_key={path!r} bytes={len(raw)} head_hex={head.hex()} error={e}",
                        flush=True,
                    )
            self._pil_cache.set(cache_key, self._missing_sentinel)
            return None
        self._pil_cache.set(cache_key, pil)
        return pil

    def get_numpy_rgb_uint8(self, path: str, size: int | None) -> np.ndarray | None:
        cache_key = (path, "np_rgb_uint8", size)
        cached = self._array_cache.get(cache_key)
        if cached is not None:
            return None if cached is self._missing_sentinel else cached
        image = self.get_pil_rgb(path)
        if image is None:
            self._array_cache.set(cache_key, self._missing_sentinel)
            return None
        if size is not None:
            image = image.resize((size, size), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.uint8)
        self._array_cache.set(cache_key, arr)
        return arr

    def get_lpips_tensor(self, path: str, size: int | None) -> torch.Tensor | None:
        cache_key = (path, "lpips_tensor", size)
        cached = self._tensor_cache.get(cache_key)
        if cached is not None:
            return None if cached is self._missing_sentinel else cached
        image = self.get_pil_rgb(path)
        if image is None:
            self._tensor_cache.set(cache_key, self._missing_sentinel)
            return None
        if size is not None:
            image = image.resize((size, size), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).mul(2.0).sub(1.0)
        self._tensor_cache.set(cache_key, tensor)
        return tensor

    def _get_rembg_session(self, model_name: str):
        with self._rembg_lock:
            session = self._rembg_sessions.get(model_name)
            if session is None:
                from rembg import new_session  # local import (expensive)

                session = new_session(model_name=model_name)
                self._rembg_sessions[model_name] = session
            return session

    def get_matted_pil_rgb(self, path: str, *, matting_model: str, background: str) -> Image.Image | None:
        cache_key = (path, "matted_rgb", matting_model, background)
        cached = self._matted_cache.get(cache_key)
        if cached is not None:
            return None if cached is self._missing_sentinel else cached

        from rembg import remove  # local import (expensive)

        image = self.get_pil_rgb(path)
        if image is None:
            self._matted_cache.set(cache_key, self._missing_sentinel)
            return None
        rgba = remove(image, session=self._get_rembg_session(matting_model))
        if rgba.mode != "RGBA":
            rgba = rgba.convert("RGBA")
        bg_color = (255, 255, 255, 255) if background == "white" else (0, 0, 0, 255)
        background_img = Image.new("RGBA", rgba.size, bg_color)
        composed = Image.alpha_composite(background_img, rgba).convert("RGB")
        self._matted_cache.set(cache_key, composed)
        return composed
