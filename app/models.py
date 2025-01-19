import asyncio
import os
import random
import time
from collections import deque

# -----------------------------
# Provider相关Enum和配置
# -----------------------------
from enum import Enum
from threading import Condition, Lock
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import Field
from pydantic_settings import BaseSettings


class Provider(Enum):
    GEMINI = "gemini"
    DEFAULT = "default"


SUPPORTED_IMAGE_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}
SUPPORTED_VIDEO_MIMES = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".mpg": "video/mpg",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
}
SUPPORTED_MIMES = {**SUPPORTED_IMAGE_MIMES, **SUPPORTED_VIDEO_MIMES}


# -----------------------------
# LLMSettings
# -----------------------------
class LLMSettings(BaseSettings):
    gemini_api_keys: Optional[List[str]] = Field(default=None, validate_default=True)
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com")
    rpm: int = Field(default=10)
    allow_concurrent: bool = Field(default=False)
    model: str = Field(default="gemini-2.0-flash-exp")


# -----------------------------
# KeyManager
# -----------------------------
class AsyncKeyManager:
    def __init__(
        self,
        rpm: int = 15,  # 每分钟每个密钥可用的请求次数上限
        rpd: Optional[int] = None,  # 可能的日限、暂不演示
        allow_concurrent: bool = False,
        cooldown_time: int = 60,  # 进入冷却状态持续时间（秒）
    ):
        self.rpm = rpm
        self.allow_concurrent = allow_concurrent
        self.cooldown_time = cooldown_time

        # 每个密钥对应的请求时间戳队列，用于计算 RPM 限制
        self.request_counts: Dict[str, deque] = {}
        # 冷却中的密钥，值为冷却结束的时间戳
        self.cooldown_keys: Dict[str, float] = {}
        # 连续冷却计数
        self.consecutive_cooldown_counts: Dict[str, int] = {}

        # 当前被占用的密钥（如果不允许并发，则同一时间只允许一个线程/协程占用）
        self.occupied_keys: set = set()

        # 使用 asyncio 的锁和条件变量
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

    def _hash_key(self, key: Any) -> str:
        """生成密钥的哈希表示，用于内部管理"""
        return str(hash(str(key)))

    def _clean_old_requests(self, internal_key: str, current_time: float):
        """移除超过60秒之前的请求记录以适应 RPM 限制"""
        if internal_key not in self.request_counts:
            self.request_counts[internal_key] = deque()
        dq = self.request_counts[internal_key]
        while dq and dq[0] <= current_time - 60:
            dq.popleft()

    def _is_key_available(self, internal_key: str, current_time: float) -> bool:
        """
        检查 key 在当前时刻是否可用：
        - 若仍在 cooldown 或 ban 中则不可用
        - 若 cooldown/ban 过期则恢复可用
        - 检查 RPM / 并发占用
        """
        # 还在冷却期
        if internal_key in self.cooldown_keys:
            if current_time < self.cooldown_keys[internal_key]:
                return False
            else:
                # 冷却期已结束，删除记录
                del self.cooldown_keys[internal_key]

        # 检查 RPM
        self._clean_old_requests(internal_key, current_time)
        if len(self.request_counts[internal_key]) >= self.rpm:
            return False

        # 并发限制
        if not self.allow_concurrent and internal_key in self.occupied_keys:
            return False

        return True

    def _get_wait_time_for_key(self, internal_key: str, current_time: float) -> float:
        """
        计算下一个该密钥可能变为可用状态的等待时间（秒）。
        可能为0表示无需等待。如果非常大，说明暂时不可用。
        """
        wait_time = 0.0

        # 如果在冷却中
        if (
            internal_key in self.cooldown_keys
            and current_time < self.cooldown_keys[internal_key]
        ):
            wait_time = max(wait_time, self.cooldown_keys[internal_key] - current_time)

        # RPM 达到上限
        if (
            internal_key in self.request_counts
            and len(self.request_counts[internal_key]) >= self.rpm
        ):
            oldest_request = self.request_counts[internal_key][0]
            rpm_wait = (oldest_request + 60) - current_time
            wait_time = max(wait_time, rpm_wait)

        # 并发限制
        if not self.allow_concurrent and internal_key in self.occupied_keys:
            # 这里无法精确计算等待时间，只能设置为0；依赖自带的 condition 唤醒机制
            wait_time = max(wait_time, 0)

        return wait_time

    async def mark_key_used(self, key: Any):
        """
        标记密钥被使用一次，并添加请求时间戳。
        如果不允许并发，则将此 key 放入 occupied_keys。
        """
        internal_key = self._hash_key(key)
        async with self._condition:
            current_time = time.time()
            self._clean_old_requests(internal_key, current_time)
            self.request_counts[internal_key].append(current_time)

            if not self.allow_concurrent:
                self.occupied_keys.add(internal_key)

            # 唤醒等待中的协程（可能有其他协程在等这个 key 的释放）
            self._condition.notify_all()

    async def release_key(self, key: Any):
        """释放密钥占用"""
        internal_key = self._hash_key(key)
        async with self._condition:
            if internal_key in self.occupied_keys:
                self.occupied_keys.remove(internal_key)
            self._condition.notify_all()

    async def mark_key_cooldown(self, key: Any):
        """
        将密钥标记为进入冷却状态，如果连续3次进入则进入1小时冷却
        """
        internal_key = self._hash_key(key)
        async with self._condition:
            current_time = time.time()
            self.consecutive_cooldown_counts[internal_key] = (
                self.consecutive_cooldown_counts.get(internal_key, 0) + 1
            )

            if self.consecutive_cooldown_counts[internal_key] >= 3:
                self.cooldown_keys[internal_key] = current_time + 3600  # 1小时冷却
            else:
                self.cooldown_keys[internal_key] = current_time + self.cooldown_time

            if internal_key in self.occupied_keys:
                self.occupied_keys.remove(internal_key)
            self._condition.notify_all()

    async def get_available_key(self, keys: List[Any]) -> Any:
        """
        获取一个可用的密钥，如果没有可用的则阻塞等待（异步等待）。
        """
        if not keys:
            raise ValueError("未提供任何 API 密钥")

        while True:
            current_time = time.time()
            async with self._condition:
                shuffled_keys = keys.copy()
                random.shuffle(shuffled_keys)

                # 优先找真正可用的 key
                available_keys = []
                for k in shuffled_keys:
                    internal_key = self._hash_key(k)
                    if self._is_key_available(internal_key, current_time):
                        available_keys.append(k)

                if available_keys:
                    # 优先选择未被占用的 key
                    for k in available_keys:
                        internal_key = self._hash_key(k)
                        if self.allow_concurrent or (
                            internal_key not in self.occupied_keys
                        ):
                            # 确保请求队列更新
                            self._clean_old_requests(internal_key, current_time)
                            if not self.allow_concurrent:
                                self.occupied_keys.add(internal_key)
                            return k

                # 如果都不可用，则计算最小等待时间
                min_wait_time = float("inf")
                for k in keys:
                    internal_key = self._hash_key(k)
                    w = self._get_wait_time_for_key(internal_key, current_time)
                    if w < min_wait_time:
                        min_wait_time = w

                if min_wait_time == float("inf"):
                    raise RuntimeError("无法确定密钥的等待时间，可能没有可用密钥。")

                # 如果等待时间>=4小时，说明极可能是API当日限额用尽，打印提示
                if min_wait_time >= 4 * 3600:
                    print("所有key都在冷却或不可用，可能当日额度已用尽。")
                    # 你可以在这里选择抛出异常，或继续等待

                # 使用 condition.wait_for() 等待 min_wait_time 秒
                # 期间如果别的协程唤醒，也可提前返回
                if min_wait_time > 0:
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(), timeout=min_wait_time
                        )
                    except asyncio.TimeoutError:
                        # 等待时间到了，循环回去再检查可用性
                        pass
                else:
                    # 如果没有具体等待时间，就单纯 await self._condition.wait()
                    await self._condition.wait()

    def context(self, keys: List[Any]):
        """
        提供一个异步上下文管理器，用法：
            async with key_manager.context(keys) as key:
                # 此处即已拿到可用key
        """
        manager = self

        class AsyncKeyContext:
            def __init__(self, manager: AsyncKeyManager, keylist: List[Any]):
                self.manager = manager
                self.keylist = keylist
                self.key = None
                self.entered = False

            async def __aenter__(self):
                # 拿到可用 key
                self.key = await self.manager.get_available_key(self.keylist)
                await self.manager.mark_key_used(self.key)
                self.entered = True
                return self.key

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.entered:
                    # 如果没异常，说明正常使用，重置连续冷却计数
                    if exc_type is None:
                        internal_key = self.manager._hash_key(self.key)
                        self.manager.consecutive_cooldown_counts[internal_key] = 0
                    # 释放占用
                    await self.manager.release_key(self.key)

        return AsyncKeyContext(manager, keys)


# -----------------------------
# GeminiClient
# -----------------------------
class GeminiClient:
    def __init__(self, key=None):
        self.settings = LLMSettings()
        self.key_manager = AsyncKeyManager()
        self.client = httpx.Client(timeout=3600)
        self.api_key = key
        self.safe = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self.generation_config = {"temperature": 1, "topP": 0.95}

    def _get_mime_type(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_MIMES:
            raise ValueError(f"Unsupported file format: {ext}")
        return SUPPORTED_MIMES[ext]

    def _base64_encode_file(self, file_path: str) -> str:
        """Base64编码文件内容"""
        import base64

        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _upload_file(self, file_path: str) -> Tuple[str, str]:
        mime_type = self._get_mime_type(file_path)
        file_size = os.path.getsize(file_path)
        display_name = os.path.basename(file_path)

        # step1: 获得上传URL
        url = f"{self.settings.gemini_base_url}/upload/v1beta/files?key={self.api_key}"
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        body = {"file": {"display_name": display_name}}
        r = self.client.post(url, headers=headers, json=body)
        r.raise_for_status()
        upload_url = r.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise RuntimeError("Failed to get upload URL")

        # step2: 上传数据
        with open(file_path, "rb") as f:
            data = f.read()
        headers = {
            "Content-Length": str(file_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }
        r = self.client.post(upload_url, headers=headers, content=data)
        r.raise_for_status()
        file_info: Dict = r.json()

        # Normalize the response to always have a uniform structure
        resource: Dict = file_info.get("file", file_info)
        file_name = resource["name"]
        state = resource.get("state")

        # 等待文件状态为 ACTIVE
        while state == "PROCESSING":
            time.sleep(1)
            get_url = (
                f"{self.settings.gemini_base_url}/v1beta/{file_name}?key={self.api_key}"
            )
            rr = self.client.get(get_url)
            rr.raise_for_status()
            info = rr.json()
            resource = info
            state = resource.get("state")

        if state != "ACTIVE":
            raise RuntimeError("Uploaded file is not ACTIVE")
        return resource["uri"], resource["name"]

    def _list_files(self):
        url = f"{self.settings.gemini_base_url}/v1beta/files?key={self.api_key}"
        r = self.client.get(url)
        r.raise_for_status()
        return [file.get("name") for file in r.json().get("files", [])]

    def _delete_file(self, file_name: str):
        url = f"{self.settings.gemini_base_url}/v1beta/{file_name}?key={self.api_key}"
        r = self.client.delete(url)
        r.raise_for_status()

    def _content_with_media(self, prompt: str, file: str) -> str:
        contents = []
        parts = []
        uploaded_files_info = []
        # 将文件先放，再放文本提示
        mime_type = self._get_mime_type(file)
        file_size = os.path.getsize(file)
        UPLOAD_LIMIT_BYTES = 4 * 1024 * 1024  # 4MB
        if file_size <= UPLOAD_LIMIT_BYTES:
            data_b64 = self._base64_encode_file(file)
            parts.append({"inline_data": {"mime_type": mime_type, "data": data_b64}})
        else:
            file_uri, file_name = self._upload_file(file)
            parts.append({"file_data": {"mime_type": mime_type, "file_uri": file_uri}})
            uploaded_files_info.append(file_name)

        # 最后加上文本提示
        parts.append({"text": prompt})
        contents.append({"parts": parts})

        url = f"{self.settings.gemini_base_url}/v1beta/models/{self.settings.model}:generateContent?key={self.api_key}"
        try:
            r = self.client.post(
                url,
                json={
                    "generationConfig": self.generation_config,
                    "safetySettings": self.safe,
                    "contents": contents,
                },
            )

            r.raise_for_status()
            resp = r.json()

            candidates = resp.get("candidates", [])
            if not candidates:
                for f in uploaded_files_info:
                    self._delete_file(f)
                return None

            texts = [
                p.get("text", "")
                for p in candidates[0].get("content", {}).get("parts", [])
                if p.get("text")
            ]
            result = "\n".join(texts) if texts else None
        finally:
            for f in uploaded_files_info:
                self._delete_file(f)
        return result

    def _content_with_text(self, prompt: str) -> str:
        contents = []
        parts = []
        parts.append({"text": prompt})
        contents.append({"parts": parts})

        url = f"{self.settings.gemini_base_url}/v1beta/models/{self.settings.model}:generateContent?key={self.api_key}"
        r = self.client.post(
            url,
            json={
                "generationConfig": self.generation_config,
                "safetySettings": self.safe,
                "contents": contents,
            },
        )
        r.raise_for_status()
        response = r.json()

        if "candidates" not in response:
            raise Exception(f"No response from Gemini API: {response=}")

        content = (
            response.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text")
        )
        if content:
            return content
        raise Exception(f"Error response from Gemini API: {response=}")

    def llmgen_content(self, prompt: str, media: Optional[str] = None) -> str:
        """生成内容的主要方法"""
        if not self.settings.gemini_api_keys:
            raise ValueError("No API keys available")

        while True:
            key = self.key_manager.get_available_key(self.settings.gemini_api_keys)
            if not key:
                time.sleep(1)  # 等待key可用
                continue

            try:
                self.api_key = key
                self.key_manager.mark_key_used(key)

                if media:
                    result = self._content_with_media(prompt, media)
                else:
                    result = self._content_with_text(prompt)

                return result

            except Exception as e:
                self.key_manager.mark_key_error(key)
                raise e

            finally:
                self.key_manager.release_key(key)
