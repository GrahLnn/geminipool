import os
import asyncio
import httpx
from typing import Optional, Tuple, Dict
from .llm_settings import LLMSettings

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


class AsyncGeminiClient:
    def __init__(self):
        self.settings = LLMSettings()
        # 用异步客户端
        self.client = httpx.AsyncClient(timeout=None)
        self.api_key = None
        self.safe = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self.generation_config = {"temperature": 1, "topP": 0.95}

    def _get_mime_type(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_MIMES:
            raise ValueError(f"Unsupported file format: {ext}")
        return SUPPORTED_MIMES[ext]

    async def _upload_file(self, file_path: str) -> Tuple[str, str]:
        """
        异步上传文件
        """
        mime_type = self._get_mime_type(file_path)
        file_size = os.path.getsize(file_path)
        display_name = os.path.basename(file_path)

        url = f"{self.settings.gemini_base_url}/upload/v1beta/files?key={self.api_key}"
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        body = {"file": {"display_name": display_name}}
        r = await self.client.post(url, headers=headers, json=body)
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
        r = await self.client.post(upload_url, headers=headers, content=data)
        r.raise_for_status()
        file_info: Dict = r.json()

        resource: Dict = file_info.get("file", file_info)
        file_name = resource["name"]
        state = resource.get("state")

        # 等待状态 ACTIVE
        while state == "PROCESSING":
            await asyncio.sleep(1)
            get_url = f"{self.settings.gemini_base_url}/v1beta/{file_name}?key={self.api_key}"
            rr = await self.client.get(get_url)
            rr.raise_for_status()
            info = rr.json()
            state = info.get("state")

        if state != "ACTIVE":
            raise RuntimeError("Uploaded file is not ACTIVE")
        return resource["uri"], resource["name"]

    async def _delete_file(self, file_name: str):
        url = f"{self.settings.gemini_base_url}/v1beta/{file_name}?key={self.api_key}"
        r = await self.client.delete(url)
        r.raise_for_status()

    async def _content_with_media(self, prompt: str, file: str) -> str:
        mime_type = self._get_mime_type(file)
        file_size = os.path.getsize(file)
        UPLOAD_LIMIT_BYTES = 4 * 1024 * 1024  # 4MB

        parts = []
        uploaded_files_info = []

        if file_size <= UPLOAD_LIMIT_BYTES:
            # base64 inline
            import base64
            with open(file, "rb") as f:
                data_b64 = base64.b64encode(f.read()).decode()
            parts.append({"inline_data": {"mime_type": mime_type, "data": data_b64}})
        else:
            # 走 resumable 上传
            file_uri, file_name = await self._upload_file(file)
            parts.append({"file_data": {"mime_type": mime_type, "file_uri": file_uri}})
            uploaded_files_info.append(file_name)

        # 最后加 prompt
        parts.append({"text": prompt})
        contents = [{"parts": parts}]

        url = f"{self.settings.gemini_base_url}/v1beta/models/{self.settings.model}:generateContent?key={self.api_key}"
        try:
            r = await self.client.post(
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
                return "No candidates returned"

            text_parts = candidates[0].get("content", {}).get("parts", [])
            texts = [p.get("text", "") for p in text_parts if p.get("text")]
            return "\n".join(texts) if texts else ""
        finally:
            # 清理上传文件
            for f in uploaded_files_info:
                await self._delete_file(f)

    async def _content_with_text(self, prompt: str) -> str:
        contents = [{"parts": [{"text": prompt}]}]

        url = f"{self.settings.gemini_base_url}/v1beta/models/{self.settings.model}:generateContent?key={self.api_key}"
        r = await self.client.post(
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
            return "No candidates returned"

        text_parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p.get("text", "") for p in text_parts if p.get("text")]
        return "\n".join(texts) if texts else ""

    async def llmgen_content(self, prompt: str, key: str, media: Optional[str] = None) -> str:
        """
        真正执行“生成内容”的方法：
          - 由外部传入 key（我们不在这里再去 KeyManager 拿 key）
          - 根据是否有media，走不同的处理
        """
        self.api_key = key
        if media:
            return await self._content_with_media(prompt, media)
        else:
            return await self._content_with_text(prompt)
