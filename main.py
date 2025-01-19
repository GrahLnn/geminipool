import asyncio
import os
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Body,
    Query,
)
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
import shutil
from pathlib import Path

# 引入自己的模块
from app.models.concurrency import AsyncKeyManager
from app.models.gemini_client import AsyncGeminiClient

app = FastAPI()

# ==========================
# 0) Key & 目录配置
# ==========================
UPLOAD_DIR = Path("./uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

KEYS_FILE_PATH = Path("./keys.json")
if not KEYS_FILE_PATH.exists():
    KEYS_FILE_PATH.write_text(json.dumps({"gemini": []}, ensure_ascii=False))


def load_keys() -> List[str]:
    data = json.loads(KEYS_FILE_PATH.read_text(encoding="utf-8"))
    return data.get("gemini", [])


def save_keys(keys: List[str]):
    with open(KEYS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump({"gemini": keys}, f, ensure_ascii=False)


# 读取现有 key
initial_keys = load_keys()

# 创建全局的异步 KeyManager & GeminiClient
key_manager = AsyncKeyManager(rpm=2, allow_concurrent=False)
async_gemini_client = AsyncGeminiClient()
# 可以把 rpm/allow_concurrent 改成你想要的


@app.get("/")
async def index():
    return JSONResponse({"message": "Gemini Docker Service is running"})


# ==========================
# 1) 文件上传接口
# ==========================
@app.post("/v1/files")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    上传文件到本地服务器保存，返回可在后续请求中使用的本地路径。
    并在后台注册一个2小时后的延迟删除。
    """
    file_location = UPLOAD_DIR / file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 这里安排一个后台任务：2小时后删除文件
    TWO_HOURS = 2 * 60 * 60
    background_tasks.add_task(delayed_file_remove, str(file_location), TWO_HOURS)

    return {
        "file_path": str(file_location),
        "detail": "File uploaded successfully, will auto-delete in 2 hours",
    }


# ==========================
# 2) 文本生成接口
# ==========================
@app.post("/v1/llmgen_content")
async def llmgen_content(
    prompt: str = Body(..., embed=True),
    file_path: Optional[str] = Body(None, embed=True),
):
    """
    如果传入 file_path，则执行多模态逻辑。用完后立即删除。
    """
    try:
        async with key_manager.context(initial_keys) as key:
            result = await async_gemini_client.llmgen_content(
                prompt, key, media=file_path
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 若有 file_path，则在用完后立即删除
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"[llmgen_content] error removing file {file_path}: {e}")

    return {"response": result}


# ==========================
# 3) Key 管理接口
# ==========================
@app.get("/v1/keys")
async def get_keys():
    return {"keys": load_keys()}


@app.post("/v1/keys")
async def add_key(new_key: str = Body(..., embed=True)):
    keys = load_keys()
    if new_key in keys:
        raise HTTPException(status_code=400, detail="Key already exists.")
    keys.append(new_key)
    save_keys(keys)
    return {"detail": "New key added successfully."}


@app.delete("/v1/keys")
async def delete_key(key: str = Query(...)):
    keys = load_keys()
    if key not in keys:
        raise HTTPException(status_code=404, detail="Key not found.")
    keys.remove(key)
    save_keys(keys)
    return {"detail": "Key deleted successfully."}


async def delayed_file_remove(file_path: str, delay_seconds: int):
    """
    等待 delay_seconds 后再删除文件，若期间文件已被删除或不存在，不报错。
    """
    try:
        await asyncio.sleep(delay_seconds)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        # 日志打印，或者仅忽略
        print(f"[delayed_file_remove] Error removing file {file_path}: {e}")
