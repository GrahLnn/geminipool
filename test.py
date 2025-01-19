import asyncio
import httpx

async def test_concurrent():
    """测试并发请求"""
    tasks = [
        httpx.AsyncClient(timeout=3600).get("http://localhost:8000/v1/test_concurrent")
        for _ in range(5)
    ]
    
    try:
        responses = await asyncio.gather(*tasks)
        for i, resp in enumerate(responses, 1):
            print(f"请求 {i} 结果:", resp.json())
    except Exception as e:
        print(f"发生错误: {str(e)}")

async def main():
    await test_concurrent()

if __name__ == "__main__":
    asyncio.run(main())
