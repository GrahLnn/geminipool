import asyncio
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional
from datetime import date


class AsyncKeyManager:
    def __init__(
        self,
        rpm: int = 15,  # 每分钟每个密钥可用的请求次数
        rpd: int = 1500,  # 每日总请求上限, 默认1500
        allow_concurrent: bool = False,
        cooldown_time: int = 60,  # 进入冷却状态持续时间（秒）
    ):
        self.rpm = rpm
        # 如果外部没传 rpd，就用 1500
        self.rpd = rpd
        self.allow_concurrent = allow_concurrent
        self.cooldown_time = cooldown_time

        # 每个密钥对应最近60秒内的请求时间戳队列（做 RPM）
        self.request_counts: Dict[str, deque] = {}

        # 记录哪些 key 在冷却中：key_hash -> 冷却结束的时间戳
        self.cooldown_keys: Dict[str, float] = {}

        # 连续冷却计数 key_hash -> 次数
        self.consecutive_cooldown_counts: Dict[str, int] = {}

        # 当前被占用的密钥
        self.occupied_keys: set = set()

        # =============== 新增的每日使用限制相关 ===============
        # 每个 key 的当日使用计数 key_hash -> int
        self.daily_usage_counts: Dict[str, int] = {}
        # 用来判断是否到了新的一天，需要重置计数
        self._day_tag: Optional[date] = None

        # asyncio 锁 & 条件
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

    # =========== 新增: 重置每日计数的辅助方法 ===========
    def _reset_daily_usage_if_needed(self):
        """
        如果今天已变更（跨日），就重置所有 key 的当日使用计数
        """
        today = date.today()
        if self._day_tag != today:
            # 新的一天，重置计数
            self.daily_usage_counts = {}
            self._day_tag = today

    def _hash_key(self, key: Any) -> str:
        return str(hash(str(key)))

    def _clean_old_requests(self, internal_key: str, now: float):
        """清理掉超过60秒的请求记录, 做 RPM 限制"""
        if internal_key not in self.request_counts:
            self.request_counts[internal_key] = deque()

        dq = self.request_counts[internal_key]
        while dq and dq[0] <= now - 60:
            dq.popleft()

    def _is_key_available(self, internal_key: str, now: float) -> bool:
        """判断某个 key 在当前时刻是否可用"""
        # 1) 是否在冷却期
        if internal_key in self.cooldown_keys:
            if now < self.cooldown_keys[internal_key]:
                return False
            else:
                # 冷却结束，移除记录
                del self.cooldown_keys[internal_key]

        # 2) 是否达到当日总请求上限 (rpd)
        daily_used = self.daily_usage_counts.get(internal_key, 0)
        if daily_used >= self.rpd:
            # 当日使用数已达上限
            return False

        # 3) RPM 限制
        self._clean_old_requests(internal_key, now)
        if len(self.request_counts[internal_key]) >= self.rpm:
            return False

        # 4) 并发限制
        if not self.allow_concurrent and internal_key in self.occupied_keys:
            return False

        return True

    def _get_wait_time_for_key(self, internal_key: str, now: float) -> float:
        """计算下一个该密钥可用所需等待时间"""
        wait_time = 0.0

        # 若在冷却中
        if (
            internal_key in self.cooldown_keys
            and now < self.cooldown_keys[internal_key]
        ):
            wait_time = max(wait_time, self.cooldown_keys[internal_key] - now)

        # 若达到 RPM 上限
        if (
            internal_key in self.request_counts
            and len(self.request_counts[internal_key]) >= self.rpm
        ):
            oldest = self.request_counts[internal_key][0]
            rpm_wait = (oldest + 60) - now
            wait_time = max(wait_time, rpm_wait)

        # 若达到当日 rpd 上限，则理应无法再用，理论上等到明天才能用
        # 这里可以直接设一个很大值，比如等到 24 小时后
        daily_used = self.daily_usage_counts.get(internal_key, 0)
        if daily_used >= self.rpd:
            # 剩余秒数 = 明日零点 - 当前时间
            # 只是示例，简单给个长时间(24h=86400s)，你也可计算准确剩余秒数
            wait_time = max(wait_time, 86400)

        # 并发限制
        if not self.allow_concurrent and internal_key in self.occupied_keys:
            # 无法算精确时间；等别的协程释放
            wait_time = max(wait_time, 0)

        return wait_time

    async def mark_key_used(self, key: Any):
        """标记此 key 被使用: RPM + 当日计数 都要更新"""
        internal_key = self._hash_key(key)
        async with self._condition:
            now = time.time()
            self._clean_old_requests(internal_key, now)
            self.request_counts[internal_key].append(now)

            # 当日使用计数 +1
            current_count = self.daily_usage_counts.get(internal_key, 0)
            self.daily_usage_counts[internal_key] = current_count + 1

            if not self.allow_concurrent:
                self.occupied_keys.add(internal_key)

            self._condition.notify_all()

    async def release_key(self, key: Any):
        """释放此 key 占用"""
        internal_key = self._hash_key(key)
        async with self._condition:
            if internal_key in self.occupied_keys:
                self.occupied_keys.remove(internal_key)
            self._condition.notify_all()

    async def mark_key_cooldown(self, key: Any, reason=""):
        """把 key 标记为冷却（如遇到报错等情况）"""
        internal_key = self._hash_key(key)
        async with self._condition:
            now = time.time()
            self.consecutive_cooldown_counts[internal_key] = (
                self.consecutive_cooldown_counts.get(internal_key, 0) + 1
            )

            if self.consecutive_cooldown_counts[internal_key] >= 3:
                # 连续三次 -> 冷却 1 小时
                self.cooldown_keys[internal_key] = now + 3600
            else:
                self.cooldown_keys[internal_key] = now + self.cooldown_time

            if internal_key in self.occupied_keys:
                self.occupied_keys.remove(internal_key)
            print(f"[KeyManager] key={key} -> cooldown ({reason})")
            self._condition.notify_all()

    async def get_available_key(self, keys: List[Any]) -> Any:
        """
        在异步环境下阻塞等待，直到拿到可用 key 或抛出异常
        """
        if not keys:
            raise ValueError("没有可用的keys")

        while True:
            now = time.time()
            async with self._condition:
                # 先检查是否需要重置每日计数
                self._reset_daily_usage_if_needed()

                # 随机打乱，避免每次都用同一个 key
                shuffled = keys.copy()
                random.shuffle(shuffled)

                available = []
                for k in shuffled:
                    ik = self._hash_key(k)
                    if self._is_key_available(ik, now):
                        available.append(k)

                if available:
                    # 如果有可用 key，就取其中第一个(或你可做更多策略，如最少使用优先)
                    key = available[0]
                    ik = self._hash_key(key)
                    # 占用
                    self._clean_old_requests(ik, now)
                    if not self.allow_concurrent:
                        self.occupied_keys.add(ik)
                    return key

                # 如果没有可用 key，计算最短等待时间
                min_wait = float("inf")
                for k in keys:
                    ik = self._hash_key(k)
                    wt = self._get_wait_time_for_key(ik, now)
                    if wt < min_wait:
                        min_wait = wt

                if min_wait == float("inf"):
                    # 说明所有 key 都彻底用尽或都达日限
                    raise RuntimeError("没有可用 key，并且无法计算下一次可用时间")

                # 如果超过4h，说明日限之类用尽
                if min_wait >= 4 * 3600:
                    print(
                        "所有 key 都不可用，可能日限已用尽，你可以在这里决定是否抛异常或继续等待"
                    )

                # 带超时地等待 condition
                if min_wait > 0:
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=min_wait)
                    except asyncio.TimeoutError:
                        # 等待时间到了，循环回去再检查
                        pass
                else:
                    # min_wait <= 0，就单纯等别的协程唤醒
                    await self._condition.wait()

    def context(self, keys: List[Any]):
        """
        提供一个异步上下文，用于自动获取/释放 key。
        用法示例：
            async with key_manager.context(key_list) as key:
                # 拿到 key 做操作
        """
        manager = self

        class AsyncKeyContext:
            def __init__(self, mgr: AsyncKeyManager, keylist: List[Any]):
                self.manager = mgr
                self.keylist = keylist
                self.key = None
                self.entered = False

            async def __aenter__(self):
                self.key = await self.manager.get_available_key(self.keylist)
                await self.manager.mark_key_used(self.key)
                print(f"[AsyncKeyContext] mark key={self.key}")
                self.entered = True
                return self.key

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.entered:
                    # 如果没有异常，重置连续冷却计数
                    if exc_type is None:
                        ik = manager._hash_key(self.key)
                        manager.consecutive_cooldown_counts[ik] = 0
                    else:
                        # 若有异常，可视场景给key加冷却
                        await self.manager.mark_key_cooldown(
                            self.key, reason="Error in usage"
                        )
                    print(f"[AsyncKeyContext] release key={self.key}")
                    await self.manager.release_key(self.key)

        return AsyncKeyContext(manager, keys)
