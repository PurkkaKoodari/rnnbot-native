import aiohttp
import asyncio
import json
import logging

LOGGER = logging.getLogger("asyncTG")

class Bot:
    def __init__(self, token):
        self._token = token
        self._offset = None
        self._bot_user = None
        self._session = aiohttp.ClientSession()
    
    async def get_me(self):
        if self._bot_user is None:
            self._bot_user = await self.request("getMe")
        return self._bot_user
    
    @property
    def userid(self):
        if self._bot_user is None:
            raise ValueError("get_me() must be called before getting username")
        return self._bot_user["id"]

    @property
    def username(self):
        if self._bot_user is None:
            raise ValueError("get_me() must be called before getting username")
        return self._bot_user["username"]
    
    async def skip_updates(self):
        updates = await self.get_updates(0)
        while updates:
            self._offset = updates[-1]["update_id"]
            updates = self.get_updates(0)

    async def get_updates(self, timeout=60):
        data = {}
        if self._offset is not None:
            data["offset"] = self._offset + 1
        if timeout > 0:
            data["timeout"] = timeout
        updates = await self.request("getUpdates", data, timeout=timeout + 10)
        if updates:
            self._offset = updates[-1]["update_id"]
        return updates

    async def get_file(self, path):
        return await self.request(path, raw=True, prefix="file/")

    async def request(self, method, data=None, *, prefix="", raw=False, timeout=10):
        try:
            apiurl = "https://api.telegram.org/{}bot{}/{}".format(prefix, self._token, method)

            if data is not None:
                data = json.dumps(data).encode("utf-8")
                request = self._session.post(apiurl, data=data, headers={"Content-Type": "application/json"}, timeout=aiohttp.ClientTimeout(total=timeout))
            else:
                request = self._session.get(apiurl)

            async with request as response:
                if raw:
                    return await response.read()
                result = await response.json()
                if not result.get("ok") or "result" not in result:
                    raise ApiError(result.get("description", "unknown error from API")) from None
                return result["result"]
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as e:
            raise ApiError("loading failed: {}: {}".format(type(e).__name__, e)) from None
    
    async def __aenter__(self):
        LOGGER.info("Initializing bot")
        await self._session.__aenter__()
        await self.get_me()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self._session.__aexit__(exc_type, exc_value, traceback)

    def __aiter__(self):
        return AsyncUpdateIterator(self)

class AsyncUpdateIterator:
    def __init__(self, bot):
        self._bot = bot
        self._updates = iter([])

    def __aiter__(self):
        return self
    
    async def __anext__(self):
        while True:
            try:
                return next(self._updates)
            except StopIteration:
                LOGGER.debug("Fetching more updates")
                self._updates = iter(await self._bot.get_updates())

class ApiError(Exception):
    pass
