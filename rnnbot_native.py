import asyncio
import aiofiles
import collections
import enum
import logging
import os
import random
import shutil
import signal
import struct
import sys
import time

from typing import Optional

import asynctg
import config

LOGGER = logging.getLogger("RNNBot")

NEW_SAVE_MAGIC = b"RNNSaved"

DUMP_SANITY = b"RNNState"
SAMPLE_SANITY = b"RNNSampl"
ITER_SANITY = b"RNNIterI"

ActionType = enum.Enum("ActionType", "SAMPLE GET_ITER COMMIT")

def utc_timestamp():
    return int(time.time())

def format_time(seconds):
    if seconds is None:
        return "N/A"
    return "{:d}:{:02d}:{:02d}".format(seconds // 3600, seconds // 60 % 60, seconds % 60)

class RNNBot:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.messages = collections.deque(maxlen=config.COMMIT_MESSAGES)
        self.last_commit_attempt = 0
        self.last_real_commit = 0
        self.messages_in_commit = 0
    
    def next_commit(self, after):
        return after + config.COMMIT_INTERVAL - (after - config.COMMIT_EPOCH) % config.COMMIT_INTERVAL

    async def run_rnn(self):
        datadir = os.path.dirname(os.path.abspath(__file__))
        datafile = os.path.join(datadir, "rnn-state.dat")
        process = None

        if os.path.exists(datafile):
            LOGGER.info("Loading saved data")
            async with aiofiles.open(datafile, "rb") as stream:
                save_data = await stream.read()

            if not save_data.startswith(NEW_SAVE_MAGIC):
                LOGGER.info("Data file version: pre-versioning")
                self.last_commit_attempt, self.last_real_commit, num_messages = struct.unpack_from("=QQI", save_data, 0)
                self.messages_in_commit = 0
                offset = 20
            else:
                save_version, = struct.unpack_from("=Q", save_data, 8)
                LOGGER.info("Data file version: {}".format(save_version))
                if save_version == 1:
                    self.last_commit_attempt, self.last_real_commit, self.messages_in_commit, num_messages = struct.unpack_from("=QQQI", save_data, 16)
                    offset = 44
                else:
                    raise ValueError("unknown save file version")

            LOGGER.info("Last attempted commit: {}".format(time.strftime("%c", time.localtime(self.last_commit_attempt))))
            LOGGER.info("Last successful commit: {}".format(time.strftime("%c", time.localtime(self.last_real_commit))))
            LOGGER.info("Messages in commit: {}".format(self.messages_in_commit))

            LOGGER.info("Loading {} messages".format(num_messages))
            for _ in range(num_messages):
                separator = save_data.index(b"\0", offset)
                self.messages.append(save_data[offset:separator].decode("utf-8"))
                offset = separator + 1

            if offset != len(save_data):
                LOGGER.info("Loading RNN state")
                process = await asyncio.create_subprocess_exec("./rnn", "--resume", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
                process.stdin.write(save_data[offset:])
                await process.stdin.drain()
            
            LOGGER.info("Saved data loaded")
        
        commit_task = asyncio.ensure_future(self.run_commits())

        try:
            while True:
                action, future = await self.queue.get()

                try:
                    if action == ActionType.COMMIT:
                        now = utc_timestamp()
                        self.last_commit_attempt = now
                        if sum(map(len, self.messages)) < config.COMMIT_MIN_LENGTH:
                            LOGGER.info("Skipping commit because too little data has been received")
                            continue
                        messages_committed = len(self.messages)
                        LOGGER.info("Committing {} messages".format(messages_committed))
                        data = struct.pack("=Q", config.HIDDEN_SIZE) + "\n".join(self.messages).encode() + b"\0"
                        self.messages.clear()
                        # kill old rnn process
                        if process is not None:
                            try:
                                LOGGER.info("Saving final state of commit")
                                process.stdin.write(b"q")
                                process.stdin.write_eof()
                                save_data = await process.stdout.read()
                                if not save_data.endswith(DUMP_SANITY):
                                    LOGGER.warn("data ended with {} instead of {}".format(save_data[-8:], DUMP_SANITY))
                                await process.wait()
                                statefile = os.path.join(datadir, "endstates/{}.dat".format(utc_timestamp()))
                                async with aiofiles.open(statefile, "wb") as stream:
                                    await stream.write(save_data)
                            except asyncio.CancelledError:
                                raise
                            except:
                                LOGGER.error("Failed to save final state", exc_info=True)
                                process.terminate()
                                await process.wait()
                        # create new process and initialize it with message data
                        process = await asyncio.create_subprocess_exec("./rnn", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
                        process.stdin.write(data)
                        await process.stdin.drain()
                        future.set_result(None)
                        self.last_real_commit = now
                        self.messages_in_commit = messages_committed

                    elif action == ActionType.SAMPLE:
                        if process is None:
                            future.set_result(None)
                            continue
                        process.stdin.write(b"s")

                        result = (await process.stdout.readuntil(b"\0"))[:-1]
                        sanity = await process.stdout.readexactly(8)
                        assert sanity == SAMPLE_SANITY, "data ended with {} instead of {}".format(sanity, SAMPLE_SANITY)
                        future.set_result(result.decode("utf-8"))
                    
                    elif action == ActionType.GET_ITER:
                        if process is None:
                            future.set_result(None)
                            continue
                        process.stdin.write(b"i")
                        result = await process.stdout.readexactly(16)
                        sanity = await process.stdout.readexactly(8)
                        assert sanity == ITER_SANITY, "data ended with {} instead of {}".format(sanity, ITER_SANITY)
                        future.set_result(struct.unpack("=Qd", result))
                    
                    else:
                        future.set_exception(ValueError("bad action type"))

                except asyncio.CancelledError:
                    raise
                except Exception as ex:
                    future.set_exception(ex)
            
        except asyncio.CancelledError:
            commit_task.cancel()
            try:
                if os.path.exists(datafile):
                    LOGGER.info("Backing up old data file")
                    shutil.copyfile(datafile, datafile + ".old")
            except:
                LOGGER.error("Failed to backup data file", exc_info=True)
            try:
                LOGGER.info("Saving {} messages".format(len(self.messages)))
                save_data = NEW_SAVE_MAGIC + struct.pack("=QQQQI", 1, self.last_commit_attempt, self.last_real_commit, self.messages_in_commit, len(self.messages))
                for message in self.messages:
                    save_data += message.encode("utf-8") + b"\0"
                if process is not None:
                    LOGGER.info("Saving RNN state")
                    process.stdin.write(b"q")
                    process.stdin.write_eof()
                    save_data += await process.stdout.read()
                    assert save_data.endswith(DUMP_SANITY), "data ended with {} instead of {}".format(save_data[-8:], DUMP_SANITY)
                    await process.wait()
                async with aiofiles.open(datafile, "wb") as stream:
                    await stream.write(save_data)
                LOGGER.info("Saved data")
            except asyncio.CancelledError:
                raise
            except:
                LOGGER.error("Failed to save state", exc_info=True)
        
    async def do_action(self, action):
        future = asyncio.get_event_loop().create_future()
        self.queue.put_nowait((action, future))
        return await future

    async def run_commits(self):
        startup = utc_timestamp()
        while True:
            if self.last_commit_attempt is None:
                # not committed yet, commit at first commit point
                next_commit = self.next_commit(startup)
            else:
                # committed previously, commit if interval elapsed
                next_commit = self.next_commit(self.last_commit_attempt)
            now = utc_timestamp()
            if next_commit > now:
                await asyncio.sleep(max(1, min(60, 0.9 * (next_commit - now))))
                continue
            await self.do_action(ActionType.COMMIT)

    async def run_bot(self, bot):
        while True:
            try:
                async for update in bot:
                    try:
                        if "message" not in update:
                            continue
                        message = update["message"]
                        if message["chat"]["id"] not in config.GROUPS or "text" not in message:
                            continue
                        text = message["text"]
                        assert text and "\0" not in text

                        if text[0] == "/":
                            command = text.split(None, 1)[0]
                            if command.endswith("@" + bot.username):
                                command = command[:-len("@" + bot.username)]

                            if command == "/rnn":
                                response = await self.do_action(ActionType.SAMPLE)
                                await bot.request("sendMessage", {
                                    "chat_id": message["chat"]["id"],
                                    "text": config.NO_DATA if response is None else response,
                                })
                            elif command == "/rnniter":
                                iteration = await self.do_action(ActionType.GET_ITER)
                                now = utc_timestamp()
                                if iteration is None:
                                    response = config.ITER_NO_DATA.format(
                                        to_next=format_time(self.next_commit(now) - now),
                                        in_next=len(self.messages),
                                    )
                                else:
                                    response = config.ITER_MESSAGE.format(
                                        iteration=iteration[0],
                                        loss=iteration[1],
                                        from_last=format_time(now - self.last_real_commit if self.last_real_commit else None),
                                        in_last=self.messages_in_commit,
                                        to_next=format_time(self.next_commit(now) - now),
                                        in_next=len(self.messages),
                                    )
                                await bot.request("sendMessage", {
                                    "chat_id": message["chat"]["id"],
                                    "text": response,
                                    "parse_mode": "html",
                                })
                            elif command == "/rnncommit" and "from" in message and message["from"]["id"] in config.ADMINS:
                                await self.do_action(ActionType.COMMIT)
                        
                        else:
                            if len(text) < config.MESSAGE_MIN_LENGTH:
                                continue
                            if len(text.split()) < config.MESSAGE_MIN_WORDS:
                                continue
                            if random.random() < config.NAME_CHANCE and "from" in message:
                                name = message["from"]["first_name"]
                                if "last_name" in message["from"]:
                                    name += " " + message["from"]["last_name"]
                                text = name + ": " + text
                            self.messages.append(text)

                    except asyncio.CancelledError:
                        raise
                    except:
                        LOGGER.error("Error handling update", exc_info=True)
            except asynctg.ApiError:
                LOGGER.error("Error fetching updates", exc_info=True)
    
    async def run(self):
        async with asynctg.Bot(config.TOKEN) as bot:
            bot_task = asyncio.ensure_future(self.run_bot(bot))
            rnn_task = asyncio.ensure_future(self.run_rnn())

            def quit():
                LOGGER.info("Quitting")
                rnn_task.cancel()
                bot_task.cancel()
            asyncio.get_event_loop().add_signal_handler(signal.SIGINT, quit)
            asyncio.get_event_loop().add_signal_handler(signal.SIGTERM, quit)

            done, _ = await asyncio.wait([bot_task, rnn_task], return_when=asyncio.FIRST_EXCEPTION)
            quit()
            # raise errors from tasks before quitting
            for task in done:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await asyncio.wait([bot_task, rnn_task])

if __name__ == "__main__":
    formatter = logging.Formatter(config.LOG_FORMAT)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    handler = logging.FileHandler("rnn.log")
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    asyncio.get_event_loop().run_until_complete(RNNBot().run())
