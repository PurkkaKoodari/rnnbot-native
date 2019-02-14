import subprocess
import threading
import collections
import struct
import os
from signal import SIGINT
from typing import Optional

from telegram import telegram

import config

assert config.TOKEN

bot = telegram.Bot(config.TOKEN)
bot.confirmToken()

class Action:
    SAMPLE = object()
    GET_ITER = object()
    QUIT = object()
    COMMIT = object()
    def __init__(self, type):
        self.type = type
        self._done = False
        self._result = None
        self._cond = threading.Condition()
    def complete(self, result=None):
        with self._cond:
            self._result = result
            self._done = True
            self._cond.notify_all()
    def wait(self):
        if not self._done:
            with self._cond:
                self._cond.wait_for(lambda: self._done)
        return self._result

class RNNThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self._new_messages = collections.deque(maxlen=config.COMMIT_MESSAGES)
        self._new_messages.append("ebin juttu hermanni " * 100) # TODO remove
        self._action_queue = collections.deque([Action(Action.COMMIT)])
        self._msg_lock = threading.Lock()
        self._queue_cond = threading.Condition()
        self._action_cond = threading.Condition()
    def run(self):
        process = None # type: Optional[subprocess.Popen]
        while True:
            with self._queue_cond:
                self._queue_cond.wait_for(lambda: self._action_queue)
                action = self._action_queue.popleft() # type: Action

            if action.type is Action.COMMIT:
                if process is not None: # TODO replace with ./rnn resume
                    process.send_signal(SIGINT)
                    process.wait()
                process = subprocess.Popen(["./rnn"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                with self._msg_lock:
                    data = "\n".join(self._new_messages)
                    self._new_messages.clear()
                process.stdin.write(data.encode() + b"\0")
                process.stdin.flush()
                action.complete()

            elif action.type is Action.SAMPLE:
                process.stdin.write(b"s")
                process.stdin.flush()
                response = b""
                while True:
                    ch = process.stdout.read(1)
                    if not ch or ch == b"\0":
                        break
                    response += ch
                action.complete(response)

            elif action.type is Action.GET_ITER:
                process.stdin.write(b"i")
                process.stdin.flush()
                response = b""
                while len(response) < 16:
                    buf = process.stdout.read(16 - len(response))
                    if not buf:
                        break
                    response += buf
                action.complete(struct.unpack("Ld", response))

            elif action.type is Action.QUIT:
                process.stdin.write(b"q")
                process.stdin.flush()
                process.stdin.close()
                dump = process.stdout.read()
                process.wait()
                with open("rnnbot-state.dat", "wb") as stream:
                    stream.write(dump)
                action.complete()
                return
            
            else:
                print("unknown action")
                action.complete()

    def put_message(self, message):
        with self._msg_lock:
            self._new_messages.append(message)
            
    def action(self, type):
        action = Action(type)
        with self._queue_cond:
            self._action_queue.append(action)
            self._queue_cond.notify_all()
        return action.wait()

rnn = RNNThread()
rnn.start()

while 1:
    try:
        cmd = input(">")
        if cmd[0] == "/":
            action = getattr(Action, cmd[1:].upper())
            result = rnn.action(action)
            print(result)
            if action is Action.QUIT:
                break
        else:
            rnn.put_message(cmd)
    except Exception as ex:
        print(ex)
