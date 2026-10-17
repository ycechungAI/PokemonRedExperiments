import asyncio
import json
import queue
import threading
import warnings

import gymnasium as gym
import websockets
from websockets.exceptions import WebSocketException

X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
MAP_N_ADDRESS = 0xD35E
DEFAULT_STREAM_METADATA = {
    "user": "pwhiddy jippity-6",
    "env_id": 0,
    "color": "#cc5522",
    "extra": "",
}


class CoordinateStreamer:
    """Batch coordinate changes and broadcast without blocking emulator ticks."""

    def __init__(self, stream_metadata=None, upload_interval=60):
        if type(upload_interval) is not int or upload_interval < 1:
            raise ValueError("upload_interval must be a positive integer")
        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.stream_metadata = {**DEFAULT_STREAM_METADATA, **(stream_metadata or {})}
        self.upload_interval = upload_interval
        self.coord_list = []
        self._last_coords = None
        self._frames = 0
        self._queue = queue.SimpleQueue()
        self._closed = False
        self._abort = threading.Event()
        self._warned = False
        self.websocket = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def record(self, coords):
        if self._closed:
            raise RuntimeError("Coordinate streamer is closed")
        coords = tuple(coords)
        if coords != self._last_coords:
            self.coord_list.append(list(coords))
            self._last_coords = coords
        self._frames += 1
        if self._frames >= self.upload_interval:
            self.flush()

    def flush(self):
        self._frames = 0
        if not self.coord_list:
            return
        metadata = dict(self.stream_metadata)
        self._queue.put({"metadata": metadata, "coords": self.coord_list})
        self.coord_list = []

    def close(self):
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            self._abort.set()
            warnings.warn("Coordinate stream did not finish uploading before shutdown")

    def _run(self):
        asyncio.run(self._broadcast_queue())

    async def _broadcast_queue(self):
        try:
            while not self._abort.is_set():
                try:
                    payload = self._queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.02)
                    continue
                if payload is None:
                    break
                await self.broadcast_ws_message(json.dumps(payload))
        finally:
            await self._disconnect()

    async def _disconnect(self):
        if self.websocket is not None:
            try:
                await asyncio.wait_for(self.websocket.close(), timeout=1)
            except (OSError, TimeoutError, WebSocketException):
                pass
            self.websocket = None

    async def broadcast_ws_message(self, message):
        try:
            if self.websocket is None:
                self.websocket = await websockets.connect(
                    self.ws_address, open_timeout=2, close_timeout=1,
                )
            await asyncio.wait_for(self.websocket.send(message), timeout=2)
            self._warned = False
        except (OSError, TimeoutError, WebSocketException) as error:
            await self._disconnect()
            if not self._warned:
                warnings.warn(f"Coordinate stream unavailable: {error}")
                self._warned = True


class StreamWrapper(gym.Wrapper):
    def __init__(self, env, stream_metadata=None):
        super().__init__(env)
        if hasattr(env, "pyboy"):
            self.emulator = env.pyboy
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise ValueError("Could not find emulator!")
        self.streamer = CoordinateStreamer(stream_metadata, upload_interval=300)
        self.stream_metadata = self.streamer.stream_metadata

    def step(self, action):
        result = self.env.step(action)
        x_pos = self.emulator.memory[X_POS_ADDRESS]
        y_pos = self.emulator.memory[Y_POS_ADDRESS]
        map_n = self.emulator.memory[MAP_N_ADDRESS]
        self.streamer.record((x_pos, y_pos, map_n))
        return result

    def close(self):
        try:
            self.streamer.close()
        finally:
            self.env.close()
