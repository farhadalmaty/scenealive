"""A small RFC 6455 WebSocket (server and client side) on asyncio streams, standard library + numpy only."""
from __future__ import annotations
import asyncio
import base64
import hashlib
import os
import struct
import numpy as np

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class ProtocolError(Exception):
    def __init__(self, msg: str, code: int = 1002): super().__init__(msg); self.code = code


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()


def apply_mask(data: bytes, key: bytes) -> bytes:
    if not data: return b""
    a = np.frombuffer(data, np.uint8); k = np.frombuffer(key, np.uint8)
    return (a ^ np.resize(k, a.size)).tobytes()


def encode_frame(opcode: int, payload: bytes = b"", fin: bool = True, mask: bool = False) -> bytes:
    n = len(payload); b0 = (0x80 if fin else 0) | opcode; m = 0x80 if mask else 0
    if n < 126: head = struct.pack("!BB", b0, m | n)
    elif n < 65536: head = struct.pack("!BBH", b0, m | 126, n)
    else: head = struct.pack("!BBQ", b0, m | 127, n)
    if mask:
        key = os.urandom(4); return head + key + apply_mask(payload, key)
    return head + payload


class WebSocket:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, max_message: int = 2_000_000, is_client: bool = False):
        self.r, self.w, self.max_message, self.is_client = reader, writer, max_message, is_client
        self._lock = asyncio.Lock(); self.closed = False

    async def _read_frame(self):
        b0, b1 = await self.r.readexactly(2)
        fin, rsv, op = bool(b0 & 0x80), b0 & 0x70, b0 & 0x0F; masked, n = bool(b1 & 0x80), b1 & 0x7F
        if rsv: raise ProtocolError("reserved bits set")
        if n == 126: n = struct.unpack("!H", await self.r.readexactly(2))[0]
        elif n == 127: n = struct.unpack("!Q", await self.r.readexactly(8))[0]
        if op >= 0x8 and (n > 125 or not fin): raise ProtocolError("bad control frame")
        if not self.is_client and not masked: raise ProtocolError("client frames must be masked")
        if n > self.max_message: raise ProtocolError("message too big", 1009)
        key = await self.r.readexactly(4) if masked else b""
        data = await self.r.readexactly(n) if n else b""
        return fin, op, (apply_mask(data, key) if masked else data)

    async def recv(self):
        """-> (OP_TEXT | OP_BIN, bytes), or None when the connection is closed. Raises ProtocolError on a broken peer."""
        buf = bytearray(); first = None
        while True:
            try: fin, op, data = await self._read_frame()
            except (asyncio.IncompleteReadError, ConnectionError): self.closed = True; return None
            except ProtocolError as ex: await self._fail(ex.code); raise
            if op == OP_PING: await self._send(OP_PONG, data); continue
            if op == OP_PONG: continue
            if op == OP_CLOSE:
                if not self.closed: await self._send(OP_CLOSE, data[:2]); self.closed = True
                return None
            if op in (OP_TEXT, OP_BIN):
                if first is not None: await self._fail(1002); raise ProtocolError("new message inside a fragmented one")
                first = op
            elif op == OP_CONT:
                if first is None: await self._fail(1002); raise ProtocolError("continuation without a start")
            else: await self._fail(1002); raise ProtocolError(f"unknown opcode {op}")
            buf += data
            if len(buf) > self.max_message: await self._fail(1009); raise ProtocolError("message too big", 1009)
            if fin:
                if first == OP_TEXT:
                    try: return OP_TEXT, bytes(buf).decode("utf-8").encode("utf-8")
                    except UnicodeDecodeError: await self._fail(1007); raise ProtocolError("invalid UTF-8", 1007)
                return first, bytes(buf)

    async def _send(self, op: int, data: bytes = b"") -> None:
        if self.closed and op != OP_CLOSE: return
        async with self._lock:
            try: self.w.write(encode_frame(op, data, mask=self.is_client)); await self.w.drain()
            except (ConnectionError, RuntimeError): self.closed = True

    async def send_text(self, s: str) -> None: await self._send(OP_TEXT, s.encode("utf-8"))
    async def send_binary(self, b: bytes) -> None: await self._send(OP_BIN, b)

    async def _fail(self, code: int) -> None:
        await self.close(code)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self.closed: return
        self.closed = True
        try:
            async with self._lock:
                self.w.write(encode_frame(OP_CLOSE, struct.pack("!H", code) + reason.encode()[:100], mask=self.is_client)); await self.w.drain()
        except (ConnectionError, RuntimeError): pass
