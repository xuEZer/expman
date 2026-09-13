"""Framed, trusted local IPC between a scheduler and its worker."""

import pickle
import socket
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Channel:
    """A non-blocking framed pickle channel over a private socketpair."""

    socket: socket.socket
    buffer: bytes = field(default=b"", repr=False)
    _send_lock: Lock = field(default_factory=Lock, repr=False)

    def send(self, message) -> None:
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        with self._send_lock:
            self.socket.sendall(len(payload).to_bytes(8, "big") + payload)

    def drain(self) -> list:
        messages = []
        while True:
            try:
                data = self.socket.recv(1024 * 1024)
            except BlockingIOError:
                break
            if not data:
                break
            self.buffer += data
        while len(self.buffer) >= 8:
            length = int.from_bytes(self.buffer[:8], "big")
            if len(self.buffer) < length + 8:
                break
            payload, self.buffer = (
                self.buffer[8 : length + 8],
                self.buffer[length + 8 :],
            )
            messages.append(pickle.loads(payload))
        return messages


def worker_channel(fd: str) -> Channel:
    """Open the inherited worker endpoint in blocking send mode."""
    return Channel(socket.socket(fileno=int(fd)))
