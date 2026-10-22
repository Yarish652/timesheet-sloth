"""Refuse any network connection that isn't to this machine.

Installed by the CLI before anything touches your data, so a library that tries to phone home
(telemetry, a model re-download, a token check) fails loudly instead of quietly leaking.
TabPFN only needs the network once, for `sloth setup`, which runs without the guard.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any

_LOCAL_NAMES = {"localhost", "localhost.localdomain", ""}
_original_connect = socket.socket.connect
_original_getaddrinfo = socket.getaddrinfo


class NonLocalNetworkError(ConnectionRefusedError):
    pass


def _is_local(host: Any) -> bool:
    if not isinstance(host, str) or host in _LOCAL_NAMES:
        return True  # AF_UNIX paths and the like
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _guarded_connect(self: socket.socket, address: Any) -> None:
    host = address[0] if isinstance(address, tuple) else address
    if not _is_local(host):
        raise NonLocalNetworkError(f"sloth netguard blocked a connection to {host!r}")
    _original_connect(self, address)


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(host, bytes):
        host = host.decode()
    if not _is_local(host):
        raise NonLocalNetworkError(f"sloth netguard blocked a DNS lookup of {host!r}")
    return _original_getaddrinfo(host, *args, **kwargs)


def install() -> None:
    """Block non-loopback connections for the rest of this process."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]


def uninstall() -> None:
    socket.socket.connect = _original_connect  # type: ignore[method-assign]
    socket.getaddrinfo = _original_getaddrinfo
