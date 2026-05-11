"""Generic utilities for Modern BioJazz."""
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from typing import Any, Dict


def validate_ip_is_external(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address, ip_str: str) -> None:
    """Validate that an IP address is not internal or reserved."""
    if (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
    ):
        raise ValueError(f"URL resolves to internal/reserved IP address: {ip_str}")


def save_json_snapshot(data: Dict[str, Any], path: str | Path) -> None:
    """Save a snapshot dictionary to JSON for offline/reproducible use."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
