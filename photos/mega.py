"""Minimal client for public MEGA links (mega.nz).

MEGA encrypts everything client-side: the key after '#' in a share link decrypts
the file keys, file names and file contents. This module talks to MEGA's public
API, decrypts the folder listing and streams decrypted file contents.

Supported link formats:
    https://mega.nz/folder/<id>#<key>          (and /folder/<id>#<key>/folder/<sub>)
    https://mega.nz/file/<id>#<key>
    https://mega.nz/#F!<id>!<key>              (legacy folder)
    https://mega.nz/#!<id>!<key>               (legacy file)
"""
import base64
import json
import re
import struct
from dataclasses import dataclass
from typing import Iterator, List

import requests
from Crypto.Cipher import AES
from Crypto.Util import Counter

API_URL = "https://g.api.mega.co.nz/cs"
TIMEOUT = 30

# Error codes returned by the MEGA API as a bare negative number.
API_ERRORS = {
    -2: "invalid link",
    -9: "the file or folder does not exist (it may have been deleted)",
    -11: "access denied",
    -16: "the file or folder is blocked",
    -17: "MEGA transfer quota exceeded, try again later",
    -18: "MEGA is temporarily unavailable, try again later",
}


class MegaError(Exception):
    pass


@dataclass
class MegaFile:
    handle: str
    name: str
    size: int
    key: bytes  # 16-byte AES key for contents and attributes
    iv: bytes  # 8-byte CTR nonce
    folder_id: str = ""  # set for files inside a shared folder


@dataclass
class MegaLink:
    kind: str  # "folder" or "file"
    id: str
    key: str


def parse_link(url: str) -> MegaLink:
    patterns = [
        (r"mega\.(?:nz|co\.nz)/folder/([\w-]+)#([\w-]+)", "folder"),
        (r"mega\.(?:nz|co\.nz)/file/([\w-]+)#([\w-]+)", "file"),
        (r"mega\.(?:nz|co\.nz)/#F!([\w-]+)!([\w-]+)", "folder"),
        (r"mega\.(?:nz|co\.nz)/#!([\w-]+)!([\w-]+)", "file"),
    ]
    for pattern, kind in patterns:
        match = re.search(pattern, url)
        if match:
            return MegaLink(kind=kind, id=match.group(1), key=match.group(2))
    raise MegaError("Unrecognised MEGA link. It must include the key after '#'.")


def list_files(link: MegaLink) -> List[MegaFile]:
    """Return every file in a shared folder (including subfolders), or the single shared file."""
    if link.kind == "file":
        return [_single_file(link)]

    folder_key = _b64decode(link.key)
    if len(folder_key) != 16:
        raise MegaError("Invalid MEGA folder key.")
    response = _api({"a": "f", "c": 1, "r": 1, "ca": 1}, folder_id=link.id)

    files = []
    for node in response.get("f", []):
        if node.get("t") != 0 or ":" not in node.get("k", ""):
            continue  # folders, or nodes without a key we can use
        encrypted_key = _b64decode(node["k"].split(":", 1)[1])
        file_key = _aes_ecb_decrypt(folder_key, encrypted_key)
        if len(file_key) != 32:
            continue
        key, iv = _split_file_key(file_key)
        name = _decrypt_name(node.get("a", ""), key) or node["h"]
        files.append(MegaFile(handle=node["h"], name=name, size=node.get("s", 0), key=key, iv=iv, folder_id=link.id))
    return files


def iter_content(session: requests.Session, file: MegaFile, chunk_size: int) -> Iterator[bytes]:
    """Download a file and yield its decrypted contents chunk by chunk."""
    if file.folder_id:
        info = _api({"a": "g", "g": 1, "n": file.handle}, folder_id=file.folder_id)
    else:
        info = _api({"a": "g", "g": 1, "p": file.handle})
    if "g" not in info:
        raise MegaError("MEGA did not return a download URL.")

    counter = Counter.new(64, prefix=file.iv, initial_value=0)
    cipher = AES.new(file.key, AES.MODE_CTR, counter=counter)
    with session.get(info["g"], stream=True, timeout=TIMEOUT) as response:
        if response.status_code != 200:
            raise MegaError(f"MEGA download failed with HTTP {response.status_code}.")
        for chunk in response.iter_content(chunk_size):
            yield cipher.decrypt(chunk)


# --------------------------------------------------------------------------- internals


def _single_file(link: MegaLink) -> MegaFile:
    file_key = _b64decode(link.key)
    if len(file_key) != 32:
        raise MegaError("Invalid MEGA file key.")
    key, iv = _split_file_key(file_key)
    info = _api({"a": "g", "p": link.id})
    name = _decrypt_name(info.get("at", ""), key) or link.id
    return MegaFile(handle=link.id, name=name, size=info.get("s", 0), key=key, iv=iv)


def _api(command: dict, folder_id: str = "") -> dict:
    params = {"id": 0}
    if folder_id:
        params["n"] = folder_id
    try:
        response = requests.post(API_URL, params=params, data=json.dumps([command]), timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise MegaError(f"Could not reach MEGA: {exc}")

    if isinstance(result, int):
        result = [result]
    result = result[0]
    if isinstance(result, int):
        raise MegaError(f"MEGA error: {API_ERRORS.get(result, f'code {result}')}.")
    return result


def _split_file_key(file_key: bytes):
    """A 32-byte file key packs the AES key (XOR of both halves) and the CTR nonce."""
    k = struct.unpack(">8I", file_key)
    key = struct.pack(">4I", k[0] ^ k[4], k[1] ^ k[5], k[2] ^ k[6], k[3] ^ k[7])
    iv = struct.pack(">2I", k[4], k[5])
    return key, iv


def _decrypt_name(encrypted_attributes: str, key: bytes) -> str:
    if not encrypted_attributes:
        return ""
    data = _b64decode(encrypted_attributes)
    data += b"\0" * (-len(data) % 16)
    plain = AES.new(key, AES.MODE_CBC, iv=b"\0" * 16).decrypt(data).rstrip(b"\0")
    if not plain.startswith(b"MEGA"):
        return ""  # wrong key
    try:
        return json.loads(plain[4:].decode("utf-8", "ignore")).get("n", "")
    except ValueError:
        return ""


def _aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    data += b"\0" * (-len(data) % 16)
    return AES.new(key, AES.MODE_ECB).decrypt(data)


def _b64decode(value: str) -> bytes:
    value = value.replace("-", "+").replace("_", "/").replace(",", "")
    return base64.b64decode(value + "=" * (-len(value) % 4))
