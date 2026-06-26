from __future__ import annotations

import argparse
import struct
from pathlib import Path


ENDOFCHAIN = 0xFFFFFFFE
FREESECT = 0xFFFFFFFF


class Ole:
    def __init__(self, path: Path):
        self.data = path.read_bytes()
        if self.data[:8] != bytes.fromhex("D0 CF 11 E0 A1 B1 1A E1"):
            raise ValueError("not an OLE compound file")
        self.sector_size = 1 << struct.unpack_from("<H", self.data, 0x1E)[0]
        self.mini_sector_size = 1 << struct.unpack_from("<H", self.data, 0x20)[0]
        self.first_dir_sector = struct.unpack_from("<I", self.data, 0x30)[0]
        self.mini_cutoff = struct.unpack_from("<I", self.data, 0x38)[0]
        first_mini_fat = struct.unpack_from("<I", self.data, 0x3C)[0]
        mini_fat_count = struct.unpack_from("<I", self.data, 0x40)[0]
        difat = list(struct.unpack_from("<109I", self.data, 0x4C))
        self.fat = self._read_fat([s for s in difat if s not in (FREESECT, ENDOFCHAIN)])
        self.dir_entries = self._read_dir()
        self.root = next(e for e in self.dir_entries if e["type"] == 5)
        self.mini_fat = []
        if mini_fat_count and first_mini_fat not in (FREESECT, ENDOFCHAIN):
            mini_fat_bytes = self._read_chain_bytes(first_mini_fat)
            self.mini_fat = list(struct.unpack("<" + "I" * (len(mini_fat_bytes) // 4), mini_fat_bytes))
        self.mini_stream = b""
        if self.root["start"] not in (FREESECT, ENDOFCHAIN):
            self.mini_stream = self._read_chain_bytes(self.root["start"])[: self.root["size"]]

    def _sector_offset(self, sector: int) -> int:
        return 512 + sector * self.sector_size

    def _sector(self, sector: int) -> bytes:
        off = self._sector_offset(sector)
        return self.data[off : off + self.sector_size]

    def _read_fat(self, sectors: list[int]) -> list[int]:
        buf = b"".join(self._sector(s) for s in sectors)
        return list(struct.unpack("<" + "I" * (len(buf) // 4), buf))

    def _chain(self, start: int, fat: list[int] | None = None) -> list[int]:
        fat = self.fat if fat is None else fat
        out = []
        cur = start
        seen = set()
        while cur not in (FREESECT, ENDOFCHAIN) and cur < len(fat) and cur not in seen:
            seen.add(cur)
            out.append(cur)
            cur = fat[cur]
        return out

    def _read_chain_bytes(self, start: int) -> bytes:
        return b"".join(self._sector(s) for s in self._chain(start))

    def _read_mini_chain_bytes(self, start: int, size: int) -> bytes:
        chunks = []
        for s in self._chain(start, self.mini_fat):
            off = s * self.mini_sector_size
            chunks.append(self.mini_stream[off : off + self.mini_sector_size])
        return b"".join(chunks)[:size]

    def _read_dir(self) -> list[dict]:
        buf = self._read_chain_bytes(self.first_dir_sector)
        entries = []
        for off in range(0, len(buf), 128):
            raw = buf[off : off + 128]
            if len(raw) < 128:
                continue
            name_len = struct.unpack_from("<H", raw, 64)[0]
            if name_len >= 2:
                name = raw[: name_len - 2].decode("utf-16le", "ignore")
            else:
                name = ""
            entries.append(
                {
                    "name": name,
                    "type": raw[66],
                    "start": struct.unpack_from("<I", raw, 116)[0],
                    "size": struct.unpack_from("<Q", raw, 120)[0],
                }
            )
        return entries

    def stream(self, name: str) -> bytes:
        ent = next(e for e in self.dir_entries if e["name"] == name)
        if ent["size"] < self.mini_cutoff and ent["type"] == 2:
            return self._read_mini_chain_bytes(ent["start"], ent["size"])
        return self._read_chain_bytes(ent["start"])[: ent["size"]]


def likely_piece_table(table: bytes):
    for i, b in enumerate(table[:-5]):
        if b != 2:
            continue
        lcb = struct.unpack_from("<I", table, i + 1)[0]
        if lcb < 16 or lcb > len(table) - i - 5 or (lcb - 4) % 12:
            continue
        n = (lcb - 4) // 12
        if n <= 0 or n > 500:
            continue
        start = i + 5
        cps = list(struct.unpack_from("<" + "I" * (n + 1), table, start))
        if cps[0] != 0 or any(cps[j] > cps[j + 1] for j in range(n)) or cps[-1] > 300000:
            continue
        yield start, n, cps


def extract(path: Path) -> str:
    ole = Ole(path)
    word = ole.stream("WordDocument")
    tables = []
    for name in ("0Table", "1Table"):
        try:
            tables.append(ole.stream(name))
        except StopIteration:
            pass
    best = ""
    for table in tables:
        for start, n, cps in likely_piece_table(table):
            pcd_start = start + 4 * (n + 1)
            parts = []
            for idx in range(n):
                pcd = table[pcd_start + idx * 8 : pcd_start + (idx + 1) * 8]
                raw_fc = struct.unpack_from("<I", pcd, 2)[0]
                compressed = bool(raw_fc & 0x40000000)
                char_count = cps[idx + 1] - cps[idx]
                if not char_count:
                    continue
                if compressed:
                    fc = (raw_fc & 0x3FFFFFFF) // 2
                    chunk = word[fc : fc + char_count]
                    text = chunk.decode("cp1252", "ignore")
                else:
                    fc = raw_fc & 0x3FFFFFFF
                    chunk = word[fc : fc + char_count * 2]
                    text = chunk.decode("utf-16le", "ignore")
                parts.append(text)
            text = "".join(parts)
            score = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
            if score > sum(1 for ch in best if "\u4e00" <= ch <= "\u9fff"):
                best = text
    return best


def clean(text: str) -> str:
    text = text.replace("\r", "\n")
    text = "".join(ch if ch == "\n" or ch == "\t" or ord(ch) >= 32 else " " for ch in text)
    lines = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("doc", type=Path)
    args = parser.parse_args()
    print(clean(extract(args.doc)))


if __name__ == "__main__":
    main()
