"""Real drone footage as a probe test set: does the simulator-trained perception transfer?

Evaluation only. UAV123 and VisDrone-SOT are research / non-commercial datasets; nothing here
trains on them, and only code and score summaries go into git. The footage is downloaded and
converted inside Modal CPU jobs straight onto the laya-datasets volume (modal_laya.py
realtest_* functions), never onto the local disk:

    modal run modal_laya.py::realtest_build         # download + convert -> /data/realtest/<set>/
    modal run modal_laya.py::realtest_score --model /ckpt/smolvlm/drone-rover-v2/last

Each converted set is /data/realtest/<set>/labels.jsonl + frames/*.jpg (long side 512 px), with
one row per frame in the probe.py shape (visible, bearing_deg, range proxy) plus the box itself,
so results can be read in pixel terms that do not depend on the assumed field of view.
"""
import io, json, math, os, re, zipfile
import numpy as np

# sources (see results/realtest/README.md for the terms). Both are read as remote zips with HTTP range
# requests (RemoteFile), so only the central directory and the members actually used are transferred:
# no whole-archive download, on the Modal worker or anywhere else.
SOURCES = {
    # UAV123 (Mueller, Smith, Ghanem, ECCV 2016), the official 10 fps release (4.7 GB zip: the same
    # sequences and boxes as the 30 fps one, every 3rd frame), from the Google Drive link on
    # https://ivul.kaust.edu.sa/benchmark-and-simulator-uav-tracking-dataset (no form or login).
    "uav123_10fps": {"url": "https://drive.usercontent.google.com/download?id=0B6sQMCU1i4NbZmFlQmJBVDlLRDg"
                            "&export=download&confirm=t&resourcekey=0--jsSKS1oGFidNhgMF75cSQ"},
    # VisDrone2019-SOT (Zhu et al., TPAMI 2021). The official Google Drive links in
    # https://github.com/VisDrone/VisDrone-Dataset answered "Quota exceeded" for every part (val,
    # test-dev, train 1/2) when this was built; the Baidu mirror needs an account. This is a
    # third-party re-upload of the full release on the Hugging Face Hub (72.7 GB zip, public, not gated).
    "visdrone_sot": {"url": "https://huggingface.co/datasets/huseyincavus/visdrone2019-sot/resolve/main/"
                            "visdrone2019-sot.zip"},
    "visdrone_sot_val_gdrive": {"url": "https://drive.usercontent.google.com/download?"
                                       "id=18SNAOlCJtApnG2m45ud-1e_OtGYill0D&export=download&confirm=t"},
    "visdrone_sot_testdev_gdrive": {"url": "https://drive.usercontent.google.com/download?"
                                           "id=1xCiHjU4JlR9QsYtiHYy2UUd3m6NthoBC&export=download&confirm=t"},
}

OUT_DIR = "/data/realtest"


class RemoteFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, for zipfile.ZipFile(RemoteFile(url))."""

    def __init__(self, url, block=4 << 20):
        import requests
        self.url, self.s, self.pos, self.block = url, requests.Session(), 0, block
        self.cache = {}
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=120, allow_redirects=True)
        if r.status_code != 206:
            raise RuntimeError("%s: no range support (HTTP %d, %s): %r" % (
                url, r.status_code, r.headers.get("content-type"), r.raw.read(300)))
        self.final = r.url
        self.size = int(r.headers["content-range"].split("/")[1])
        r.close()
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _get(self, a, b):
        import time
        for k in range(6):
            try:
                r = self.s.get(self.final, headers={"Range": "bytes=%d-%d" % (a, b)}, timeout=300)
                if r.status_code in (401, 403):   # a signed redirect expired: resolve it again
                    self.final = self.s.get(self.url, headers={"Range": "bytes=0-0"}, stream=True,
                                            timeout=120).url
                    raise RuntimeError("HTTP %d" % r.status_code)
                if r.status_code != 206:
                    raise RuntimeError("HTTP %d" % r.status_code)
                self.bytes_read += len(r.content)
                return r.content
            except Exception:
                if k == 5:
                    raise
                time.sleep(2 * (k + 1))

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > self.block:            # big reads (a whole member) go straight through
            out = self._get(self.pos, self.pos + n - 1)
        else:                         # small ones (headers, the central directory) through a block cache
            out = b""
            while len(out) < n:
                bi = (self.pos + len(out)) // self.block
                if bi not in self.cache:
                    if len(self.cache) > 64:
                        self.cache.clear()
                    a = bi * self.block
                    self.cache[bi] = self._get(a, min(self.size, a + self.block) - 1)
                blk = self.cache[bi]
                o = self.pos + len(out) - bi * self.block
                out += blk[o:o + n - len(out)]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def open_zip(name):
    return zipfile.ZipFile(RemoteFile(SOURCES[name]["url"]))


def inspect(name, n=40):
    """A summary of a source zip's layout: member count, top-level dirs, sample names, a few annotation lines."""
    z = open_zip(name)
    names = z.namelist()
    dirs = sorted({"/".join(x.split("/")[:3]) for x in names})
    txt = [x for x in names if x.endswith(".txt")]
    sample = {t: z.read(t).decode("utf8", "replace").splitlines()[:3] for t in txt[:6]}
    return {"name": name, "bytes": z.fp.size if hasattr(z.fp, "size") else None, "members": len(names), "dirs": dirs[:400],
            "txt": txt[:600], "sample_names": names[:n], "sample_txt": sample}
