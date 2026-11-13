"""Export repeated-row throughput proxies; not independent equilibrium fixtures."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--repeat-rows", nargs="+", type=int, default=[4, 16])
    args = parser.parse_args()
    assert all(1 <= count <= 64 for count in args.repeat_rows)
    source = json.loads(args.manifest.read_text())
    jobs = {job["name"]: job for job in source["jobs"]}
    args.out.mkdir(parents=True, exist_ok=False)
    exported = []
    for name in args.job:
        job = jobs[name]
        data = (args.manifest.parent / job["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == job["sha256"]
        assert data[:8] == b"TEGEMM1\0"
        m, n, k = struct.unpack_from("<3Q", data, 8)
        a = data[32 : 32 + 8 * m * k]
        b = data[32 + 8 * m * k : 32 + 8 * (m * k + k * n)]
        c = data[32 + 8 * (m * k + k * n) :]
        assert len(c) == 8 * m * n
        for count in args.repeat_rows:
            label = f"{name}-rows{count}"
            path = args.out / (label + ".bin")
            with path.open("wb") as stream:
                stream.write(b"TEGEMM1\0" + struct.pack("<3Q", m * count, n, k))
                for column in range(k):
                    stream.write(a[8 * m * column : 8 * m * (column + 1)] * count)
                stream.write(b)
                for column in range(n):
                    stream.write(c[8 * m * column : 8 * m * (column + 1)] * count)
            exported.append(
                {
                    "name": label,
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_sha256": job["sha256"],
                    "repeat_rows": count,
                    "M": m * count,
                    "N": n,
                    "K": k,
                }
            )
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "scope": "Repeated real rows: occupancy/throughput proxy, not independent boundaries or solver batching",
                "exporter_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "jobs": exported,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
