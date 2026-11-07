#!/usr/bin/env bash
# Prepare only the isolated leased container. Never use shell tracing here.
set -eu
cd /opt/vmecpp-experiment
for source in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources; do
  if [ -f "$source" ]; then
    sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g;s|http://security.ubuntu.com|https://security.ubuntu.com|g' "$source"
  fi
done
apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15 update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv \
  libhdf5-dev libnetcdf-dev liblapack-dev libgomp1 libgl1 libsm6 libxext6
python3 -m venv baseline
tar -xzf oracle-packages.tar.gz -C baseline/lib/python3.12/site-packages
for variant in candidate gpu; do
  python3 -m venv "$variant"
  printf '%s\n' /opt/vmecpp-experiment/baseline/lib/python3.12/site-packages \
    > "$variant/lib/python3.12/site-packages/oracle_dependencies.pth"
  cp -a baseline/lib/python3.12/site-packages/vmecpp \
    "$variant/lib/python3.12/site-packages/"
done
cp locality-extension.so candidate/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so
cp gpu-extension.so gpu/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so
python3 - <<'PY'
import json
from pathlib import Path
p = Path('fixtures/manifest.json')
rows = json.loads(p.read_text())
for row in rows:
    row['path'] = str(Path('fixtures', Path(row['path']).name).resolve())
p.write_text(json.dumps(rows, indent=2))
PY
mkdir -p results
nvidia-smi -q > results/gpu.txt
lscpu > results/cpu.txt
ldd gpu/lib/python3.12/site-packages/vmecpp/cpp/_vmecpp.cpython-312-x86_64-linux-gnu.so \
  > results/gpu-extension-libraries.txt
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
baseline/bin/python -c 'import vmecpp, constellaration; print("Pinned oracle imports succeeded")'
if [ -x /usr/local/cuda/bin/nvcc ]; then
  /usr/local/cuda/bin/nvcc -O3 -arch=sm_80 tools/optimization/roofline.cu -o results/roofline
else
  cp roofline-a100 results/roofline
fi
results/roofline > results/roofline.json
