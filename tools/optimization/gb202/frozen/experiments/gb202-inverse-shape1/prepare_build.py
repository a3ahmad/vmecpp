import ctypes
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

B = Path('/mnt/f/Users/might/Sync/vmecpp/build/optimization')
R = B / 'gb202-inverse-shape1'
S = R / 'source'
O = R / 'build1'
S.mkdir(parents=True, exist_ok=True)
O.mkdir(exist_ok=True)
base = B / 'inverse-load-hoist1/source'
for name in ('gpu_fourier.cu', 'gpu_fourier.h', 'generated_fourier_kernel.cuh'):
    shutil.copyfile(base / name, S / name)
header = (S / 'generated_fourier_kernel.cuh').read_text()
start = header.index('__global__ void FourierToRealKernelSeparable(')
original = header[start:]
specialized = original.replace(
    '__global__ void FourierToRealKernelSeparable(',
    'template <int kMpol, int kNtor, int kNZeta, int kNTheta, int kNnyq2>\n'
    '__global__ void FourierToRealKernelAngularShape(', 1)
anchor = '    KernelParams params) {\n'
constants = ('  params.mpol = kMpol;\n  params.ntor = kNtor;\n'
             '  params.n_zeta = kNZeta;\n  params.n_theta_eff = kNTheta;\n'
             '  params.n_theta_reduced = kNTheta;\n  params.nnyq2 = kNnyq2;\n')
assert specialized.count(anchor) == 1
specialized = specialized.replace(anchor, anchor + constants, 1)
header += '\n#if VMECPP_INVERSE_ANGULAR_SHAPE\n' + specialized + '\n#endif\n'
header = ('#ifndef VMECPP_INVERSE_ANGULAR_SHAPE\n'
          '#define VMECPP_INVERSE_ANGULAR_SHAPE 0\n#endif\n') + header
(S / 'generated_fourier_kernel.cuh').write_text(header)
cu = (S / 'gpu_fourier.cu').read_text()
begin = cu.index('  FourierToRealKernel<<<grid, block_size, 0, stream>>>(')
end = cu.index('  Check(cudaGetLastError());', begin)
launch = cu[begin:end]
fixed = launch.replace('FourierToRealKernel<<<',
                       'FourierToRealKernelAngularShape<10, 10, 24, 14, 24><<<', 1)
replacement = ('#if VMECPP_INVERSE_ANGULAR_SHAPE\n'
               '  if (d[5] == 10 && d[6] == 10 && d[7] == 24 &&\n'
               '      d[8] == 14 && d[9] == 14 && d[10] == 24) {\n' +
               fixed + '  } else {\n#endif\n' + launch +
               '#if VMECPP_INVERSE_ANGULAR_SHAPE\n  }\n#endif\n')
cu = cu[:begin] + replacement + cu[end:]
(S / 'gpu_fourier.cu').write_text(cu)
assert original in header
assert specialized.replace(constants, '', 1).replace(
    'template <int kMpol, int kNtor, int kNZeta, int kNTheta, int kNnyq2>\n'
    '__global__ void FourierToRealKernelAngularShape(',
    '__global__ void FourierToRealKernelSeparable(', 1) == original
(R / 'source-delta.diff').write_text(''.join(
    ''.join(difflib.unified_diff((base / name).read_text().splitlines(True),
                               (S / name).read_text().splitlines(True),
                               fromfile='baseline/' + name, tofile='candidate/' + name))
    for name in ('gpu_fourier.cu', 'generated_fourier_kernel.cuh')))
spec = importlib.util.spec_from_file_location('owner', B / 'resident-solver-energy-ragged-clean1/compiler_owner.py')
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
old = json.loads((B / 'gb202-inverse-groups1/build1/receipt.json').read_text())
compile_cmd = old['jobs'][0]['command']
compile_cmd = [x.replace('gb202-inverse-groups1', 'gb202-inverse-shape1')
               .replace('-DVMECPP_INVERSE_SPLIT_OUTPUT_GROUPS=1',
                        '-DVMECPP_INVERSE_ANGULAR_SHAPE=1') for x in compile_cmd]
link_cmd = [x.replace('gb202-inverse-groups1', 'gb202-inverse-shape1')
            for x in old['jobs'][1]['command']]
jobs = []
started = time.time()
for name, cmd in [('shape-SM120-compile', compile_cmd), ('sameotherobjects-link', link_cmd),
                  ('resources', ['/usr/local/cuda-12.9/bin/cuobjdump', '--dump-resource-usage', str(O / 'resident.so')]),
                  ('sass', ['/usr/local/cuda-12.9/bin/cuobjdump', '--dump-sass', str(O / 'resident.so')])]:
    result = owner.run_job(cmd, name, 180, dict(os.environ), O)
    jobs.append(result)
    assert result['direct_exit_status'] == 0 and result['owned_children_reaped']
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
receipt = {'jobs': jobs, 'wall_seconds': time.time() - started,
           'baseline': sha(B / 'gb202-packed-sm1201/build2/resident.so'),
           'candidate': sha(O / 'resident.so'),
           'source': {str(p.relative_to(R)): sha(p) for p in S.iterdir()},
           'generic_kernel_body_exact': True, 'specialized_fp_body_exact': True,
           'runtime_executed': False}
(O / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
