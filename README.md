# BioSensIA-DC


Welcome to the repository for the **BioSensIA-DC prototype**. This is
work in progress.

## What BioSensIA-DC is

??? Cite DrugCLIP paper

## How to install BioSensIA-DC

### Install `uv`

[`uv`](https://docs.astral.sh/uv/) is a modern and efficient package and
environment manager for Python. Please refer to their website for
[instructions on how to install
it](https://docs.astral.sh/uv/getting-started/installation/) for your
system.

### Check your CUDA driver

1.  Make sure your GPU is available — for example, if you use
    [Slurm](https://slurm.schedmd.com/overview.html) on an HPC cluster,
    open a shell on a GPU node with the following command:

    ``` bash
    srun --partition=gpu --gres=gpu:1 --time=00:05:00 --pty bash
    ```

2.  Run the [`nvcc`
    command](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html):

    ``` bash
    nvcc --version
    ```

    The output will be something like

        nvcc: NVIDIA (R) Cuda compiler driver
        Copyright (c) 2005-2024 NVIDIA Corporation
        Built on Tue_Feb_27_16:19:38_PST_2024
        Cuda compilation tools, release 12.4, V12.4.99
        Build cuda_12.4.r12.4/compiler.33961263_0

    This says we have release 12.4.

### Clone this repo

Go to a directory of your choice and run

``` bash
git clone https://github.com/fnaufel/BioSensIA-DC.git
```

or download a [zip file of the master
branch](https://github.com/fnaufel/BioSensIA-DC/archive/refs/heads/master.zip)
and unzip it.

Then go to the project’s directory:

``` bash
cd BioSensIA-DC
```

### Edit `pyproject.toml` to set correct values for your system

Open `pyproject.toml` in a text editor. You will see these lines at the
end of the file:

``` toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]
torchvision = [
  { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]
torchaudio = [
  { index = "pytorch-cu124", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

1.  If necessary, change `cu124` to `cuxxx`, where `xxx` is the release
    of your CUDA driver.

2.  If necessary, change `linux` to your operating system (e.g.,
    `win32`).

3.  **Do not** change anything else in this file, unless you know what
    you are doing.

For more details, see the [uv page about installing
PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).

### Use `uv` to install the dependencies

``` bash
uv sync
```

### Activate the virtual environment

``` bash
source .venv/bin/activate
```

### Confirm everything is ok so far

#### Python version

1.  Run

    ``` bash
    which python
    ```

    The output should end with `BioSensIA-DC/.venv/bin/python`

2.  Run

    ``` bash
    python --version
    ```

    The output be of the form

        Python 3.11.xx

    where `xx` varies according to the [patch
    version](https://semver.org) installed.

#### PyTorch with GPU access

1.  Make sure your GPU is available — for example, if you use
    [Slurm](https://slurm.schedmd.com/overview.html) on an HPC cluster,
    open a shell on a GPU node with the following command:

    ``` bash
    srun --partition=gpu --gres=gpu:1 --time=00:05:00 --pty bash
    ```

2.  Run the following in your shell:

    ``` bash
    python - << PY
    import os
    import torch

    x = torch.rand(5, 3)
    print(x)
    print("torch.__version__                   =", torch.__version__)
    print("torch.version.cuda                  =", torch.version.cuda)
    print("torch.backends.cuda.is_built        =", torch.backends.cuda.is_built())
    print("torch.cuda.is_available             =", torch.cuda.is_available())
    print("torch.cuda.device_count             =", torch.cuda.device_count())

    if torch.cuda.is_available():
      print("device 0 name                       =", torch.cuda.get_device_name(0))
    print("torch.distributed.is_nccl_available =", torch.distributed.is_nccl_available())
    PY
    ```

    The output should be something like

        tensor([[0.4314, 0.7784, 0.5975],
                [0.5675, 0.0199, 0.3430],
                [0.9700, 0.3156, 0.9717],
                [0.1576, 0.8838, 0.7823],
                [0.1090, 0.2737, 0.9940]])
        torch.__version__                   = 2.6.0+cu124
        torch.version.cuda                  = 12.4
        torch.backends.cuda.is_built        = True
        torch.cuda.is_available             = True
        torch.cuda.device_count             = 1
        device 0 name                       = NVIDIA A2
        torch.distributed.is_nccl_available = True

#### RDKit

Run

``` bash
python - <<'PY'
from rdkit import Chem
from rdkit.Chem import Descriptors
print("RDKit version:", Chem.rdBase.rdkitVersion)
mol = Chem.MolFromSmiles("CCO")
print("Canonical SMILES:", Chem.MolToSmiles(mol))
print("Num atoms:", mol.GetNumAtoms())
print("Molecular weight:", Descriptors.MolWt(mol))
PY
```

The output should be

    RDKit version: 2022.09.5
    Canonical SMILES: CCO
    Num atoms: 3
    Molecular weight: 46.069

### Install Uni-Core

[Uni-Core](https://github.com/dptech-corp/Uni-Core) is a distributed
PyTorch framework on which BioSensIA-DC depends.

It must be built from source (with a build option to enable CUDA
extensions) and manually overlaid into the virtual environment.

The Uni-Core source is included in this repo. Change to the appropriate
directory:

``` bash
cd ./external/Uni-Core/
```

Make sure the virtual environment is activated:

``` bash
source ../../.venv/bin/activate
```

Build with

``` bash
python setup.py install --enable-cuda-ext
```

> [!WARNING] Urgent info that needs immediate user attention to avoid
> problems.

> [!CAUTION] As Uni-Core has been built from source (and not installed
> via `uv`), it is not formally included in the list of dependencies in
> `pyproject.toml`.
>
> This means that running `uv sync` after this point **will remove it**
> from the environment.
>
> To avoid this, if you need to synchronize the environment, use

> ``` bash
> uv sync --inexact
> ```

> instead. This will preserve packages installed by means other than
> `uv`.

## How to use BioSensIA-DC

???
