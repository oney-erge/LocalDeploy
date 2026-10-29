# Installation and runtime commands

The stable project entry points are `run.bat` on Windows, `run.command` on
macOS, and `run.sh` on Linux. Re-running them is safe. They use the existing
environment when it is current and repair missing dependencies when needed.
Each also accepts `doctor`, `repair`, `docker`, `logs`, and `stop`.

Setup checks disk space, prevents concurrent environment changes, retries
temporary network failures up to three times, and writes persistent failures
to `.setup/install.log`. The `logs` action follows the active native or
Docker logs rather than requiring Docker. If the existing `.venv` has a Python
that no longer starts (for example one built from an Anaconda that was later
removed), the launcher moves it to `.setup/venv-broken-<time>` and builds a new
one; nothing is deleted.

Decision models (`/v1/systemone`) are optional and need extra packages: run
`python -m pip install -r requirements-decision.txt` in LocalDeploy's
environment. The packages take about 4.5 GB with CUDA support, and each model
downloads on first use (about 2.2 GB for d1-omni-600M, 2.3 GB for Laya, 5.9 GB
for d1-3B). See [Decision models](API_OPTIONS.md#decision-models).

From a clone:

```powershell
.\run.bat          # Windows
```

```bash
./run.command      # macOS
./run.sh           # Linux
```

Use `.\run.ps1` from PowerShell. Add an action after any launcher, such as
`.\run.ps1 doctor` or `./run.sh docker`. The default `run` action waits for
LocalDeploy to become ready before opening the UI.

Install from PyPI when Python and Ollama are already available:

```bash
python -m pip install localdeploy
localdeploy
```

The launchers below can also set up a local clone and help install prerequisites.

### Windows

The installer can offer to install Python and Ollama through winget, clone the repository, create a virtual environment, and start the UI. Download it first so you can inspect it before running it:

```powershell
$installer = Join-Path $env:TEMP "localdeploy-install.ps1"
Invoke-RestMethod https://raw.githubusercontent.com/oney-erge/LocalDeploy/main/scripts/install.ps1 -OutFile $installer
& $installer
```

From an existing clone, use the stable root launcher:

```powershell
git clone https://github.com/oney-erge/LocalDeploy.git
cd LocalDeploy
.\run.bat
```

The launcher creates `.env`, `config.json`, and `.venv`, starts Ollama when it is installed, and opens `http://localhost:8000/ui`. Stop it with `.\run.ps1 stop`. The lower-level `scripts\start.ps1` remains available for development-specific flags.

### macOS and Linux

```bash
git clone https://github.com/oney-erge/LocalDeploy.git
cd LocalDeploy
./run.command  # macOS
./run.sh       # Linux
```

If Python 3 or Ollama is missing, the launcher offers to install it for you (Homebrew on macOS; apt, dnf, pacman, or zypper for Python and the [official installer](https://ollama.com/download) for Ollama on Linux) and asks before running anything. Answer no, or run non-interactively, and it prints the exact manual command instead.

On macOS you can double-click `run.command` from a clone. A ZIP download can strip the executable bit, so after unzipping run `chmod +x run.command run.sh scripts/*.sh` once, or use `git clone` as above.

The launcher starts Ollama when it is installed but not already reachable, then keeps the LocalDeploy server in the foreground. Press Ctrl+C to stop LocalDeploy. Run `./scripts/stop.sh --ollama` if you also want to stop an Ollama process started by the launcher. Set `START_OLLAMA=false` in `.env` when another service manages Ollama.

### Docker

The published container bundles Ollama, so Python is not needed on the host:

```bash
docker run -d --name localdeploy \
  -p 127.0.0.1:8000:8000 \
  -v localdeploy-data:/data/localdeploy \
  -v ollama-data:/root/.ollama \
  ghcr.io/oney-erge/localdeploy:latest
```

To build from source instead:

```bash
git clone https://github.com/oney-erge/LocalDeploy.git
cd LocalDeploy
docker compose up --build -d
```

Open `http://localhost:8000/ui`. Use `docker compose down` to stop it. The default port mapping only listens on `127.0.0.1`. Model and profile data live in named volumes. NVIDIA passthrough is available in `docker-compose.yml` and requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).


For the first-use walkthrough, see [UI](UI.md). For privacy and network settings, see [Security](../SECURITY.md). Desktop build instructions are in [Packaging](PACKAGING.md).
