# Security Policy

## Scope

LocalDeploy is designed for one person on a trusted workstation. The API listens on `127.0.0.1` by default and local inference backends must use loopback addresses.

It is not an internet-facing service. It has no TLS setup, user accounts, tenant isolation, rate limiting, quotas, or audit log. Do not put it behind a public proxy or tunnel. A shared `API_TOKEN` can reduce accidental local or trusted-LAN access, but it does not turn LocalDeploy into a multi-user service.

## What it can do

LocalDeploy can start and stop Ollama, pull and delete models, choose CPU or GPU placement, write configuration and report files, and run generated benchmark code in a restricted child process. Run it as a normal user, not an administrator or root account unless the platform requires that for installation.

The web control routes can change local state. In particular, they can download large model files, delete installed models, unload models, run benchmarks, and update `config.json`. Keep the API on loopback and set `API_TOKEN` if another local process should not be able to call it freely.

## Network boundary

Inference backend URLs are checked by `localdeploy/utils.py::is_loopback_url`. Ollama normally uses `http://127.0.0.1:11434`; llama.cpp and other OpenAI-compatible runtimes must also use localhost, `127.0.0.1`, or `::1`.

If `API_HOST` is not a loopback address, startup prints a warning. `REQUIRE_TOKEN_ON_LAN=true` turns a tokenless non-loopback bind into a startup failure. Docker binds to `0.0.0.0` inside the container, but the provided Compose file maps the host port to `127.0.0.1` only. Changing that host mapping changes the security boundary.

## Data and outbound requests

Prompts, responses, images, hardware details, and benchmark reports are not uploaded by LocalDeploy. Monitor request history contains numeric timing and token counts, not prompt or response text. A separately managed runtime can have its own network behavior. The supplied Docker setup and local launchers set `OLLAMA_NO_CLOUD=true` when they start Ollama.

Model search sends the search text to the Ollama library and Hugging Face. The UI also checks this repository's GitHub releases once per page load. Set `OFFLINE=true` to disable these LocalDeploy requests. Local backend calls continue to work. `python scripts/egress_selftest.py` checks the offline path.

Do not commit `.env`, `config.json`, `calibration.json`, model files, private screenshots, or benchmark reports that contain private prompts. These paths are ignored by Git, but check unfamiliar files with `git check-ignore -v <path>` before committing.

## Known limits

| Limit | Why it matters |
|---|---|
| One optional shared token | There is no per-user identity, revocation, or authorization policy. |
| Plain HTTP | A token can be observed if traffic crosses an untrusted network. |
| No rate or concurrency limits | A caller can fill disk, occupy memory, or overload a backend. |
| Backend error details may be returned | Local paths or runtime details can appear in errors. |
| No prompt or output filtering | Do not assume model input or output is safe to pass to another system. |
| Restricted code grader | The grader blocks imports, builtins, filesystem, process, network, registry, and native-library access, with time and POSIX resource limits. It is defense in depth, not a hardened virtual machine. Do not benchmark deliberately hostile code as a privileged user. |

## Reporting a vulnerability

Use the **Report a vulnerability** button on the repository's Security tab instead of opening a public issue. GitHub keeps the report private between you and the maintainer.

If that button is not available, open a minimal issue asking for a private contact method. Do not put exploit details, tokens, private prompts, images, model files, or local paths in the public issue.

## Disclaimer

LocalDeploy is provided without warranty under the [MIT License](LICENSE). It controls processes, hardware placement, and files on the machine. Review the code and configuration for your environment and keep recoverable copies of anything important.
