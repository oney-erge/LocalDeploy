# Examples

Small clients for the LocalDeploy API. They expect the server at `http://127.0.0.1:8000`. Start it with `.\run.bat` on Windows, `./run.command` on macOS, or `./run.sh` on Linux.

| File | What it shows |
|---|---|
| [curl_chat.sh](curl_chat.sh) | Native `/chat` and OpenAI-compatible `/v1/chat/completions` via `curl`. |
| [python_client.py](python_client.py) | Native `/chat`, profile listing, optional token auth, and OpenAI SDK usage from Python. |
| [vision_chat.ps1](vision_chat.ps1) | Sending an image to `/vision` from PowerShell. |

The native `/chat` API is the shortest path for a custom client. Existing OpenAI-compatible clients can use `http://127.0.0.1:8000/v1` as their base URL.

Set `API_TOKEN` in the shell before running an example when the server uses token protection. The shell example also accepts `LOCALDEPLOY_API_TOKEN`.

See [../docs/API_OPTIONS.md](../docs/API_OPTIONS.md) for the full request schema and [../docs/MODELS.md](../docs/MODELS.md) for the model catalog.
