# Terminal Usage

Use these commands to chat or compare models without the browser.

Start the server first if it is not already running:

```powershell
.\run.ps1 -NoBrowser
```

On macOS or Linux, use `./run.sh --no-browser`.

## Chat helper

One-shot prompt:

```powershell
.\scripts\chat.ps1 -Prompt "How are you?"
```

Interactive chat:

```powershell
.\scripts\chat.ps1
```

Useful commands inside interactive mode:

```text
:profiles
:profile gemma3_12b_ollama_safe
:tokens 512
:raw
:quit
```

Start the server automatically if it is not running:

```powershell
.\scripts\chat.ps1 -StartServer -Prompt "Say ready."
```

Use a specific profile (a profile name from `config.json`):

```powershell
.\scripts\chat.ps1 -ProfileName gemma3_12b_ollama_safe -Prompt "Answer in one paragraph: what are you best at?"
```

If `-ProfileName` is omitted, the script falls back to the `DEFAULT_MODEL_PROFILE` env var, then to the default profile configured in `config.json`.

Show the full JSON response:

```powershell
.\scripts\chat.ps1 -Prompt "Give me a short test response." -Raw
```

List configured profiles:

```powershell
.\scripts\chat.ps1 -Profiles
```

## Comparing models

`compare_models.py` runs a fixed prompt battery (reasoning, coding, JSON, vision) against one or more profiles and prints timing and output quality side by side:

```powershell
python compare_models.py --all
```

For the leaderboard, heatmap, report cards, and run comparison, use the Benchmark and Compare tab described in [UI.md](UI.md).

## Calling the API by hand

Swagger UI (`/docs`) shows generated placeholder values such as `"profile": "string"` and `max_output_tokens: 0`. Use a real profile name and omit fields you do not need. For `/chat`, this is enough:

```json
{
  "prompt": "how are you"
}
```

Full request schema and limit rules: [API_OPTIONS.md](API_OPTIONS.md).
