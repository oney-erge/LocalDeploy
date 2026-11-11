# Lifecycle boundaries

- `localdeploy.backends.llamacpp_managed` owns the single managed chat-model subprocess. The synchronous inference adapter holds a request lease until its response is handled. Another profile cannot replace a live server with active requests or an uncertain request outcome; idle cleanup follows the same rule.
- A timeout or lost connection can leave server-side generation running. Status exposes `outcome_uncertain`; the manager refuses automatic switch/stop until the process is known dead. Restarting LocalDeploy stops its owned managed server during shutdown. A stop failure retains the process owner rather than claiming memory was released.
- Native Serve/Stop/Switch use the same owner for managed profiles. Stop checks the requested profile under the owner lock and never stops a different or externally managed server. Ollama and externally managed profiles retain their own lifecycle and existing controls.
- Installed runtime, reachable server, loaded model, active request, and observed unload are different facts. `GET /system/status` includes the managed runtime snapshot; compatible inference routes retain their existing response shape.

Focused coverage lives in `tests/test_llamacpp_managed.py` and model-control tests. The managed runtime is process-local: multiple LocalDeploy processes are not coordinated. Real llama.cpp, GPU release, and behavior after a disconnected client require target-platform verification.
