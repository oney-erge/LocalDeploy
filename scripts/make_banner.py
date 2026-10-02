#!/usr/bin/env python3
"""Render docs/assets/banner.png, the 1280x640 social-preview card, from the real logo.

The tagline and the pills are claims about the product, so they live here in source
instead of in a hand-edited PNG. Keep them in line with the README: model search and
the release check make outbound requests unless OFFLINE=true, so the banner must not
say "100% offline".

    python -m playwright install chromium   # once
    python scripts/make_banner.py
"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
LOGO = (ROOT / "localdeploy" / "web" / "logo.svg").read_text(encoding="utf-8")
OUT = ROOT / "docs" / "assets" / "banner.png"

TAGLINE = "Pick, deploy &amp; benchmark the best local AI model for your machine."
PILLS = ["Ollama", "Fit-checked pulls", "Benchmarks", "No telemetry"]

HTML = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  html, body {{ margin: 0; width: 1280px; height: 640px; overflow: hidden; }}
  body {{
    background:
      radial-gradient(900px 520px at 100% 0%, rgba(59, 99, 190, .28), transparent 60%),
      radial-gradient(700px 420px at 0% 100%, rgba(16, 120, 110, .22), transparent 60%),
      #0f1115;
    font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
    display: flex; align-items: center; justify-content: center; gap: 56px;
  }}
  .logo {{ width: 206px; height: 206px; filter: drop-shadow(0 0 46px rgba(34, 160, 238, .38)); flex: none; }}
  .logo svg {{ width: 100%; height: 100%; display: block; }}
  .copy {{ width: 730px; }}
  h1 {{ margin: 0; font-size: 92px; font-weight: 700; letter-spacing: -2px; line-height: 1; color: #eceef3; }}
  h1 span {{ background: linear-gradient(90deg, #7ba8ff, #22d3ee); -webkit-background-clip: text;
             background-clip: text; color: transparent; }}
  p {{ margin: 22px 0 22px; font-size: 31px; line-height: 1.32; color: #9aa7bd; }}
  .pills {{ display: flex; gap: 12px; flex-wrap: nowrap; }}
  .pill {{ font-size: 21px; color: #b9c6dd; padding: 9px 20px; border-radius: 999px;
           border: 1px solid #2c3a55; background: rgba(23, 31, 49, .85); }}
</style></head><body>
  <div class="logo">{LOGO}</div>
  <div class="copy">
    <h1>Local<span>Deploy</span></h1>
    <p>{TAGLINE}</p>
    <div class="pills">{"".join(f'<div class="pill">{p}</div>' for p in PILLS)}</div>
  </div>
</body></html>
"""


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 640})
        page.set_content(HTML)
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT), type="png")
        browser.close()
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
