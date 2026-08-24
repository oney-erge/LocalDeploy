# Releasing LocalDeploy

GitHub releases publish the Python package to PyPI and the container image to GHCR. The wheel and source archive are also attached to the GitHub release. PyPI uses trusted publishing, so the repository does not store an API token.

## Prepare a release

1. Update `localdeploy/__init__.py::__version__` and add the matching section to `CHANGELOG.md`.
2. Run the local checks from a clean checkout.
3. Merge the release change to `main` and wait for CI.
4. Create a GitHub release whose tag is exactly `v<package-version>`, for example `v0.6.0`.

Local checks:

```powershell
python -m ruff check .
pytest -q
node --test tests/js/frontend-modules.test.mjs
python scripts\egress_selftest.py
python -m build
python -m twine check dist/*
```

The publishing workflows check that the tag and package version match. PyPI uses OpenID Connect to request a short-lived token. GHCR uses the release workflow's repository-scoped token.

After the workflow completes, verify the package from a clean environment outside the repository:

```powershell
python -m venv release-check
.\release-check\Scripts\python.exe -m pip install localdeploy==0.6.0
.\release-check\Scripts\localdeploy.exe --version
```

Use the released version in place of `0.6.0`. Verify the container separately:

```bash
docker pull ghcr.io/oney-erge/localdeploy:0.6.0
```
