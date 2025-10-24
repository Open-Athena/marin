# Step 2 Migration Issues and Fixes

## Issues Found During Manual Completion

### 1. ModuleNotFoundError: yaya not installed

**Issue**: The `update_workflows.py` script crashed with `ModuleNotFoundError: No module named 'yaya'` when run from `step2/main.sh`.

**Root cause**: The inline script has `# dependencies = ["yaya @ file:///Users/ryan/c/yaya"]` in the uv shebang, but there was also an obsolete `sys.path.insert(0, str(Path.home() / "c/lossless-yaml/src"))` line that referenced the old package name.

**Fix**: Removed the `sys.path.insert` line in commit 7857a5b8.

**Impact**: Without this fix, `step2/main.sh` would crash after moving workflow files but before updating their content.

---

### 2. WorkflowConflict: marin-tpu-tests.yaml not in expected list

**Issue**: `update_workflows.py` raised `WorkflowConflict` for `marin-tpu-tests.yaml`, complaining it wasn't in the expected workflows list.

**Root cause**: The workflow was added to the marin repo in commit 4b7225f9d (after step 1 was developed), so it wasn't in the `EXPECTED_MARIN_WORKFLOWS` whitelist.

**Fix**: Added `"marin-tpu-tests.yaml"` to the `EXPECTED_MARIN_WORKFLOWS` set in commit 7857a5b8.

**Impact**: Without this fix, the script would exit with an error and the workflow name wouldn't get the "Marin - " prefix.

---

### 3. YAML corruption in levanter-publish_dev.yaml

**Issue**: The `build-package` job in `levanter-publish_dev.yaml` had corrupted YAML:
```yaml
if: $
defaults:
  run:
    working-directory: lib/levanter{{  github.event_name == 'workflow_dispatch' || github.event.workflow_run.conclusion == 'success'}}
```

The `if:` condition value was split, with `$` on the `if:` line and the rest concatenated onto the `working-directory:` value.

**Root cause**: Possible bug in yaya's `add_key_after` API when adding a key after an `if:` key with a jinja2 expression. The original file had:
```yaml
if: ${{  github.event_name == 'workflow_dispatch' || workflow_run.conclusion == 'success'}}
```

**Fix**: Manually corrected the YAML to:
```yaml
if: ${{ github.event_name == 'workflow_dispatch' || github.event.workflow_run.conclusion == 'success'}}
defaults:
  run:
    working-directory: lib/levanter
```

**Impact**: Without this fix, the workflow would have invalid YAML and fail to parse.

**TODO**: Investigate whether this is reproducible with yaya's `add_key_after` and file a bug report if so.

---

## Testing Recommendations

When testing future step2 runs:

1. **Use `--lock-ref` to skip `uv sync`**: The `step2/main.sh` script supports `-l REF` or `--lock-ref REF` to check out `uv.lock` from a git ref instead of re-resolving (which can take 5-10 minutes). Example:
   ```bash
   ./workspace-migration/step2/main.sh --lock-ref ws-2
   ```

2. **Verify workflow transformations**: After running step2, check:
   - All workflow files have correct name prefixes ("Marin - " / "Levanter - ")
   - Step names were NOT changed (only workflow names)
   - Levanter workflows have `working-directory: lib/levanter` in job defaults
   - setup-uv steps have `working-directory: lib/levanter` in their `with:` sections

3. **Check for YAML corruption**: Validate that all workflow files parse correctly:
   ```bash
   for f in .github/workflows/*.yaml; do
     python -c "import yaml; yaml.safe_load(open('$f'))" || echo "ERROR: $f"
   done
   ```
