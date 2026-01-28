# Test Setup Requirements - Truly Real Tests

## Summary

The `truly_real_tests.sh` script reports **5 skipped tests** that require additional setup. This document identifies the root causes and required fixes.

## Issues Identified

### 1. Semgrep Finding 0 Vulnerabilities (Primary Issue)

**Problem:**
- Scans complete but find 0 vulnerabilities: `"Scan complete: 0 findings in 0 files"`
- This causes multiple tests to skip because they expect actual findings

**Root Cause:**
- Semgrep rules may not match the patterns in `test/data/python_sql_injection.py`
- The test data contains vulnerabilities, but Semgrep isn't detecting them

**Required Setup:**
1. **Verify Semgrep rules match test data patterns:**
   - Check that `engine/semgrep/rules/secrets/` contains rules that detect hardcoded passwords
   - Check that `engine/semgrep/rules/injection/` contains rules that detect SQL injection and command injection
   - The test data has:
     - SQL injection: `query = "SELECT * FROM users WHERE id = " + user_id`
     - Command injection: `subprocess.run(f"convert {filename} output.jpg", shell=True)`
     - Hardcoded secrets: `DATABASE_PASSWORD = "admin123!SuperSecret"`

2. **Verify Semgrep registry packs are working:**
   - The scanner uses both local rules AND Semgrep registry packs (e.g., `p/secrets`, `p/command-injection`)
   - Ensure network access allows downloading Semgrep packs
   - Check that Semgrep can access the registry

3. **Test Semgrep directly:**
   ```bash
   # Test if Semgrep can detect the vulnerabilities
   cd test/data
   semgrep --config=p/secrets python_sql_injection.py
   semgrep --config=p/command-injection python_sql_injection.py
   ```

### 2. Test Script Bugs with SARIF File Handling

**Problem:**
- Line 115: `if [ -f "$FINDINGS" ]` fails because `$FINDINGS` contains multiple file paths from `find`
- Lines 151-152, 156-157: When scans return 0 findings, `SECRETS_SARIF` may be empty or the grep returns 0

**Required Fix:**
Update `test/truly_real_tests.sh` to properly handle:
- Multiple SARIF files (use `head -1` or iterate)
- Empty SARIF files (check file exists AND has content)
- 0 findings (handle gracefully instead of skipping)

**Example Fix:**
```bash
# Line 115 - Fix FINDINGS check
FIRST_FINDING=$(echo "$FINDINGS" | head -1)
if [ -f "$FIRST_FINDING" ] && [ -s "$FIRST_FINDING" ]; then
    # Check for shell injection
    if grep -q "shell\|subprocess" "$FIRST_FINDING"; then
        test_case "Semgrep detects shell injection vulnerability" "PASS"
    else
        test_case "Semgrep detects shell injection vulnerability" "SKIP" "Not in default rules"
    fi
else
    test_case "Semgrep detects shell injection vulnerability" "SKIP" "No SARIF output"
fi
```

### 3. Analyze Command Missing Flags

**Problem:**
- Error: `agent.py: error: unrecognized arguments: --no-exploits --no-patches`
- The test script (line 224) tries to use flags that don't exist in `packages/llm_analysis/agent.py`

**Required Fix:**
Add support for `--no-exploits` and `--no-patches` flags to `packages/llm_analysis/agent.py`:

```python
# In main() function around line 1135
ap.add_argument("--no-exploits", action="store_true", help="Skip exploit generation")
ap.add_argument("--no-patches", action="store_true", help="Skip patch generation")

# Then pass these to the agent or modify process_findings to accept them
# The agent would need to check these flags before calling generate_exploit() and generate_patch()
```

**Alternative Fix:**
Remove `--no-exploits --no-patches` from the test script if these flags aren't needed for the test.

### 4. SARIF File Path Issues

**Problem:**
- Tests 4.1, 4.2, 4.3 skip because `$SECRETS_SARIF` is not set or is empty
- This happens when the scan finds 0 findings and the SARIF file path isn't captured correctly

**Required Fix:**
Ensure `$SECRETS_SARIF` is always set to a valid SARIF file path, even if it contains 0 findings:

```bash
# After line 151
if [ -z "$SECRETS_SARIF" ] || [ ! -f "$SECRETS_SARIF" ]; then
    # Try to find the combined SARIF file
    SECRETS_SARIF=$(find "$PROJECT_ROOT/out" -name "combined.sarif" -type f 2>/dev/null | head -1)
fi
```

## Specific Test Failures

### Test 2.3: "Semgrep detects shell injection vulnerability"
- **Status:** SKIP (requires: No SARIF output)
- **Fix:** Fix `$FINDINGS` variable handling (Issue #2)

### Test 3.1: "Policy groups filter results"
- **Status:** SKIP (requires: Both returned same count)
- **Fix:** Ensure Semgrep detects vulnerabilities (Issue #1)

### Test 4.1: "SARIF output is valid JSON"
- **Status:** SKIP (requires: No SARIF file)
- **Fix:** Fix `$SECRETS_SARIF` path handling (Issue #4)

### Test 4.2: "SARIF has required SARIF 2.1 fields"
- **Status:** SKIP (requires: No SARIF file)
- **Fix:** Fix `$SECRETS_SARIF` path handling (Issue #4)

### Test 4.3: "SARIF results have rule/message/location info"
- **Status:** SKIP (requires: No SARIF file)
- **Fix:** Fix `$SECRETS_SARIF` path handling (Issue #4)

## Recommended Action Plan

1. **Immediate:** Fix test script bugs (Issues #2, #4)
   - This will allow tests to run even with 0 findings
   - Tests will properly skip with clear reasons

2. **Short-term:** Add missing flags to analyze command (Issue #3)
   - Add `--no-exploits` and `--no-patches` support
   - Or remove them from test if not needed

3. **Long-term:** Fix Semgrep detection (Issue #1)
   - Verify rules match test data patterns
   - Update test data or rules as needed
   - Ensure Semgrep registry access works in CI

## Verification Steps

After fixes, verify:

```bash
# 1. Test Semgrep directly
cd test/data
semgrep --config=p/secrets python_sql_injection.py
semgrep --config=p/command-injection python_sql_injection.py

# 2. Run the test script
bash test/truly_real_tests.sh

# 3. Check that SARIF files are created
ls -la out/scan_data_*/combined.sarif

# 4. Verify SARIF contains findings
grep -c '"ruleId"' out/scan_data_*/combined.sarif
```

## Notes

- The test script currently passes (7/12 tests) but skips 5 tests
- The skipped tests are all related to the same root causes above
- Once Semgrep detects vulnerabilities, most tests should pass
- The test script improvements will make failures more informative
