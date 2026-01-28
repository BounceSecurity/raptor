# Semgrep Rules Fixes for Test Data Detection

## Summary

Updated Semgrep rules to properly detect vulnerabilities in `test/data/python_sql_injection.py`. The rules were not matching the actual patterns in the test code.

## Issues Fixed

### 1. SQL Injection Rule (`engine/semgrep/rules/injection/sql-concat.yaml`)

**Problem:**
- Pattern `query = "SELECT " + ...` was too restrictive
- Taint rule didn't match `cursor.execute(query)` (only looked for `$DB.execute($QUERY)`)
- Taint rule didn't match `request.form.get()` pattern

**Test Code Pattern:**
```python
query = "SELECT * FROM users WHERE id = " + user_id
cursor.execute(query)
```

**Fixes Applied:**
- Added more flexible patterns: `$QUERY = "SELECT" + ... + $USER_INPUT` and `$QUERY = "..." + $VAR`
- Added `cursor.execute()` and `cursor.executemany()` to pattern-sinks
- Added `request.form.get()`, `request.args.get()`, `flask.request.form.get()`, `flask.request.args.get()` to pattern-sources

### 2. Command Injection Rule (`engine/semgrep/rules/injection/command-taint.yaml`)

**Problem:**
- Pattern-sources only matched `flask.request.form[$_]` but test code uses `request.form.get('filename')`
- Sink pattern might not match f-string usage

**Test Code Pattern:**
```python
filename = request.form.get('filename')
result = subprocess.run(f"convert {filename} output.jpg", shell=True)
```

**Fixes Applied:**
- Added `request.form.get()`, `request.args.get()`, `flask.request.form.get()`, `flask.request.args.get()` to pattern-sources
- Added `subprocess.run(f"...", shell=True, ...)` and `subprocess.run($CMD + ..., shell=True, ...)` to pattern-sinks

### 3. Hardcoded Secrets Rule (`engine/semgrep/rules/secrets/hardcoded-api-key.yaml`)

**Problem:**
- Only matched specific regex patterns (AWS keys, Stripe keys, private keys)
- Didn't match plain password strings like `DATABASE_PASSWORD = "admin123!SuperSecret"`

**Test Code Pattern:**
```python
DATABASE_PASSWORD = "admin123!SuperSecret"
```

**Fixes Applied:**
- Added new rule `raptor.secrets.hardcoded.password` that matches:
  - Variables with names containing PASSWORD, SECRET, KEY, TOKEN, or API_KEY (case-insensitive)
  - Assigned to string literals (not environment variables)

## Testing the Rules

To verify the rules work, you can test them directly:

```bash
# From project root
cd test/data

# Test SQL injection rule
semgrep --config=../../engine/semgrep/rules/injection/sql-concat.yaml python_sql_injection.py

# Test command injection rule
semgrep --config=../../engine/semgrep/rules/injection/command-taint.yaml python_sql_injection.py

# Test secrets rule
semgrep --config=../../engine/semgrep/rules/secrets/hardcoded-api-key.yaml python_sql_injection.py

# Test all rules at once
semgrep --config=../../engine/semgrep/rules/ python_sql_injection.py
```

## Expected Results

After these fixes, Semgrep should detect:
1. ✅ SQL injection in line 19: `query = "SELECT * FROM users WHERE id = " + user_id`
2. ✅ Command injection in line 58: `subprocess.run(f"convert {filename} output.jpg", shell=True)`
3. ✅ Hardcoded password in line 26: `DATABASE_PASSWORD = "admin123!SuperSecret"`

## Notes

- The taint rules require Semgrep's taint analysis mode, which tracks data flow from sources to sinks
- Pattern-based rules are simpler but may have more false positives
- The combination of both approaches provides better coverage

## Next Steps

1. Run the test script to verify Semgrep now detects vulnerabilities:
   ```bash
   bash test/truly_real_tests.sh
   ```

2. If rules still don't match, check Semgrep output for specific pattern matching issues:
   ```bash
   semgrep --config=../../engine/semgrep/rules/injection/sql-concat.yaml python_sql_injection.py --debug
   ```

3. Consider adding more test cases to ensure rules work across different code patterns
