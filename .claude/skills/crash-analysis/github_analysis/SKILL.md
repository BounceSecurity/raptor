---
name: Github Archive Analysis
description: Analyze OSS Github repositories using Github historical archive
version: 1.0
author: mbrg
tags:
  - github
  - git
  - forensics
---

# GitHub Forensics using GH Archive 

## Overview

GH Archive is an immutable record of public GitHub activity. It's an OSS project that stores GitHub log events for all public repositories. Unlike local git repositories (which can be rewritten), GitHub Archive provides tamper-proof evidence of GitHub events as they occurred. We are aware of rare cases where events were missing from the GH Archive, so the absence of event records is not evidence that something did not occur. GH Archive is available to query via Google BigQuery (requires Google Cloud credentials).

## Core Principles

**FORENSIC INTEGRITY PROTOCOL**: GitHub Archive, accessed via BigQuery, is an immutable record of all public GitHub activity. Unlike local git repositories (which can be rewritten), GitHub Archive provides tamper-proof evidence of GitHub events as they occurred.

**ALWAYS PREFER GitHub Archive as forensic evidence over**:
- Local git command outputs (git log, git show) - commits can be backdated/forged
- Unverified claims from articles or reports - require independent confirmation
- GitHub web interface screenshots - can be manipulated
- Single-source evidence - always cross-verify

**GitHub Archive IS your ground truth for**:
- Actor attribution (who performed actions)
- Timeline reconstruction (when events occurred)
- Event verification (what actually happened)
- Pattern analysis (behavioral fingerprinting)
- Cross-repository activity tracking
- **Deleted content recovery** (issues, PRs, tags, commits references remain in archive)
- **Repository deletion forensics** (commit SHAs persist even after repo deletion and history rewrites)

## GitHub Archive Infrastructure

### Access via BigQuery

GitHub Archive data is available through Google Cloud BigQuery. To query it, you need:

1. **Google Cloud Credentials**: Service account JSON file with BigQuery access
2. **Project Setup**: BigQuery client initialized with credentials
3. **Free Tier**: 1 TB of data processed per month free

**Standard Setup Pattern**:
```python
from google.cloud import bigquery
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    'path/to/credentials.json',
    scopes=['https://www.googleapis.com/auth/bigquery']
)

client = bigquery.Client(
    credentials=credentials,
    project=credentials.project_id
)
```

### Table Organization

**Dataset**: `githubarchive`

**Table Patterns**:
- **Daily tables**: `githubarchive.day.YYYYMMDD` (e.g., `githubarchive.day.20250713`)
- **Monthly tables**: `githubarchive.month.YYYYMM` (e.g., `githubarchive.month.202507`)
- **Yearly tables**: `githubarchive.year.YYYY` (e.g., `githubarchive.year.2025`)

**Wildcard Patterns**:
- All days in June 2025: `githubarchive.day.202506*`
- All months in 2025: `githubarchive.month.2025*`
- All data in 2025: `githubarchive.year.2025*`

**Data Availability**: February 12, 2011 to present (updated hourly)

### Schema Structure

**Top-Level Fields**:
```sql
type              -- Event type (PushEvent, IssuesEvent, etc.)
created_at        -- Timestamp when event occurred (UTC)
actor.login       -- GitHub username who performed the action
actor.id          -- GitHub user ID
repo.name         -- Repository name (org/repo format)
repo.id           -- Repository ID
org.login         -- Organization login (if applicable)
org.id            -- Organization ID
payload           -- JSON string with event-specific data
```

**Payload Field**: JSON-encoded string containing event-specific details. Must be parsed with `JSON_EXTRACT_SCALAR()` or loaded with `json.loads()` in Python.

## Event Types Reference

### Repository Events

**PushEvent** - Commits pushed to a repository
```sql
-- Payload fields:
JSON_EXTRACT_SCALAR(payload, '$.ref')        -- Branch (refs/heads/master)
JSON_EXTRACT_SCALAR(payload, '$.before')     -- SHA before push
JSON_EXTRACT_SCALAR(payload, '$.after')      -- SHA after push
JSON_EXTRACT_SCALAR(payload, '$.size')       -- Number of commits
-- payload.commits[] contains array of commit objects with sha, message, author
```

**PullRequestEvent** - Pull request opened, closed, merged
```sql
-- Payload fields:
JSON_EXTRACT_SCALAR(payload, '$.action')              -- opened, closed, merged
JSON_EXTRACT_SCALAR(payload, '$.pull_request.number')
JSON_EXTRACT_SCALAR(payload, '$.pull_request.title')
JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') -- true/false
```

**CreateEvent** - Branch or tag created
```sql
-- Payload fields:
JSON_EXTRACT_SCALAR(payload, '$.ref_type')   -- branch, tag, repository
JSON_EXTRACT_SCALAR(payload, '$.ref')        -- Name of branch/tag
```

**DeleteEvent** - Branch or tag deleted
```sql
-- Payload fields:
JSON_EXTRACT_SCALAR(payload, '$.ref_type')   -- branch or tag
JSON_EXTRACT_SCALAR(payload, '$.ref')        -- Name of deleted ref
```

**ForkEvent** - Repository forked
```sql
-- Payload fields:
JSON_EXTRACT_SCALAR(payload, '$.forkee.full_name')  -- New fork name
```

### Automation & CI/CD Events

**WorkflowRunEvent** - GitHub Actions workflow run status changes
```sql
-- Payload fields:
JSON_EXTRACT_SCALAR(payload, '$.action')               -- requested, completed
JSON_EXTRACT_SCALAR(payload, '$.workflow_run.name')
JSON_EXTRACT_SCALAR(payload, '$.workflow_run.path')    -- .github/workflows/file.yml
JSON_EXTRACT_SCALAR(payload, '$.workflow_run.status')  -- queued, in_progress, completed
JSON_EXTRACT_SCALAR(payload, '$.workflow_run.conclusion') -- success, failure, cancelled
JSON_EXTRACT_SCALAR(payload, '$.workflow_run.head_sha')
JSON_EXTRACT_SCALAR(payload, '$.workflow_run.head_branch')
```

**WorkflowJobEvent** - Individual job within workflow
**CheckRunEvent** - Check run status (CI systems)
**CheckSuiteEvent** - Check suite for commits

### Issue & Discussion Events

**IssuesEvent** - Issue opened, closed, edited
```sql
-- Payload fields:
JSON_EXTRACT_SCALAR(payload, '$.action')        -- opened, closed, reopened
JSON_EXTRACT_SCALAR(payload, '$.issue.number')
JSON_EXTRACT_SCALAR(payload, '$.issue.title')
JSON_EXTRACT_SCALAR(payload, '$.issue.body')
```

**IssueCommentEvent** - Comment on issue or pull request

**PullRequestReviewEvent** - PR review submitted
**PullRequestReviewCommentEvent** - Comment on PR diff

### Other Events

**WatchEvent** - Repository starred
**ReleaseEvent** - Release published
**MemberEvent** - Collaborator added/removed
**PublicEvent** - Repository made public

## Critical Forensic Capability: Deleted Content Persistence

**"GitHub Never Forgets"** - One of the most powerful forensic capabilities of GitHub Archive is that **deleted content remains queryable indefinitely**. This creates an immutable audit trail independent of repository or account deletion attempts.

### What Persists After Deletion

**Deleted Issues & PRs**:
- Issue creation events (IssuesEvent) remain in archive
- Issue comments (IssueCommentEvent) remain accessible
- PR open/close/merge events (PullRequestEvent) persist
- **Forensic Value**: Recover deleted evidence of social engineering, reconnaissance, or coordination

**Deleted Tags & Branches**:
- CreateEvent records for tag/branch creation persist
- DeleteEvent records document when deletion occurred
- **Forensic Value**: Reconstruct attack staging infrastructure (e.g., malicious payload delivery tags)

**Deleted Repositories**:
- All PushEvents to the repository remain queryable
- Commit SHAs are permanently recorded in archive
- Fork relationships (ForkEvent) survive deletion
- **Forensic Value**: Access commit metadata even after threat actor deletes evidence

**Deleted User Accounts**:
- All activity events remain attributed to deleted username
- Timeline reconstruction remains possible
- **Limitation**: Direct code access lost, but commit SHAs can be searched elsewhere

### Real-World Investigation Pattern: Deleted PR Analysis

**Scenario**: Attacker claims to have submitted a PR in "late June" containing malicious code, but PR is now deleted.

**Forensic Approach**:
```sql
-- Search for ALL PR events by suspected actor in June 2025
SELECT
    type,
    created_at,
    repo.name,
    payload
FROM `githubarchive.day.202506*`
WHERE
    actor.login = 'suspected-actor'
    AND repo.name = 'target/repository'
    AND type = 'PullRequestEvent'
ORDER BY created_at
```

**Evidence Validation**:
- If claim is TRUE: Archive will show PullRequestEvent with action='opened'
- If claim is FALSE: No events found, claim is disproven
- **Investigation Outcome**: Can definitively verify or refute attacker's timeline claims

**Real Example**: Amazon Q investigation verified no PR from lkmanka58 in late June 2025, disproving hacker's claim of receiving admin credentials via deleted PR.

### Real-World Investigation Pattern: Deleted Repository Forensics

**Scenario**: Threat actor creates staging repository, pushes malicious code, then deletes repo to cover tracks.

**Forensic Approach**:
```sql
-- Find repository creation and all push events
SELECT
    type,
    created_at,
    payload
FROM `githubarchive.day.2025*`
WHERE
    actor.login = 'threat-actor'
    AND type IN ('CreateEvent', 'PushEvent')
    AND (
        JSON_EXTRACT_SCALAR(payload, '$.repository.name') = 'staging-repo'
        OR repo.name LIKE 'threat-actor/staging-repo'
    )
ORDER BY created_at
```

**Evidence Recovery**:
- CreateEvent reveals repository creation timestamp
- PushEvents contain commit SHAs and metadata
- Commit SHAs can be used to recover code content via other archives or forks
- **Investigation Outcome**: Complete reconstruction of attacker's staging infrastructure

**Real Example**: lkmanka58/code_whisperer repository deleted after attack, but GitHub Archive revealed June 13 creation with 3 commits containing AWS IAM role assumption attempts.

### Real-World Investigation Pattern: Deleted Tag Analysis

**Scenario**: Malicious tag used for payload delivery, then deleted to hide evidence.

**Forensic Approach**:
```sql
-- Search for tag creation and deletion events
SELECT
    type,
    created_at,
    actor.login,
    payload
FROM `githubarchive.day.20250713`
WHERE
    repo.name = 'target/repository'
    AND type IN ('CreateEvent', 'DeleteEvent')
    AND JSON_EXTRACT_SCALAR(payload, '$.ref_type') = 'tag'
ORDER BY created_at
```

**Timeline Reconstruction**:
```json
{
  "19:41:44 UTC": "CreateEvent - tag 'stability' created by aws-toolkit-automation",
  "20:30:24 UTC": "PushEvent - malicious commit references tag",
  "Next day": "DeleteEvent - tag 'stability' deleted (cleanup attempt)"
}
```

**Real Example**: Amazon Q attack used 'stability' tag for malicious payload delivery. Tag was deleted, but CreateEvent in GitHub Archive preserved creation timestamp and actor, proving 48-hour staging window.

### Attribution in Deleted Content Investigations

**Challenge**: Determining if actors are threat participants vs benign bystanders.

**Analysis Framework**:

1. **Temporal Correlation**: Does actor's activity align with attack timeline?
2. **Behavioral Analysis**: Normal pattern vs anomalous actions during incident?
3. **Technical Sophistication**: Actions require special knowledge/access?
4. **Cross-Repository Patterns**: Similar activity in other compromise attempts?

**Example Classification**:
- **Malicious Actor**: lkmanka58 - systematic preparation repo creation, coordinated timing
- **Compromised Automation**: aws-toolkit-automation - legitimate service account, abnormal API usage pattern
- **Uncertain Attribution**: atonaamz - workflow secret exposure, but unclear if deliberate or accidental

**Recommendation**: Document "confidence levels" for attribution:
- **CONFIRMED** (technical proof via commit signatures, logs)
- **HIGH** (strong behavioral + temporal correlation)
- **MEDIUM** (circumstantial technical evidence)
- **LOW** (temporal correlation only)
- **UNCERTAIN** (insufficient technical evidence)

### Time-Sensitive Investigation Tactics

**CRITICAL**: While GitHub Archive preserves event metadata, some content is only accessible before deletion:

**Act Quickly On**:
1. **Repository Code Content**: Clone repos immediately when investigating active threats
2. **Issue/PR Body Text**: Screenshot or save full text before deletion
3. **Workflow Logs**: May be purged after 90 days by GitHub
4. **Profile Information**: User bios, organizations, other repos

**Archive Preserves Forever**:
1. **Event timestamps and types**
2. **Actor attribution (usernames)**
3. **Repository names and references**
4. **Commit SHAs**
5. **Tag/branch names**
6. **Basic payload metadata**

**Investigation Protocol**:
```python
# When investigating suspicious activity:
# 1. Document everything NOW (screenshots, git clones)
# 2. Extract actor timelines from GitHub Archive
# 3. Cross-reference immediate documentation with archive records
# 4. Even if repos/accounts get deleted, Archive maintains evidence chain
```

**Real Example**: "Fortunately, I looked at it yesterday before it got deleted" - Investigator accessed lkmanka58/code_whisperer repo before account deletion, combining real-time documentation with GitHub Archive timeline proof.

## Common Query Patterns

### 1. Actor Timeline Reconstruction

Build complete activity timeline for a specific user:

```sql
SELECT
    type,
    created_at,
    repo.name as repo_name,
    payload
FROM `githubarchive.day.2025*`
WHERE
    actor.login = 'username'
ORDER BY created_at
LIMIT 1000
```

**Use Case**: Threat actor profiling, attribution analysis, behavioral fingerprinting

### 2. Repository Event Filtering

Get all activity for a specific repository in a time window:

```sql
SELECT
    type,
    created_at,
    actor.login as actor_login,
    payload
FROM `githubarchive.day.20250713`
WHERE
    repo.name = 'org/repository'
    AND type IN ('PushEvent', 'PullRequestEvent', 'IssuesEvent')
ORDER BY created_at
```

**Use Case**: Incident timeline reconstruction, compromise window analysis

### 3. Cross-Repository Pattern Analysis

Find an actor's activity across multiple repositories:

```sql
WITH all_events AS (
    SELECT
        created_at,
        repo.name,
        type,
        payload
    FROM `githubarchive.day.202506*`
    WHERE actor.login = 'username'

    UNION ALL

    SELECT
        created_at,
        repo.name,
        type,
        payload
    FROM `githubarchive.day.202507*`
    WHERE actor.login = 'username'
)
SELECT * FROM all_events
ORDER BY created_at
```

**Use Case**: Supply chain attack investigation, lateral movement detection

### 4. Workflow Execution Verification

Verify whether workflows ran during specific time windows:

```sql
SELECT
    type,
    created_at,
    actor.login,
    payload
FROM `githubarchive.day.20250713`
WHERE
    repo.name = 'org/repository'
    AND type IN ('WorkflowRunEvent', 'WorkflowJobEvent')
    AND created_at >= '2025-07-13T19:00:00'
    AND created_at <= '2025-07-13T21:00:00'
ORDER BY created_at
```

**Use Case**: API attack vs workflow attack differentiation, automation compromise analysis

### 5. Push Event Analysis

Examine all pushes to a specific branch:

```sql
SELECT
    created_at,
    actor.login,
    payload
FROM `githubarchive.day.202507*`
WHERE
    repo.name = 'org/repository'
    AND type = 'PushEvent'
    AND JSON_EXTRACT_SCALAR(payload, '$.ref') = 'refs/heads/master'
ORDER BY created_at
```

**Use Case**: Unauthorized commit detection, access control verification

### 6. Multi-Day Event Aggregation

Combine data across multiple days for critical windows:

```sql
WITH critical_window AS (
    SELECT * FROM `githubarchive.day.20250711`
    WHERE repo.name = 'org/repository'

    UNION ALL

    SELECT * FROM `githubarchive.day.20250712`
    WHERE repo.name = 'org/repository'

    UNION ALL

    SELECT * FROM `githubarchive.day.20250713`
    WHERE repo.name = 'org/repository'
)
SELECT
    type,
    created_at,
    actor.login,
    payload
FROM critical_window
ORDER BY created_at
```

**Use Case**: Attack campaign analysis, multi-day operation reconstruction

### 7. Actor Behavioral Analysis

Analyze activity patterns by event type:

```sql
SELECT
    actor.login,
    type,
    COUNT(*) as event_count,
    MIN(created_at) as first_seen,
    MAX(created_at) as last_seen
FROM `githubarchive.day.2025*`
WHERE
    actor.login = 'username'
GROUP BY actor.login, type
ORDER BY event_count DESC
```

**Use Case**: Normal vs anomalous behavior detection, capability assessment

### 8. Token Exposure Investigation

Search for workflow runs with potential token references:

```sql
SELECT
    type,
    created_at,
    actor.login,
    payload
FROM `githubarchive.day.202507*`
WHERE
    repo.name = 'org/repository'
    AND type IN ('WorkflowRunEvent', 'WorkflowJobEvent')
    AND (
        REGEXP_CONTAINS(payload, r'GITHUB_TOKEN')
        OR REGEXP_CONTAINS(payload, r'secrets')
        OR REGEXP_CONTAINS(payload, r'credentials')
    )
ORDER BY created_at
```

**Use Case**: Secret exposure analysis, token compromise investigation

## Python Integration Patterns

### Basic Query Execution

```python
def query_github_archive(query_string):
    """Execute BigQuery and return results"""
    job = client.query(query_string)
    results = list(job.result())
    return results

def save_results(results, output_path):
    """Save query results to JSON"""
    events = []
    for row in results:
        event = {
            'type': row.type,
            'created_at': row.created_at.isoformat() if row.created_at else None,
            'actor_login': row.actor_login,
            'repo_name': row.repo_name,
            'payload': json.loads(row.payload) if row.payload else {}
        }
        events.append(event)

    with open(output_path, 'w') as f:
        json.dump(events, f, indent=2)
```

### Payload Parsing

```python
def analyze_push_events(results):
    """Extract commit details from PushEvent payloads"""
    for row in results:
        if row.type != 'PushEvent':
            continue

        payload = json.loads(row.payload) if row.payload else {}

        ref = payload.get('ref')
        commits = payload.get('commits', [])

        print(f"\n{row.created_at} - Push by {row.actor_login}")
        print(f"  Branch: {ref}")

        for commit in commits:
            sha = commit.get('sha', '')[:7]
            message = commit.get('message', '').split('\n')[0]
            author = commit.get('author', {})

            print(f"  - {sha}: {message}")
            print(f"    Author: {author.get('name')} <{author.get('email')}>")
```

### Multi-Source Verification

```python
def verify_commit_in_archive(repo_name, commit_sha, date):
    """Verify commit exists in GitHub Archive"""
    query = f"""
    SELECT
        type,
        created_at,
        actor.login as actor_login,
        payload
    FROM `githubarchive.day.{date}`
    WHERE
        repo.name = '{repo_name}'
        AND type = 'PushEvent'
        AND payload LIKE '%{commit_sha[:7]}%'
    """

    job = client.query(query)
    results = list(job.result())

    if results:
        print(f"✅ Commit {commit_sha[:7]} VERIFIED in GitHub Archive")
        return True
    else:
        print(f"❌ Commit {commit_sha[:7]} NOT FOUND in GitHub Archive")
        return False
```

## Investigation Use Cases from Real Analysis

### Use Case 1: Threat Actor Attribution

**Objective**: Build complete operational timeline for suspected threat actor

**Approach**:
1. Query GitHub Archive for ALL activity by actor across all repositories
2. Analyze temporal patterns (timezone, working hours)
3. Identify repository targeting patterns
4. Map capability progression over time
5. Extract geographic/behavioral fingerprints

**Evidence Quality**: IMMUTABLE - Actor activity timestamps cannot be forged

**Example**: lkmanka58 16-month timeline reconstruction revealed systematic progression from blockchain experimentation to AWS supply chain compromise

### Use Case 2: Automation Compromise Verification

**Objective**: Determine if malicious commits came from workflows or direct API

**Approach**:
1. Extract exact commit timestamp from GitHub API
2. Query GitHub Archive for WorkflowRunEvent in ±10 minute window
3. Search for workflows triggered by specific commit SHA
4. Analyze workflow execution patterns on attack day
5. Compare with normal automation behavior baseline

**Evidence Quality**: DEFINITIVE - Workflow execution is recorded in archive

**Example**: Zero WorkflowRunEvent during malicious commit window proved direct API attack vector, not workflow-based compromise

### Use Case 3: Multi-Phase Attack Timeline

**Objective**: Reconstruct complete attack sequence across multiple days

**Approach**:
1. Query GitHub Archive for all events in critical time window
2. Filter by repository and relevant event types
3. Cross-reference with GitHub API for commit details
4. Build chronological timeline with all actors
5. Identify coordination patterns and staging windows

**Evidence Quality**: COMPREHENSIVE - Complete event record across attack timeline

**Example**: 48-hour staging window analysis revealed tag creation 12 hours before malicious commit, demonstrating deliberate preparation

### Use Case 4: Secret Exposure Investigation

**Objective**: Identify potential workflow log token exposure

**Approach**:
1. Query WorkflowRunEvent and WorkflowJobEvent for repository
2. Filter for workflows referencing tokens or secrets
3. Analyze workflow execution timing and actors
4. Cross-reference with known secret exposure patterns
5. Identify high-risk workflow configurations

**Evidence Quality**: PATTERN-BASED - Indicates potential exposure vectors

**Example**: Workflows with GITHUB_TOKEN in logs identified as potential compromise vector for aws-toolkit-automation token

### Use Case 5: Supply Chain Reconnaissance Detection

**Objective**: Identify systematic reconnaissance of development ecosystem

**Approach**:
1. Track actor activity across multiple related repositories
2. Identify issue creation and engagement patterns
3. Map repository fork and star activity
4. Analyze temporal clustering around specific technologies
5. Detect preparation repositories (e.g., code_whisperer)

**Evidence Quality**: BEHAVIORAL - Pattern indicates intent and preparation

**Example**: lkmanka58 progression from Microsoft vscode-python to aws-toolkit-vscode via code_whisperer repository creation revealed systematic targeting

## Cost Management Best Practices

**Query Optimization**:
- Use specific date ranges, not wildcards across years: `githubarchive.day.20250713` not `githubarchive.day.*`
- Filter early with WHERE clauses to reduce data scanned
- Use LIMIT to cap result sizes during exploration
- Leverage table previews before running expensive queries

**Data Reuse**:
- Download query results once and save to local files
- Reanalyze saved JSON files instead of re-querying
- Build incremental analysis on cached datasets
- Document queries to avoid duplicate investigations

**Incremental Analysis**:
```python
# Good: Query once, save results, analyze locally
results = query_github_archive(query)
save_results(results, 'data/actor_timeline.json')

# Then analyze the saved file multiple times
with open('data/actor_timeline.json') as f:
    data = json.load(f)
    analyze_pattern_1(data)
    analyze_pattern_2(data)
```

## Forensic Verification Standards

When using GitHub Archive data for security investigations, follow these verification protocols:

### Multi-Source Verification
✅ Cross-reference GitHub Archive findings with:
- GitHub API (for commit details, workflow logs)
- GitHub web interface (for visual confirmation)
- Local git repository analysis (for code content)
- AWS build logs (for infrastructure correlation)

### Timestamp Integrity
✅ Always verify timestamps across sources:
- `created_at` in GitHub Archive (event time)
- `commit.author.date` in git (claimed author time)
- `commit.committer.date` in git (actual commit time)
- GitHub API commit timestamp (immutable record)

### Chain of Custody Documentation
✅ Document every evidence extraction:
- Exact BigQuery query used
- Query execution timestamp
- Result count and date ranges
- Storage location of extracted data
- Hash of saved evidence files

### Reproducible Investigations
✅ All queries should be:
- Saved as documented Python scripts
- Include exact table references
- Specify date ranges explicitly
- Include comments explaining forensic logic
- Runnable by independent investigators

## Integration with GitHub API

**GitHub Archive Strengths**: Historical timeline, bulk analysis, actor tracking across repos

**GitHub API Strengths**: Current state, commit content, workflow logs, PR details

**Combined Investigation Pattern**:
1. Use GitHub Archive to identify suspicious events and time windows
2. Use GitHub API to retrieve detailed content for flagged items
3. Cross-verify timestamps between Archive and API
4. Use Archive as primary evidence, API for enrichment

**Example Workflow**:
```python
# Step 1: Find suspicious push in Archive
archive_results = query_github_archive("""
    SELECT created_at, actor.login, payload
    FROM `githubarchive.day.20250713`
    WHERE repo.name = 'aws/aws-toolkit-vscode'
    AND type = 'PushEvent'
    AND actor.login = 'suspicious-actor'
""")

# Step 2: Extract commit SHA from Archive payload
for row in archive_results:
    payload = json.loads(row.payload)
    for commit in payload.get('commits', []):
        commit_sha = commit['sha']

        # Step 3: Get full commit details from GitHub API
        api_url = f"https://api.github.com/repos/aws/aws-toolkit-vscode/commits/{commit_sha}"
        commit_details = requests.get(api_url).json()

        # Step 4: Cross-verify timestamp
        archive_time = row.created_at
        api_time = commit_details['commit']['committer']['date']

        print(f"Archive: {archive_time}")
        print(f"API: {api_time}")

        if archive_time != api_time:
            print("⚠️ TIMESTAMP DISCREPANCY DETECTED")
```

## Investigation Workflow Template

```python
#!/usr/bin/env python3
"""
GitHub Archive Investigation Template
Forensic analysis of [INCIDENT NAME]
"""

from google.cloud import bigquery
from google.oauth2 import service_account
import json
from datetime import datetime

# Setup
credentials = service_account.Credentials.from_service_account_file(
    'path/to/credentials.json',
    scopes=['https://www.googleapis.com/auth/bigquery']
)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

def investigate_actor(actor_login, start_date, end_date):
    """Complete activity timeline for suspected actor"""
    query = f"""
    SELECT
        type,
        created_at,
        repo.name as repo_name,
        payload
    FROM `githubarchive.day.{start_date.replace('-', '')}*`
    WHERE
        actor.login = '{actor_login}'
        AND created_at >= '{start_date}'
        AND created_at <= '{end_date}'
    ORDER BY created_at
    """

    job = client.query(query)
    results = list(job.result())

    # Save raw results
    events = []
    for row in results:
        events.append({
            'type': row.type,
            'created_at': row.created_at.isoformat() if row.created_at else None,
            'repo_name': row.repo_name,
            'payload': json.loads(row.payload) if row.payload else {}
        })

    output_path = f'data/{actor_login}_timeline.json'
    with open(output_path, 'w') as f:
        json.dump(events, f, indent=2)

    print(f"✅ Saved {len(events)} events to {output_path}")
    return events

def investigate_repository(repo_name, date):
    """All activity for repository on specific date"""
    query = f"""
    SELECT
        type,
        created_at,
        actor.login as actor_login,
        payload
    FROM `githubarchive.day.{date.replace('-', '')}`
    WHERE
        repo.name = '{repo_name}'
    ORDER BY created_at
    """

    job = client.query(query)
    results = list(job.result())

    events = []
    for row in results:
        events.append({
            'type': row.type,
            'created_at': row.created_at.isoformat() if row.created_at else None,
            'actor_login': row.actor_login,
            'payload': json.loads(row.payload) if row.payload else {}
        })

    output_path = f'data/{repo_name.replace("/", "_")}_{date}.json'
    with open(output_path, 'w') as f:
        json.dump(events, f, indent=2)

    print(f"✅ Saved {len(events)} events to {output_path}")
    return events

if __name__ == "__main__":
    # Investigation parameters
    ACTOR = 'suspicious-actor'
    REPO = 'org/repository'
    DATE = '2025-07-13'

    print(f"GitHub Archive Forensic Investigation")
    print(f"Actor: {ACTOR}")
    print(f"Repository: {REPO}")
    print(f"Date: {DATE}")
    print("="*60)

    # Run investigations
    actor_timeline = investigate_actor(ACTOR, DATE, DATE)
    repo_events = investigate_repository(REPO, DATE)

    print("\n📊 Investigation Complete")
    print(f"Actor events: {len(actor_timeline)}")
    print(f"Repository events: {len(repo_events)}")
```

## Lessons from Real-World Investigations

### Claim Verification: The "Late June PR" Example

**Investigation Challenge**: Hacker claimed receiving admin credentials via a PR submitted in "late June 2025" but PR was deleted.

**GitHub Archive Approach**:
```sql
-- Search for ALL PR activity in June 2025
SELECT
    type,
    created_at,
    actor.login,
    payload
FROM `githubarchive.day.202506*`
WHERE
    repo.name = 'aws/aws-toolkit-vscode'
    AND actor.login = 'lkmanka58'
    AND type IN ('PullRequestEvent', 'PullRequestReviewEvent', 'IssueCommentEvent')
ORDER BY created_at
```

**Investigation Outcome**:
- **ZERO PR events** from lkmanka58 in June 2025
- Claim **DEFINITIVELY DISPROVEN** via immutable archive
- First activity: July 3, 2025 (Microsoft vscode issues)
- Demonstrates power of archive for claim verification

**Key Lesson**: Use GitHub Archive to verify/refute claims about deleted activity. The absence of evidence IS evidence when source is immutable.

### Multi-Repository Pattern Detection

**Investigation Challenge**: Determine if attack was opportunistic or systematic targeting.

**GitHub Archive Approach**:
```sql
-- Complete activity timeline across ALL repositories
SELECT
    type,
    created_at,
    repo.name,
    payload
FROM `githubarchive.day.2025*`
WHERE
    actor.login = 'lkmanka58'
ORDER BY created_at
```

**Pattern Discovered**:
1. **Feb 2024**: Blockchain infrastructure (dymensionxyz/chain-registry)
2. **466-day gap**: Strategic dormancy period
3. **May 2025**: Microsoft vscode-python reconnaissance
4. **June 2025**: AWS code_whisperer preparation repository
5. **July 2025**: Attack execution on aws-toolkit-vscode

**Investigation Outcome**: Systematic 16-month supply chain targeting campaign, not isolated incident

**Key Lesson**: Cross-repository timeline reveals strategic patterns invisible from single-repo analysis.

### Deleted Repository Staging Infrastructure

**Investigation Challenge**: Attacker deleted code_whisperer repository. How to recover evidence?

**GitHub Archive Approach**:
```sql
-- Find all activity for deleted repository
SELECT
    type,
    created_at,
    payload
FROM `githubarchive.day.202506*`
WHERE
    repo.name = 'lkmanka58/code_whisperer'
    OR actor.login = 'lkmanka58'
ORDER BY created_at
```

**Evidence Recovered**:
- Repository created: June 13, 2025, 04:59 UTC
- 3 PushEvents over 6-hour window (04:59-10:25 UTC)
- Commit SHAs preserved in archive payload
- AWS IAM role assumption attempts in commit metadata

**Investigation Outcome**: Complete reconstruction of attack staging infrastructure despite deletion

**Key Lesson**: "Fortunately, I looked at it yesterday before it got deleted" - Document NOW, but Archive provides insurance even if you miss live evidence.

### Automation vs Direct API Attribution

**Investigation Challenge**: Did malicious commits come from compromised workflow or direct API abuse?

**GitHub Archive Approach**:
```sql
-- Search for workflows during commit window
SELECT type, created_at, actor.login, payload
FROM `githubarchive.day.20250713`
WHERE
    repo.name = 'aws/aws-toolkit-vscode'
    AND type IN ('WorkflowRunEvent', 'WorkflowJobEvent')
    AND created_at >= '2025-07-13T20:25:00'
    AND created_at <= '2025-07-13T20:35:00'
```

**Evidence Found**:
- **ZERO workflow events** during malicious commit time (20:30:24 UTC)
- Normal automation had 18 workflows on same day
- All normal workflows clustered in 20:48-21:02 UTC window
- 9+ hour abnormal gap in automation pattern

**Investigation Outcome**: Direct API attack proven via workflow absence

**Key Lesson**: Negative evidence (absence of expected workflows) can be as powerful as positive evidence when analyzing automation systems.

### Tag-Based Attack Staging Analysis

**Investigation Challenge**: Malicious tag 'stability' was deleted. When was it created and who created it?

**GitHub Archive Approach**:
```sql
-- Find tag creation event
SELECT type, created_at, actor.login, payload
FROM `githubarchive.day.20250713`
WHERE
    repo.name = 'aws/aws-toolkit-vscode'
    AND type = 'CreateEvent'
    AND JSON_EXTRACT_SCALAR(payload, '$.ref_type') = 'tag'
    AND JSON_EXTRACT_SCALAR(payload, '$.ref') = 'stability'
```

**Timeline Discovered**:
- **19:41:44 UTC**: Tag 'stability' created by aws-toolkit-automation
- **20:30:24 UTC**: Malicious commit pushed (48 minutes later)
- **Next day**: Tag deleted (cleanup attempt)

**Investigation Outcome**: 48-hour staging window identified, proving deliberate preparation phase

**Key Lesson**: CreateEvent + DeleteEvent correlation reveals attacker operational timeline and staging infrastructure.

### Attribution Confidence Levels

**Investigation Challenge**: Multiple actors involved - who is malicious vs compromised vs benign?

**GitHub Archive Analysis Framework**:

**lkmanka58 Attribution: CONFIRMED MALICIOUS**
- 16-month timeline showing systematic progression
- Preparation repository (code_whisperer) created before attack
- Coordinated psychological operations (distraction issues)
- **Confidence**: Technical + behavioral + temporal evidence

**aws-toolkit-automation Attribution: CONFIRMED COMPROMISED**
- Legitimate service account with normal workflow patterns
- Abnormal direct API usage during attack (no workflows)
- Created malicious infrastructure (stability tag)
- **Confidence**: Pattern disruption + technical evidence

**atonaamz Attribution: UNCERTAIN**
- Workflow secret potentially exposed credentials
- No other suspicious activity pattern detected
- Could be accidental exposure or deliberate collaboration
- **Confidence**: Single technical indicator, insufficient behavioral evidence

**Key Lesson**: Document confidence levels explicitly. Not all attribution is equal - some is definitive, some is circumstantial.

### The Power of Immutability for Public Verification

**Investigation Principle**: Publishing methodology enables community fact-checking.

**Why This Matters**:
1. **Reproducible Evidence**: Anyone can run same BigQuery queries
2. **Peer Review**: Community can verify or challenge findings
3. **Trust Building**: Transparent methodology increases credibility
4. **Error Correction**: Public scrutiny catches investigator mistakes

**Real Example**: Amazon Q investigation published with exact GitHub Archive queries, allowing independent verification of timeline claims.

**Key Lesson**: Share your BigQuery queries. Immutable data + reproducible methodology = verifiable forensics.

## Key Takeaways

1. **GitHub Archive is IMMUTABLE** - It's your ground truth for "what actually happened"
2. **Deletion doesn't destroy evidence** - Events persist forever even if content is deleted
3. **Always query by date** - Use day/month tables to minimize costs
4. **Parse payload fields** - Event-specific data is JSON-encoded in payload column
5. **Cross-verify findings** - Use GitHub API to enrich Archive data
6. **Save query results** - Download once, analyze many times
7. **Document everything** - Forensic investigations require reproducible evidence
8. **Think in events** - GitHub Archive records actions, not state
9. **Timestamp is king** - created_at field is your forensic anchor point
10. **Absence is evidence** - Missing expected events (workflows, PRs) proves claims false
11. **Act fast, but Archive has your back** - Document now, but Archive provides insurance
12. **Share your queries** - Reproducible methodology enables verification and trust

## When to Use This Skill

- Investigating security incidents involving GitHub repositories
- Building threat actor attribution profiles
- Verifying claims about repository activity
- Reconstructing attack timelines
- Analyzing automation system compromises
- Detecting supply chain reconnaissance
- Cross-repository behavioral analysis
- Workflow execution verification
- Pattern-based anomaly detection

GitHub Archive analysis should be your FIRST step in any GitHub-related security investigation. Start with the immutable record, then enrich with additional sources.


## To Learn More

Detailed information in case a drill-down is needed:
- About GH Archive https://www.gharchive.org/
- Full schema of GitHub events https://docs.github.com/en/rest/using-the-rest-api/github-event-types
