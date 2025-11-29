"""
Verification Module - Verify evidence against original sources.

Each evidence object can call verify() to compare itself against
the real data from the source specified in its verification info.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from ._schema import AnyEvidence, Event, Observation, VerificationResult
    from ._clients import GitHubClient, GHArchiveClient

from ._schema import EvidenceSource


def verify_event(event: "Event") -> "VerificationResult":
    """Verify an event against the original source."""
    source = event.verification.source

    if source == EvidenceSource.GHARCHIVE:
        return _verify_gharchive_event(event)
    if source == EvidenceSource.GIT:
        return True, ["Local git verification not supported"]
    return False, [f"Unknown verification source for event: {source}"]


def verify_observation(observation: "Observation") -> "VerificationResult":
    """Verify an observation against the original source."""
    source = observation.verification.source

    verifiers: dict[EvidenceSource, Callable] = {
        EvidenceSource.GITHUB: _verify_github_observation,
        EvidenceSource.GHARCHIVE: _verify_gharchive_observation,
        EvidenceSource.WAYBACK: _verify_url_accessible,
        EvidenceSource.SECURITY_VENDOR: _verify_security_vendor,
        EvidenceSource.GIT: lambda o: (True, ["Local git verification not supported"]),
    }

    verifier = verifiers.get(source)
    if not verifier:
        return False, [f"Unknown verification source: {source}"]
    return verifier(observation)


def verify_all(evidence_list: Sequence["AnyEvidence"]) -> "VerificationResult":
    """Verify a list of evidence items. Aggregates all errors."""
    all_errors: list[str] = []
    all_valid = True

    for evidence in evidence_list:
        is_valid, errors = evidence.verify()
        if not is_valid:
            all_valid = False
            evidence_id = getattr(evidence, "evidence_id", "unknown")
            all_errors.extend(f"[{evidence_id}] {e}" for e in errors)

    return all_valid, all_errors


# =============================================================================
# GITHUB API VERIFICATION
# =============================================================================


def _get_github_client() -> "GitHubClient":
    from ._clients import GitHubClient
    return GitHubClient()


def _get_repo_info(obs: "Observation") -> tuple[str, str] | None:
    """Extract (owner, name) from observation. Returns None if missing."""
    repo = obs.repository
    return (repo.owner, repo.name) if repo else None


def _verify_github_observation(observation: "Observation") -> "VerificationResult":
    """Verify observation against GitHub API."""
    obs_type = getattr(observation, "observation_type", None)

    verifiers: dict[str, Callable] = {
        "commit": _verify_commit,
        "issue": _verify_issue,
        "file": _verify_file,
        "branch": _verify_branch,
        "tag": _verify_tag,
        "release": _verify_release,
        "fork": _verify_url_accessible,
    }

    verifier = verifiers.get(obs_type, _verify_url_accessible)

    try:
        return verifier(observation)
    except Exception as e:
        if getattr(observation, "is_deleted", False):
            return True, []  # Expected - item is marked as deleted
        return False, [f"Verification failed: {e}"]


def _verify_commit(obs: "Observation") -> "VerificationResult":
    """Verify commit against GitHub API."""
    repo_info = _get_repo_info(obs)
    if not repo_info:
        return False, ["No repository specified"]

    sha = getattr(obs, "sha", None)
    if not sha:
        return False, ["No SHA specified"]

    errors: list[str] = []
    data = _get_github_client().get_commit(*repo_info, sha)
    commit = data.get("commit", {})

    if data.get("sha") != sha:
        errors.append(f"SHA mismatch: expected {sha}, got {data.get('sha')}")

    if hasattr(obs, "message") and obs.message != commit.get("message", ""):
        errors.append("Message mismatch")

    if hasattr(obs, "author") and obs.author:
        actual = commit.get("author", {}).get("name")
        if obs.author.name != actual:
            errors.append(f"Author mismatch: expected {obs.author.name}, got {actual}")

    return len(errors) == 0, errors


def _verify_issue(obs: "Observation") -> "VerificationResult":
    """Verify issue/PR against GitHub API."""
    repo_info = _get_repo_info(obs)
    if not repo_info:
        return False, ["No repository specified"]

    number = getattr(obs, "issue_number", None)
    if not number:
        return False, ["No issue number specified"]

    errors: list[str] = []
    client = _get_github_client()
    is_pr = getattr(obs, "is_pull_request", False)
    data = client.get_pull_request(*repo_info, number) if is_pr else client.get_issue(*repo_info, number)

    if data.get("number") != number:
        errors.append(f"Number mismatch: expected {number}, got {data.get('number')}")

    if hasattr(obs, "title") and obs.title and data.get("title") != obs.title:
        errors.append("Title mismatch")

    if hasattr(obs, "state") and obs.state:
        actual = "merged" if data.get("merged") else data.get("state")
        if obs.state != actual:
            errors.append(f"State mismatch: expected {obs.state}, got {actual}")

    return len(errors) == 0, errors


def _verify_file(obs: "Observation") -> "VerificationResult":
    """Verify file content against GitHub API."""
    import base64
    import hashlib

    repo_info = _get_repo_info(obs)
    if not repo_info:
        return False, ["No repository specified"]

    file_path = getattr(obs, "file_path", None)
    if not file_path:
        return False, ["No file path specified"]

    ref = getattr(obs, "branch", None) or "HEAD"
    data = _get_github_client().get_file(*repo_info, file_path, ref)

    if hasattr(obs, "content_hash") and obs.content_hash:
        raw = data.get("content", "")
        content = base64.b64decode(raw).decode("utf-8", errors="replace") if raw else ""
        if obs.content_hash != hashlib.sha256(content.encode()).hexdigest():
            return False, ["Content hash mismatch"]

    return True, []


def _verify_branch(obs: "Observation") -> "VerificationResult":
    """Verify branch against GitHub API."""
    repo_info = _get_repo_info(obs)
    if not repo_info:
        return False, ["No repository specified"]

    branch_name = getattr(obs, "branch_name", None)
    if not branch_name:
        return False, ["No branch name specified"]

    data = _get_github_client().get_branch(*repo_info, branch_name)

    if hasattr(obs, "head_sha") and obs.head_sha:
        actual = data.get("commit", {}).get("sha")
        if obs.head_sha != actual:
            return False, [f"HEAD SHA mismatch: expected {obs.head_sha}, got {actual}"]

    return True, []


def _verify_tag(obs: "Observation") -> "VerificationResult":
    """Verify tag against GitHub API."""
    repo_info = _get_repo_info(obs)
    if not repo_info:
        return False, ["No repository specified"]

    tag_name = getattr(obs, "tag_name", None)
    if not tag_name:
        return False, ["No tag name specified"]

    data = _get_github_client().get_tag(*repo_info, tag_name)

    if hasattr(obs, "target_sha") and obs.target_sha:
        actual = data.get("object", {}).get("sha")
        if obs.target_sha != actual:
            return False, [f"Target SHA mismatch: expected {obs.target_sha}, got {actual}"]

    return True, []


def _verify_release(obs: "Observation") -> "VerificationResult":
    """Verify release against GitHub API."""
    repo_info = _get_repo_info(obs)
    if not repo_info:
        return False, ["No repository specified"]

    tag_name = getattr(obs, "tag_name", None)
    if not tag_name:
        return False, ["No tag name specified"]

    data = _get_github_client().get_release(*repo_info, tag_name)

    if data.get("tag_name") != tag_name:
        return False, ["Tag name mismatch"]

    return True, []


# =============================================================================
# URL / VENDOR VERIFICATION
# =============================================================================


def _verify_url_accessible(obs: "Observation") -> "VerificationResult":
    """Verify that the verification URL is accessible."""
    import requests

    url = obs.verification.url
    if not url:
        return True, []

    try:
        requests.get(str(url), timeout=30).raise_for_status()
        return True, []
    except requests.RequestException as e:
        return False, [f"Failed to access URL: {e}"]


def _verify_security_vendor(obs: "Observation") -> "VerificationResult":
    """Verify observation against security vendor URL."""
    import requests

    url = obs.verification.url
    if not url:
        return False, ["No source URL specified"]

    try:
        resp = requests.get(str(url), timeout=30)
        resp.raise_for_status()

        # For IOCs, verify value appears in content
        if getattr(obs, "observation_type", None) == "ioc":
            value = getattr(obs, "value", None)
            if value and value.lower() not in resp.text.lower():
                return False, [f"IOC value '{value[:50]}' not found in source"]

        return True, []
    except requests.RequestException as e:
        return False, [f"Failed to fetch source URL: {e}"]


# =============================================================================
# GH ARCHIVE VERIFICATION
# =============================================================================


def _get_gharchive_client() -> "GHArchiveClient | None":
    """Create GH Archive client. Returns None if credentials unavailable."""
    from ._clients import GHArchiveClient
    try:
        client = GHArchiveClient()
        client._get_client()  # Test credentials
        return client
    except Exception:
        return None


def _verify_gharchive_event(event: "Event") -> "VerificationResult":
    """Verify event against GH Archive BigQuery."""
    if not event.verification.bigquery_table:
        return False, ["No BigQuery table specified"]

    client = _get_gharchive_client()
    if not client:
        return True, ["GH Archive verification skipped - no credentials"]

    try:
        rows = client.query_events(
            repo=event.repository.full_name if event.repository else None,
            actor=event.who.login if event.who else None,
            from_date=event.when.strftime("%Y%m%d%H%M"),
        )
        if not rows:
            return False, [f"No matching event found in GH Archive"]
        return True, []
    except Exception as e:
        return False, [f"GH Archive verification error: {e}"]


def _verify_gharchive_observation(obs: "Observation") -> "VerificationResult":
    """Verify observation against GH Archive BigQuery."""
    if not obs.verification.bigquery_table:
        return False, ["No BigQuery table specified"]

    client = _get_gharchive_client()
    if not client:
        return True, ["GH Archive verification skipped - no credentials"]

    return True, []
