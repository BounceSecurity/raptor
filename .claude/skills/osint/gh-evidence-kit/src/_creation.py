"""
GitHub Forensics Evidence Creation Functions (OSINT)

Factory functions for creating verified evidence objects from public sources.
Consumer provides identifiers + source, we look up and verify independently.

All sources are public - no authentication required:
- GHArchive: BigQuery for Events (immutable, free 1TB/month)
- GitHub: REST API for Observations (60 req/hr unauthenticated)
- Wayback: CDX API for archived Observations (public)
- Git: Local git commands for Events
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import HttpUrl

from ._helpers import (
    generate_evidence_id as _generate_evidence_id,
    make_actor as _make_github_actor,
    make_repo as _make_github_repo,
    parse_datetime_strict as _parse_datetime,
)
from ._queries import (
    BranchQuery,
    CommitQuery,
    FileQuery,
    ForkQuery,
    IssueQuery,
    ReleaseQuery,
    RepositoryQuery,
    TagQuery,
    WaybackQuery,
)
from ._schema import (
    AnyEvent,
    ArticleObservation,
    BranchObservation,
    CommitAuthor,
    CommitObservation,
    EvidenceSource,
    FileChange,
    FileObservation,
    ForkObservation,
    GitHubActor,
    GitHubRepository,
    IOC,
    IOCType,
    IssueObservation,
    ReleaseObservation,
    SnapshotObservation,
    TagObservation,
    VerificationInfo,
    WaybackSnapshot,
)

# Import clients from dedicated module
from ._clients import GHArchiveClient, GitHubClient, WaybackClient

# Import event parsers from dedicated module
from ._parsers import parse_gharchive_event




# =============================================================================
# GH ARCHIVE RECOVERY FUNCTIONS
# =============================================================================


def create_issue_observation_from_gharchive(
    repo: str,
    issue_number: int,
    timestamp: str,
    client: GHArchiveClient,
) -> IssueObservation:
    """Recover deleted issue content from GH Archive.

    Args:
        repo: Full repo name (owner/repo)
        issue_number: Issue number
        timestamp: ISO timestamp when event occurred (e.g. "2025-07-13T20:30:24Z")
        client: GH Archive BigQuery client

    Raises ValueError if issue not found at specified timestamp.
    """
    owner, name = repo.split("/", 1)
    date = timestamp[:10].replace("-", "")

    rows = client.query_events(
        repo=repo,
        event_type="IssuesEvent",
        from_date=date,
    )

    for row in rows:
        payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        issue = payload.get("issue", {})
        row_ts = str(row.get("created_at", ""))

        if issue.get("number") == issue_number and timestamp in row_ts:
            state = issue.get("state", "open")
            return IssueObservation(
                evidence_id=_generate_evidence_id("issue-gharchive", repo, str(issue_number), timestamp),
                original_when=_parse_datetime(issue.get("created_at")),
                original_who=_make_github_actor(issue.get("user", {}).get("login", row["actor_login"])),
                original_what=f"Issue #{issue_number} created",
                observed_when=_parse_datetime(row["created_at"]),
                observed_by=EvidenceSource.GHARCHIVE,
                observed_what=f"Issue #{issue_number} recovered from GH Archive",
                repository=_make_github_repo(owner, name),
                verification=VerificationInfo(
                    source=EvidenceSource.GHARCHIVE,
                    bigquery_table=f"githubarchive.day.{date}",
                    query=f"repo.name='{repo}' AND type='IssuesEvent' AND created_at='{timestamp}'",
                ),
                issue_number=issue_number,
                is_pull_request=False,
                title=issue.get("title"),
                body=issue.get("body"),
                state=state,
                is_deleted=True,
            )

    raise ValueError(f"Issue #{issue_number} not found in GH Archive for {repo} at {timestamp}")


def create_pr_observation_from_gharchive(
    repo: str,
    pr_number: int,
    timestamp: str,
    client: GHArchiveClient,
) -> IssueObservation:
    """Recover deleted PR content from GH Archive.

    Args:
        repo: Full repo name (owner/repo)
        pr_number: PR number
        timestamp: ISO timestamp when event occurred
        client: GH Archive BigQuery client

    Raises ValueError if PR not found at specified timestamp.
    """
    owner, name = repo.split("/", 1)
    date = timestamp[:10].replace("-", "")

    rows = client.query_events(
        repo=repo,
        event_type="PullRequestEvent",
        from_date=date,
    )

    for row in rows:
        payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        pr = payload.get("pull_request", {})
        row_ts = str(row.get("created_at", ""))

        if pr.get("number") == pr_number and timestamp in row_ts:
            state = pr.get("state", "open")
            if pr.get("merged"):
                state = "merged"
            return IssueObservation(
                evidence_id=_generate_evidence_id("pr-gharchive", repo, str(pr_number), timestamp),
                original_when=_parse_datetime(pr.get("created_at")),
                original_who=_make_github_actor(pr.get("user", {}).get("login", row["actor_login"])),
                original_what=f"PR #{pr_number} created",
                observed_when=_parse_datetime(row["created_at"]),
                observed_by=EvidenceSource.GHARCHIVE,
                observed_what=f"PR #{pr_number} recovered from GH Archive",
                repository=_make_github_repo(owner, name),
                verification=VerificationInfo(
                    source=EvidenceSource.GHARCHIVE,
                    bigquery_table=f"githubarchive.day.{date}",
                    query=f"repo.name='{repo}' AND type='PullRequestEvent' AND created_at='{timestamp}'",
                ),
                issue_number=pr_number,
                is_pull_request=True,
                title=pr.get("title"),
                body=pr.get("body"),
                state=state,
                is_deleted=True,
            )

    raise ValueError(f"PR #{pr_number} not found in GH Archive for {repo} at {timestamp}")


def create_commit_observation_from_gharchive(
    repo: str,
    sha: str,
    timestamp: str,
    client: GHArchiveClient,
) -> CommitObservation:
    """Recover commit metadata from GH Archive.

    Args:
        repo: Full repo name (owner/repo)
        sha: Commit SHA (full or prefix)
        timestamp: ISO timestamp when push event occurred
        client: GH Archive BigQuery client

    Raises ValueError if commit not found at specified timestamp.
    """
    owner, name = repo.split("/", 1)
    date = timestamp[:10].replace("-", "")

    rows = client.query_events(
        repo=repo,
        event_type="PushEvent",
        from_date=date,
    )

    for row in rows:
        row_ts = str(row.get("created_at", ""))
        if timestamp not in row_ts:
            continue

        payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        for commit in payload.get("commits", []):
            if commit["sha"].startswith(sha) or sha.startswith(commit["sha"]):
                return CommitObservation(
                    evidence_id=_generate_evidence_id("commit-gharchive", repo, commit["sha"]),
                    original_when=_parse_datetime(row["created_at"]),
                    original_who=GitHubActor(login=commit.get("author", {}).get("name", "")),
                    original_what=commit.get("message", "").split("\n")[0],
                    observed_when=_parse_datetime(row["created_at"]),
                    observed_by=EvidenceSource.GHARCHIVE,
                    observed_what=f"Commit {commit['sha'][:8]} recovered from GH Archive",
                    repository=_make_github_repo(owner, name),
                    verification=VerificationInfo(
                        source=EvidenceSource.GHARCHIVE,
                        bigquery_table=f"githubarchive.day.{date}",
                        query=f"repo.name='{repo}' AND type='PushEvent' AND created_at='{timestamp}'",
                    ),
                    sha=commit["sha"],
                    message=commit.get("message", ""),
                    author=CommitAuthor(
                        name=commit.get("author", {}).get("name", ""),
                        email=commit.get("author", {}).get("email", ""),
                        date=_parse_datetime(row["created_at"]),
                    ),
                    committer=CommitAuthor(
                        name=commit.get("author", {}).get("name", ""),
                        email=commit.get("author", {}).get("email", ""),
                        date=_parse_datetime(row["created_at"]),
                    ),
                    parents=[],
                    files=[],
                    is_dangling=True,
                )

    raise ValueError(f"Commit {sha} not found in GH Archive for {repo} at {timestamp}")


def create_force_push_observation_from_gharchive(
    repo: str,
    timestamp: str,
    client: GHArchiveClient,
) -> CommitObservation:
    """Recover force-pushed commit from GH Archive.

    Args:
        repo: Full repo name (owner/repo)
        timestamp: ISO timestamp when force push occurred
        client: GH Archive BigQuery client

    Raises ValueError if no force push found at specified timestamp.
    """
    owner, name = repo.split("/", 1)
    date = timestamp[:10].replace("-", "")

    rows = client.query_events(
        repo=repo,
        event_type="PushEvent",
        from_date=date,
    )

    for row in rows:
        row_ts = str(row.get("created_at", ""))
        if timestamp not in row_ts:
            continue

        payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        size = int(payload.get("size", 0))
        before_sha = payload.get("before", "0" * 40)

        if size == 0 and before_sha != "0" * 40:
            return CommitObservation(
                evidence_id=_generate_evidence_id("forcepush-gharchive", repo, before_sha, timestamp),
                original_when=_parse_datetime(row["created_at"]),
                original_who=_make_github_actor(row["actor_login"]),
                original_what="Commit overwritten by force push",
                observed_when=_parse_datetime(row["created_at"]),
                observed_by=EvidenceSource.GHARCHIVE,
                observed_what=f"Force push detected, before SHA: {before_sha[:8]}",
                repository=_make_github_repo(owner, name),
                verification=VerificationInfo(
                    source=EvidenceSource.GHARCHIVE,
                    bigquery_table=f"githubarchive.day.{date}",
                    query=f"repo.name='{repo}' AND type='PushEvent' AND created_at='{timestamp}' AND size=0",
                ),
                sha=before_sha,
                message="[Force pushed - fetch content via GitHub API]",
                author=CommitAuthor(
                    name="unknown",
                    email="unknown",
                    date=_parse_datetime(row["created_at"]),
                ),
                committer=CommitAuthor(
                    name="unknown",
                    email="unknown",
                    date=_parse_datetime(row["created_at"]),
                ),
                parents=[],
                files=[],
                is_dangling=True,
            )

    raise ValueError(f"Force push not found in GH Archive for {repo} at {timestamp}")


# =============================================================================
# GITHUB API OBSERVATION FUNCTIONS
# =============================================================================


def create_commit_observation(
    query: CommitQuery,
    client: GitHubClient,
    observed_when: datetime | None = None,
) -> CommitObservation:
    """Create CommitObservation by fetching from GitHub API."""
    data = client.get_commit(query.repo.owner, query.repo.name, query.sha)
    commit = data["commit"]
    now = observed_when or datetime.now(timezone.utc)

    files = []
    for f in data.get("files", []):
        files.append(
            FileChange(
                filename=f["filename"],
                status=f.get("status", "modified"),
                additions=f.get("additions", 0),
                deletions=f.get("deletions", 0),
                patch=f.get("patch"),
            )
        )

    author = commit["author"]
    committer = commit["committer"]

    # GitHub API may return author as None for commits without a linked GitHub account
    gh_author = data.get("author") or {}

    return CommitObservation(
        evidence_id=_generate_evidence_id("commit", query.repo.full_name, data["sha"]),
        original_when=_parse_datetime(committer.get("date")),
        original_who=_make_github_actor(gh_author.get("login", author.get("name", "unknown"))),
        original_what=commit.get("message", "").split("\n")[0],
        observed_when=now,
        observed_by=EvidenceSource.GITHUB,
        observed_what=f"Commit {data['sha'][:8]} observed via GitHub API",
        repository=_make_github_repo(query.repo.owner, query.repo.name),
        verification=VerificationInfo(
            source=EvidenceSource.GITHUB,
            url=HttpUrl(f"https://github.com/{query.repo.full_name}/commit/{data['sha']}"),
        ),
        sha=data["sha"],
        message=commit.get("message", ""),
        author=CommitAuthor(
            name=author.get("name", ""),
            email=author.get("email", ""),
            date=_parse_datetime(author.get("date")),
        ),
        committer=CommitAuthor(
            name=committer.get("name", ""),
            email=committer.get("email", ""),
            date=_parse_datetime(committer.get("date")),
        ),
        parents=[p["sha"] for p in data.get("parents", [])],
        files=files,
        is_dangling=False,
    )


def create_issue_observation(
    query: IssueQuery,
    client: GitHubClient,
    observed_when: datetime | None = None,
) -> IssueObservation:
    """Create IssueObservation by fetching from GitHub API."""
    if query.is_pull_request:
        data = client.get_pull_request(query.repo.owner, query.repo.name, query.number)
    else:
        data = client.get_issue(query.repo.owner, query.repo.name, query.number)

    now = observed_when or datetime.now(timezone.utc)
    state = data.get("state", "open")
    if data.get("merged"):
        state = "merged"

    return IssueObservation(
        evidence_id=_generate_evidence_id("issue", query.repo.full_name, str(query.number)),
        original_when=_parse_datetime(data.get("created_at")),
        original_who=_make_github_actor(data.get("user", {}).get("login", "unknown")),
        original_what=f"{'PR' if query.is_pull_request else 'Issue'} #{query.number} created",
        observed_when=now,
        observed_by=EvidenceSource.GITHUB,
        observed_what=f"{'PR' if query.is_pull_request else 'Issue'} #{query.number} observed via GitHub API",
        repository=_make_github_repo(query.repo.owner, query.repo.name),
        verification=VerificationInfo(
            source=EvidenceSource.GITHUB,
            url=HttpUrl(f"https://github.com/{query.repo.full_name}/{'pull' if query.is_pull_request else 'issues'}/{query.number}"),
        ),
        issue_number=query.number,
        is_pull_request=query.is_pull_request,
        title=data.get("title"),
        body=data.get("body"),
        state=state,
        is_deleted=False,
    )


def create_file_observation(
    query: FileQuery,
    client: GitHubClient,
    observed_when: datetime | None = None,
) -> FileObservation:
    """Create FileObservation by fetching from GitHub API."""
    import base64
    import hashlib as hl

    data = client.get_file(query.repo.owner, query.repo.name, query.path, query.ref)
    now = observed_when or datetime.now(timezone.utc)

    content = ""
    if data.get("content"):
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")

    content_hash = hl.sha256(content.encode()).hexdigest()

    return FileObservation(
        evidence_id=_generate_evidence_id("file", query.repo.full_name, query.path, query.ref),
        observed_when=now,
        observed_by=EvidenceSource.GITHUB,
        observed_what=f"File {query.path} at {query.ref} observed via GitHub API",
        repository=_make_github_repo(query.repo.owner, query.repo.name),
        verification=VerificationInfo(
            source=EvidenceSource.GITHUB,
            url=HttpUrl(f"https://github.com/{query.repo.full_name}/blob/{query.ref}/{query.path}"),
        ),
        file_path=query.path,
        branch=query.ref if query.ref != "HEAD" else None,
        content_hash=content_hash,
        size_bytes=data.get("size", 0),
    )


def create_branch_observation(
    query: BranchQuery,
    client: GitHubClient,
    observed_when: datetime | None = None,
) -> BranchObservation:
    """Create BranchObservation by fetching from GitHub API."""
    data = client.get_branch(query.repo.owner, query.repo.name, query.branch_name)
    now = observed_when or datetime.now(timezone.utc)

    return BranchObservation(
        evidence_id=_generate_evidence_id("branch", query.repo.full_name, query.branch_name),
        observed_when=now,
        observed_by=EvidenceSource.GITHUB,
        observed_what=f"Branch {query.branch_name} observed via GitHub API",
        repository=_make_github_repo(query.repo.owner, query.repo.name),
        verification=VerificationInfo(
            source=EvidenceSource.GITHUB,
            url=HttpUrl(f"https://github.com/{query.repo.full_name}/tree/{query.branch_name}"),
        ),
        branch_name=query.branch_name,
        head_sha=data.get("commit", {}).get("sha"),
        protected=data.get("protected", False),
    )


def create_tag_observation(
    query: TagQuery,
    client: GitHubClient,
    observed_when: datetime | None = None,
) -> TagObservation:
    """Create TagObservation by fetching from GitHub API."""
    data = client.get_tag(query.repo.owner, query.repo.name, query.tag_name)
    now = observed_when or datetime.now(timezone.utc)

    return TagObservation(
        evidence_id=_generate_evidence_id("tag", query.repo.full_name, query.tag_name),
        observed_when=now,
        observed_by=EvidenceSource.GITHUB,
        observed_what=f"Tag {query.tag_name} observed via GitHub API",
        repository=_make_github_repo(query.repo.owner, query.repo.name),
        verification=VerificationInfo(
            source=EvidenceSource.GITHUB,
            url=HttpUrl(f"https://github.com/{query.repo.full_name}/releases/tag/{query.tag_name}"),
        ),
        tag_name=query.tag_name,
        target_sha=data.get("object", {}).get("sha"),
    )


def create_release_observation(
    query: ReleaseQuery,
    client: GitHubClient,
    observed_when: datetime | None = None,
) -> ReleaseObservation:
    """Create ReleaseObservation by fetching from GitHub API."""
    data = client.get_release(query.repo.owner, query.repo.name, query.tag_name)
    now = observed_when or datetime.now(timezone.utc)

    return ReleaseObservation(
        evidence_id=_generate_evidence_id("release", query.repo.full_name, query.tag_name),
        observed_when=now,
        observed_by=EvidenceSource.GITHUB,
        observed_what=f"Release {query.tag_name} observed via GitHub API",
        repository=_make_github_repo(query.repo.owner, query.repo.name),
        verification=VerificationInfo(
            source=EvidenceSource.GITHUB,
            url=HttpUrl(f"https://github.com/{query.repo.full_name}/releases/tag/{query.tag_name}"),
        ),
        tag_name=query.tag_name,
        release_name=data.get("name"),
        release_body=data.get("body"),
        created_at=_parse_datetime(data.get("created_at")),
        published_at=_parse_datetime(data.get("published_at")),
        is_prerelease=data.get("prerelease", False),
        is_draft=data.get("draft", False),
    )


def create_fork_observations(
    query: ForkQuery,
    client: GitHubClient,
    observed_when: datetime | None = None,
) -> list[ForkObservation]:
    """Create ForkObservations by fetching from GitHub API."""
    data = client.get_forks(query.repo.owner, query.repo.name)
    now = observed_when or datetime.now(timezone.utc)

    observations = []
    for fork in data:
        observations.append(
            ForkObservation(
                evidence_id=_generate_evidence_id("fork", query.repo.full_name, fork["full_name"]),
                observed_when=now,
                observed_by=EvidenceSource.GITHUB,
                observed_what=f"Fork {fork['full_name']} observed via GitHub API",
                repository=_make_github_repo(query.repo.owner, query.repo.name),
                verification=VerificationInfo(
                    source=EvidenceSource.GITHUB,
                    url=HttpUrl(f"https://github.com/{fork['full_name']}"),
                ),
                fork_full_name=fork["full_name"],
                parent_full_name=query.repo.full_name,
                fork_owner=fork["owner"]["login"],
                fork_repo=fork["name"],
                forked_at=_parse_datetime(fork.get("created_at")),
            )
        )

    return observations


def create_snapshot_observation(
    query: WaybackQuery,
    client: WaybackClient,
    observed_when: datetime | None = None,
) -> SnapshotObservation:
    """Create SnapshotObservation by fetching from Wayback Machine."""
    snapshots_data = client.search_cdx(
        url=str(query.url),
        from_date=query.from_date,
        to_date=query.to_date,
    )
    now = observed_when or datetime.now(timezone.utc)

    snapshots = []
    for s in snapshots_data:
        snapshots.append(
            WaybackSnapshot(
                timestamp=s.get("timestamp", ""),
                original=s.get("original", ""),
                digest=s.get("digest", ""),
                mimetype=s.get("mimetype", ""),
            )
        )

    return SnapshotObservation(
        evidence_id=_generate_evidence_id("snapshot", str(query.url)),
        observed_when=now,
        observed_by=EvidenceSource.WAYBACK,
        observed_what=f"Found {len(snapshots)} Wayback snapshots for {query.url}",
        verification=VerificationInfo(
            source=EvidenceSource.WAYBACK,
            url=HttpUrl(f"https://web.archive.org/cdx/search/cdx?url={query.url}"),
        ),
        original_url=query.url,
        snapshots=snapshots,
        total_snapshots=len(snapshots),
    )


def create_ioc(
    ioc_type: IOCType,
    value: str,
    source_url: HttpUrl,
    extracted_from: str | None = None,
) -> IOC:
    """Create an IOC by verifying it exists in the vendor report.

    Fetches source_url and confirms IOC value appears in content.
    Raises ValueError if IOC cannot be verified at source.
    """
    import requests

    try:
        resp = requests.get(str(source_url), timeout=30)
        resp.raise_for_status()
        content = resp.text
    except Exception as e:
        raise ValueError(f"Failed to fetch source URL {source_url}: {e}")

    if value.lower() not in content.lower():
        raise ValueError(f"IOC value '{value[:50]}' not found in source {source_url}")

    now = datetime.now(timezone.utc)

    return IOC(
        evidence_id=_generate_evidence_id("ioc", ioc_type.value, value),
        observed_when=now,
        observed_by=EvidenceSource.SECURITY_VENDOR,
        observed_what=f"IOC {ioc_type.value}: {value[:50]}{'...' if len(value) > 50 else ''}",
        verification=VerificationInfo(
            source=EvidenceSource.SECURITY_VENDOR,
            url=source_url,
        ),
        ioc_type=ioc_type,
        value=value,
        first_seen=now,
        last_seen=now,
        extracted_from=extracted_from,
    )


def create_article_observation(
    url: str,
    title: str,
    author: str | None = None,
    published_date: datetime | None = None,
    source_name: str | None = None,
    summary: str | None = None,
    observed_when: datetime | None = None,
) -> ArticleObservation:
    """Create an ArticleObservation for a blog post or security report."""
    now = observed_when or datetime.now(timezone.utc)

    return ArticleObservation(
        evidence_id=_generate_evidence_id("article", url),
        observed_when=now,
        observed_by=EvidenceSource.SECURITY_VENDOR,
        observed_what=f"Article: {title[:50]}{'...' if len(title) > 50 else ''}",
        verification=VerificationInfo(
            source=EvidenceSource.SECURITY_VENDOR,
            url=HttpUrl(url),
        ),
        url=HttpUrl(url),
        title=title,
        author=author,
        published_date=published_date,
        source_name=source_name,
        summary=summary,
    )


# =============================================================================
# EVIDENCE FACTORY
# =============================================================================


class EvidenceFactory:
    """Factory for creating verified OSINT evidence objects.

    All data sources are public and require no authentication:
    - GitHub API: Public repos, commits, issues, PRs (60 req/hr)
    - Wayback Machine: Archived web pages
    - GH Archive: BigQuery (requires GCP project, free tier: 1TB/month)
    """

    def __init__(
        self,
        gharchive_credentials: str | None = None,
        gharchive_project: str | None = None,
    ):
        self._github_client: GitHubClient | None = None
        self._wayback_client: WaybackClient | None = None
        self._gharchive_client: GHArchiveClient | None = None
        self._gharchive_credentials = gharchive_credentials
        self._gharchive_project = gharchive_project

    @property
    def github(self) -> GitHubClient:
        if self._github_client is None:
            self._github_client = GitHubClient()
        return self._github_client

    @property
    def wayback(self) -> WaybackClient:
        if self._wayback_client is None:
            self._wayback_client = WaybackClient()
        return self._wayback_client

    @property
    def gharchive(self) -> GHArchiveClient:
        if self._gharchive_client is None:
            self._gharchive_client = GHArchiveClient(
                credentials_path=self._gharchive_credentials,
                project_id=self._gharchive_project,
            )
        return self._gharchive_client

    def commit(self, owner: str, repo: str, sha: str) -> CommitObservation:
        """Create verified CommitObservation."""
        query = CommitQuery(repo=RepositoryQuery(owner=owner, name=repo), sha=sha)
        return create_commit_observation(query, self.github)

    def issue(self, owner: str, repo: str, number: int) -> IssueObservation:
        """Create verified IssueObservation."""
        query = IssueQuery(repo=RepositoryQuery(owner=owner, name=repo), number=number)
        return create_issue_observation(query, self.github)

    def pull_request(self, owner: str, repo: str, number: int) -> IssueObservation:
        """Create verified PR observation (as IssueObservation)."""
        query = IssueQuery(repo=RepositoryQuery(owner=owner, name=repo), number=number, is_pull_request=True)
        return create_issue_observation(query, self.github)

    def file(self, owner: str, repo: str, path: str, ref: str = "HEAD") -> FileObservation:
        """Create verified FileObservation."""
        query = FileQuery(repo=RepositoryQuery(owner=owner, name=repo), path=path, ref=ref)
        return create_file_observation(query, self.github)

    def branch(self, owner: str, repo: str, branch_name: str) -> BranchObservation:
        """Create verified BranchObservation."""
        query = BranchQuery(repo=RepositoryQuery(owner=owner, name=repo), branch_name=branch_name)
        return create_branch_observation(query, self.github)

    def tag(self, owner: str, repo: str, tag_name: str) -> TagObservation:
        """Create verified TagObservation."""
        query = TagQuery(repo=RepositoryQuery(owner=owner, name=repo), tag_name=tag_name)
        return create_tag_observation(query, self.github)

    def release(self, owner: str, repo: str, tag_name: str) -> ReleaseObservation:
        """Create verified ReleaseObservation."""
        query = ReleaseQuery(repo=RepositoryQuery(owner=owner, name=repo), tag_name=tag_name)
        return create_release_observation(query, self.github)

    def forks(self, owner: str, repo: str) -> list[ForkObservation]:
        """Create verified ForkObservations."""
        query = ForkQuery(repo=RepositoryQuery(owner=owner, name=repo))
        return create_fork_observations(query, self.github)

    def wayback_snapshots(
        self,
        url: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> SnapshotObservation:
        """Create SnapshotObservation from Wayback Machine."""
        query = WaybackQuery(url=HttpUrl(url), from_date=from_date, to_date=to_date)
        return create_snapshot_observation(query, self.wayback)

    def events_from_gharchive(
        self,
        timestamp: str,
        repo: str | None = None,
        actor: str | None = None,
        event_type: str | None = None,
    ) -> list[AnyEvent]:
        """Query GH Archive and create Events.

        Args:
            timestamp: Specific time in YYYYMMDDHHMM format
            repo: Repository in "owner/name" format (required if no actor)
            actor: GitHub username (required if no repo)
            event_type: Filter by event type (e.g., "PushEvent")

        Raises:
            ValueError: If timestamp is not 12 digits or neither repo nor actor is specified
        """
        if len(timestamp) != 12 or not timestamp.isdigit():
            raise ValueError(f"timestamp must be YYYYMMDDHHMM format (12 digits), got: {timestamp}")

        if not repo and not actor:
            raise ValueError("Must specify at least 'repo' or 'actor' to avoid expensive full-table scans")

        rows = self.gharchive.query_events(
            repo=repo,
            actor=actor,
            event_type=event_type,
            from_date=timestamp,
            to_date=timestamp,
        )

        events = []
        for row in rows:
            try:
                events.append(parse_gharchive_event(row))
            except (KeyError, ValueError):
                continue

        return events

    def ioc(
        self,
        ioc_type: IOCType | str,
        value: str,
        source_url: str,
        extracted_from: str | None = None,
    ) -> IOC:
        """Create an IOC by verifying it exists in vendor report."""
        if isinstance(ioc_type, str):
            ioc_type = IOCType(ioc_type)
        return create_ioc(
            ioc_type=ioc_type,
            value=value,
            source_url=HttpUrl(source_url),
            extracted_from=extracted_from,
        )

    def article(
        self,
        url: str,
        title: str,
        author: str | None = None,
        published_date: datetime | None = None,
        source_name: str | None = None,
        summary: str | None = None,
    ) -> ArticleObservation:
        """Create an ArticleObservation for a blog post or security report."""
        return create_article_observation(
            url=url,
            title=title,
            author=author,
            published_date=published_date,
            source_name=source_name,
            summary=summary,
        )

    # GH Archive recovery methods

    def recover_issue(self, repo: str, issue_number: int, timestamp: str) -> IssueObservation:
        """Recover deleted issue content from GH Archive."""
        return create_issue_observation_from_gharchive(repo, issue_number, timestamp, self.gharchive)

    def recover_pr(self, repo: str, pr_number: int, timestamp: str) -> IssueObservation:
        """Recover deleted PR content from GH Archive."""
        return create_pr_observation_from_gharchive(repo, pr_number, timestamp, self.gharchive)

    def recover_commit(self, repo: str, sha: str, timestamp: str) -> CommitObservation:
        """Recover commit metadata from GH Archive."""
        return create_commit_observation_from_gharchive(repo, sha, timestamp, self.gharchive)

    def recover_force_push(self, repo: str, timestamp: str) -> CommitObservation:
        """Recover force-pushed commit from GH Archive."""
        return create_force_push_observation_from_gharchive(repo, timestamp, self.gharchive)
