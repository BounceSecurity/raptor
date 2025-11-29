"""
GitHub Forensics Evidence Schema

Evidence objects are created ONLY through the EvidenceFactory which fetches
and verifies data from trusted third-party sources (GitHub API, GH Archive BigQuery,
Wayback Machine, security vendor URLs).

Usage:
    from src import EvidenceFactory

    factory = EvidenceFactory()

    # Verified from GitHub API
    commit = factory.commit("aws", "aws-toolkit-vscode", "678851b...")
    pr = factory.pull_request("aws", "aws-toolkit-vscode", 7710)

    # Verified from GH Archive BigQuery
    events = factory.events_from_gharchive(timestamp="202507132037", repo="aws/aws-toolkit-vscode")

    # Verified IOC (fetches source URL to confirm value exists)
    ioc = factory.ioc(IOCType.COMMIT_SHA, "678851b...", source_url="https://...")

For loading previously serialized evidence from JSON:
    from src import load_evidence_from_json
    evidence = load_evidence_from_json(json_data)

Type hints (for static analysis and IDE autocomplete):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from src import CommitObservation, IssueObservation
"""

from typing import Annotated, Union, TYPE_CHECKING

from pydantic import Field

from ._creation import EvidenceFactory
from ._store import EvidenceStore
from ._verification import verify_all

# Enums - Safe to expose, these are just constants
from ._schema import (
    EvidenceSource,
    EventType,
    RefType,
    PRAction,
    IssueAction,
    WorkflowConclusion,
    IOCType,
)

# Type aliases for external use
from ._schema import AnyEvidence, AnyEvent, AnyObservation

# Import all schema classes for discriminated union and TYPE_CHECKING exports
from ._schema import (
    # Events
    PushEvent,
    PullRequestEvent,
    IssueEvent,
    IssueCommentEvent,
    CreateEvent,
    DeleteEvent,
    ForkEvent,
    WorkflowRunEvent,
    ReleaseEvent,
    WatchEvent,
    MemberEvent,
    PublicEvent,
    # Observations
    CommitObservation,
    IssueObservation,
    FileObservation,
    ForkObservation,
    BranchObservation,
    TagObservation,
    ReleaseObservation,
    SnapshotObservation,
    IOC,
    ArticleObservation,
    # Common models (for type hints)
    GitHubActor,
    GitHubRepository,
    VerificationInfo,
    Event,
    Observation,
    CommitAuthor,
    FileChange,
    CommitInPush,
    WaybackSnapshot,
)

# Pydantic discriminated union for efficient JSON deserialization
_EventUnion = Annotated[
    Union[
        PushEvent,
        PullRequestEvent,
        IssueEvent,
        IssueCommentEvent,
        CreateEvent,
        DeleteEvent,
        ForkEvent,
        WorkflowRunEvent,
        ReleaseEvent,
        WatchEvent,
        MemberEvent,
        PublicEvent,
    ],
    Field(discriminator="event_type"),
]

_ObservationUnion = Annotated[
    Union[
        CommitObservation,
        IssueObservation,
        FileObservation,
        ForkObservation,
        BranchObservation,
        TagObservation,
        ReleaseObservation,
        SnapshotObservation,
        IOC,
        ArticleObservation,
    ],
    Field(discriminator="observation_type"),
]

from pydantic import TypeAdapter

_event_adapter = TypeAdapter(_EventUnion)
_observation_adapter = TypeAdapter(_ObservationUnion)


def load_evidence_from_json(data: dict) -> AnyEvidence:
    """
    Load a previously serialized evidence object from JSON.

    Args:
        data: Dictionary from JSON deserialization (e.g., json.load())

    Returns:
        The appropriate Event or Observation instance

    Raises:
        ValueError: If the data cannot be parsed into a known evidence type
    """
    if "event_type" in data:
        try:
            return _event_adapter.validate_python(data)
        except Exception as e:
            raise ValueError(f"Unknown event_type: {data.get('event_type')}") from e

    if "observation_type" in data:
        try:
            return _observation_adapter.validate_python(data)
        except Exception as e:
            raise ValueError(f"Unknown observation_type: {data.get('observation_type')}") from e

    raise ValueError("Data must contain 'event_type' or 'observation_type' field")


__all__ = [
    # Factory - Create evidence from sources
    "EvidenceFactory",
    # Store - Persist and query evidence collections
    "EvidenceStore",
    # Verification - Validate evidence against sources
    "verify_all",
    # Enums
    "EvidenceSource",
    "EventType",
    "RefType",
    "PRAction",
    "IssueAction",
    "WorkflowConclusion",
    "IOCType",
    # Loading from JSON
    "load_evidence_from_json",
    # Type aliases
    "AnyEvidence",
    "AnyEvent",
    "AnyObservation",
    # Type hints (for static analysis)
    "GitHubActor",
    "GitHubRepository",
    "VerificationInfo",
    "Event",
    "Observation",
    "CommitAuthor",
    "FileChange",
    "CommitInPush",
    "WaybackSnapshot",
    "PushEvent",
    "PullRequestEvent",
    "IssueEvent",
    "IssueCommentEvent",
    "CreateEvent",
    "DeleteEvent",
    "ForkEvent",
    "WorkflowRunEvent",
    "ReleaseEvent",
    "WatchEvent",
    "MemberEvent",
    "PublicEvent",
    "CommitObservation",
    "IssueObservation",
    "FileObservation",
    "ForkObservation",
    "BranchObservation",
    "TagObservation",
    "ReleaseObservation",
    "SnapshotObservation",
    "IOC",
    "ArticleObservation",
]
