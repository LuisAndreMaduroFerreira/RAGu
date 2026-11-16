from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Login(Input):
    username: str = Field(min_length=1, max_length=128)
    password: SecretStr = Field(min_length=1, max_length=128)


class PasswordChange(Input):
    current_password: SecretStr = Field(min_length=1, max_length=128)
    new_password: SecretStr = Field(min_length=15, max_length=128)


class ConfluenceCredential(Input):
    token: SecretStr = Field(min_length=10, max_length=4096)


def normalize_source(value: object) -> object:
    return "local_consumption" if value == "markdown" else value


ScheduledSourceKind = Annotated[
    Literal["local_consumption", "confluence"], BeforeValidator(normalize_source)
]
SourceKind = Annotated[
    Literal["local_consumption", "confluence", "multi_source"], BeforeValidator(normalize_source)
]
DEFAULT_SOURCE: SourceKind = "local_consumption"


class IndexMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class CandidateMethod(StrEnum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID_RRF = "hybrid_rrf"


class RerankMethod(StrEnum):
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"


class ProjectRole(StrEnum):
    READER = "reader"
    MAINTAINER = "maintainer"


class CorpusPermission(StrEnum):
    READ = "read"
    INGEST = "ingest"


class IngestionRequest(Input):
    source: ScheduledSourceKind = DEFAULT_SOURCE
    corpus_id: UUID | None = None
    index_mode: IndexMode | None = None


class FolderInput(Input):
    kind: Literal["folder"]
    path: Path

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError("A folder must be an absolute path without parent traversal.")
        return value


class ConfluenceInput(Input):
    kind: Literal["confluence"]
    url: str = Field(min_length=1, max_length=2048)


class MultiSourceRequest(Input):
    sources: list[Annotated[FolderInput | ConfluenceInput, Field(discriminator="kind")]] = Field(
        min_length=1, max_length=16
    )
    corpus_id: UUID | None = None
    index_mode: IndexMode | None = None


class DailySchedule(IngestionRequest):
    enabled: bool
    hour_utc: int = Field(ge=0, le=23)
    minute_utc: int = Field(ge=0, le=59)


class SearchQuery(Input):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)
    source: SourceKind = DEFAULT_SOURCE
    corpus_id: UUID | None = None
    candidate_method: CandidateMethod = CandidateMethod.DENSE
    candidate_limit: int | None = Field(default=None, ge=1, le=100)
    rerank_method: RerankMethod = RerankMethod.NONE

    @model_validator(mode="after")
    def candidate_bounds(self):
        if self.candidate_limit is not None and self.candidate_limit < self.limit:
            raise ValueError("candidate_limit must be at least limit.")
        return self


class CandidateQuery(SearchQuery):
    rerank_method: Literal[RerankMethod.NONE] = RerankMethod.NONE


class RerankRequest(Input):
    query: str = Field(min_length=1, max_length=2000)
    source: SourceKind = DEFAULT_SOURCE
    corpus_id: UUID | None = None
    generation_id: UUID
    candidate_ids: list[UUID] = Field(min_length=1, max_length=100)
    rerank_method: RerankMethod = RerankMethod.CROSS_ENCODER
    limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def unique_candidates(self):
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("Candidate IDs must be unique.")
        if self.limit > len(self.candidate_ids):
            raise ValueError("limit cannot exceed the number of candidates.")
        return self


class ProjectCreate(Input):
    name: str = Field(min_length=1, max_length=100, pattern=r"\S")


class MemberUpdate(Input):
    role: ProjectRole


class GrantUpdate(Input):
    permission: CorpusPermission


class CorpusCreate(Input):
    name: str = Field(min_length=1, max_length=100, pattern=r"\S")
    source: SourceKind = DEFAULT_SOURCE
    folder: Path | None = None

    @field_validator("folder")
    @classmethod
    def absolute_folder(cls, value):
        if value is not None and (not value.is_absolute() or ".." in value.parts):
            raise ValueError("A corpus folder must be an absolute path without parent traversal.")
        return value

    @model_validator(mode="after")
    def folder_source(self):
        if self.folder is not None and self.source != "local_consumption":
            raise ValueError("folder is only valid for a local-consumption corpus.")
        return self


class UserView(BaseModel):
    id: UUID
    username: str
    role: str


class TokenView(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class JobView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    source: SourceKind
    corpus_id: UUID | None = None
    index_mode: IndexMode = IndexMode.DENSE
    unchanged: bool
    file_count: int
    duplicate_count: int = Field(
        description="Local files excluded as exact duplicates or superseded filename versions."
    )
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    page_count: int
    chunk_count: int
    chunk_reference_count: int
    reused_chunk_count: int
    error: str | None
