from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EpisodeRow(Base):
    __tablename__ = "episodes"

    uuid: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String, index=True)
    project: Mapped[str] = mapped_column(String)
    domain: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    text: Mapped[str] = mapped_column(Text)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    signals: Mapped[list[str]] = mapped_column(JSON, default=list)


class ClassificationRow(Base):
    __tablename__ = "classifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_uuid: Mapped[str] = mapped_column(ForeignKey("episodes.uuid"), index=True)
    category: Mapped[str] = mapped_column(String)
    gist: Mapped[str] = mapped_column(Text)
    classifier_version: Mapped[str] = mapped_column(String)


class GoldLabelRow(Base):
    __tablename__ = "gold_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_uuid: Mapped[str] = mapped_column(ForeignKey("episodes.uuid"), index=True)
    category: Mapped[str] = mapped_column(String)
    split: Mapped[str] = mapped_column(String)  # dev | test
    labeled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    is_relabel: Mapped[bool] = mapped_column(Boolean, default=False)


class GoldSampleRow(Base):
    """Выборка эпизодов под разметку; порядок — по id. Нужна, чтобы разметку можно было продолжить."""

    __tablename__ = "gold_sample"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_uuid: Mapped[str] = mapped_column(ForeignKey("episodes.uuid"), unique=True)
    split: Mapped[str] = mapped_column(String)  # dev | test


class EvalRunRow(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    classifier_version: Mapped[str] = mapped_column(String)
    claude_version: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    split: Mapped[str] = mapped_column(String)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class PatternRow(Base):
    __tablename__ = "patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gist: Mapped[str] = mapped_column(Text)
    episode_uuids: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ProposalRow(Base):
    __tablename__ = "proposals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pattern_id: Mapped[int] = mapped_column(ForeignKey("patterns.id"), index=True)
    target_file: Mapped[str] = mapped_column(String)
    section: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default="proposed")  # proposed|applied|rejected
