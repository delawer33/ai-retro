"""Вся схема этапа 1

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None


def upgrade() -> None:
    op.create_table(
        "episodes",
        sa.Column("uuid", sa.String, primary_key=True),
        sa.Column("conversation_id", sa.String, nullable=False),
        sa.Column("project", sa.String, nullable=False),
        sa.Column("domain", sa.String, nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("context", sa.Text, nullable=True),
        sa.Column("signals", sa.JSON, nullable=False),
    )
    op.create_index("ix_episodes_conversation_id", "episodes", ["conversation_id"])
    op.create_table(
        "classifications",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("episode_uuid", sa.String, sa.ForeignKey("episodes.uuid"), nullable=False),
        sa.Column("category", sa.String, nullable=False),
        sa.Column("gist", sa.Text, nullable=False),
        sa.Column("classifier_version", sa.String, nullable=False),
    )
    op.create_index("ix_classifications_episode_uuid", "classifications", ["episode_uuid"])
    op.create_table(
        "gold_labels",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("episode_uuid", sa.String, sa.ForeignKey("episodes.uuid"), nullable=False),
        sa.Column("category", sa.String, nullable=False),
        sa.Column("split", sa.String, nullable=False),
        sa.Column("labeled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_relabel", sa.Boolean, nullable=False),
    )
    op.create_index("ix_gold_labels_episode_uuid", "gold_labels", ["episode_uuid"])
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("classifier_version", sa.String, nullable=False),
        sa.Column("claude_version", sa.String, nullable=False),
        sa.Column("model", sa.String, nullable=False),
        sa.Column("split", sa.String, nullable=False),
        sa.Column("metrics", sa.JSON, nullable=False),
    )
    op.create_table(
        "patterns",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("gist", sa.Text, nullable=False),
        sa.Column("episode_uuids", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "proposals",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("pattern_id", sa.Integer, sa.ForeignKey("patterns.id"), nullable=False),
        sa.Column("target_file", sa.String, nullable=False),
        sa.Column("section", sa.String, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("rationale", sa.Text, nullable=False),
        sa.Column("status", sa.String, nullable=False),
    )
    op.create_index("ix_proposals_pattern_id", "proposals", ["pattern_id"])


def downgrade() -> None:
    for table in (
        "proposals",
        "patterns",
        "eval_runs",
        "gold_labels",
        "classifications",
        "episodes",
    ):
        op.drop_table(table)
