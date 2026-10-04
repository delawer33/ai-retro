"""Выборка для разметки эталона

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    op.create_table(
        "gold_sample",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "episode_uuid", sa.String, sa.ForeignKey("episodes.uuid"), nullable=False, unique=True
        ),
        sa.Column("split", sa.String, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("gold_sample")
