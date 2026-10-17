"""Паттерн помнит отчёт, который его породил (ADR 0015)

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"


def upgrade() -> None:
    op.add_column("patterns", sa.Column("period", sa.String, nullable=True))
    op.add_column("patterns", sa.Column("classifier_version", sa.String, nullable=True))


def downgrade() -> None:
    op.drop_column("patterns", "classifier_version")
    op.drop_column("patterns", "period")
