"""Вызов скилла — признак, а не категория (ADR 0004)

Добавляет episodes.skill, переводит старые метки «вызов скилла»: в эталоне — «нейтрально» для
голого вызова, иначе метка снимается, эпизод возвращается в разметку и записывается в
gold_returned; старые классификации этой категории удаляются (scan пересчитает).

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"

OLD = "вызов скилла"
NEUTRAL = "нейтрально"

_episodes = sa.table(
    "episodes",
    sa.column("uuid", sa.String),
    sa.column("text", sa.Text),
    sa.column("signals", sa.JSON),
    sa.column("skill", sa.String),
)
_classifications = sa.table(
    "classifications", sa.column("episode_uuid", sa.String), sa.column("category", sa.String)
)
_gold = sa.table(
    "gold_labels",
    sa.column("episode_uuid", sa.String),
    sa.column("category", sa.String),
    sa.column("is_relabel", sa.Boolean),
)
_returned = sa.table("gold_returned", sa.column("episode_uuid", sa.String))


def upgrade() -> None:
    op.add_column("episodes", sa.Column("skill", sa.String, nullable=True))
    op.create_table(
        "gold_returned",
        sa.Column("episode_uuid", sa.String, sa.ForeignKey("episodes.uuid"), primary_key=True),
    )
    bind = op.get_bind()
    old_uuids = {
        row[0]
        for table in (_classifications, _gold)
        for row in bind.execute(sa.select(table.c.episode_uuid).where(table.c.category == OLD))
    }
    has_args: dict[str, bool] = {}
    for uuid in sorted(old_uuids):
        text, signals = bind.execute(
            sa.select(_episodes.c.text, _episodes.c.signals).where(_episodes.c.uuid == uuid)
        ).one()
        parts = text.split(maxsplit=1)
        has_args[uuid] = len(parts) == 2
        bind.execute(
            _episodes.update()
            .where(_episodes.c.uuid == uuid)
            .values(skill=parts[0].lstrip("/"), signals=[*signals, OLD])
        )
    bind.execute(_classifications.delete().where(_classifications.c.category == OLD))
    for uuid, args in has_args.items():
        if args:
            labeled = bind.execute(
                sa.select(_gold.c.episode_uuid).where(
                    _gold.c.episode_uuid == uuid, _gold.c.category == OLD
                )
            ).first()
            if labeled is not None:
                bind.execute(_gold.delete().where(_gold.c.episode_uuid == uuid))
                bind.execute(_returned.insert().values(episode_uuid=uuid))
        else:
            bind.execute(
                _gold.update()
                .where(_gold.c.episode_uuid == uuid, _gold.c.category == OLD)
                .values(category=NEUTRAL)
            )


def downgrade() -> None:
    op.drop_table("gold_returned")
    op.drop_column("episodes", "skill")
