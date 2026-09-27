import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from shot_clipper.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# SHOT_CLIPPER_DATABASE_URL takes precedence over alembic.ini's placeholder
# URL - same env var the app itself reads (db/session.py), so `alembic
# upgrade head` always targets whatever database the app would connect to.
db_url = os.environ.get("SHOT_CLIPPER_DATABASE_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
