from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.infrastructure.config.settings import settings

engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
AsyncSessionFactory = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


async def get_session():
    async with AsyncSessionFactory() as session:
        async with session.begin():
            yield session
