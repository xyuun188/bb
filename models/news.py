from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class NewsArticle(Base, TimestampMixin):
    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(50), index=True)
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(Text, unique=True)
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    symbols_mentioned: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )


class SocialPost(Base, TimestampMixin):
    __tablename__ = "social_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(String(20), index=True)
    post_id: Mapped[str] = mapped_column(String(100), unique=True)
    content: Mapped[str] = mapped_column(Text)
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    engagement_count: Mapped[int] = mapped_column(Integer, default=0)
    symbols: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    posted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
