# bb_bot.py
import discord
from discord.ext import commands, tasks
from discord import app_commands
import feedparser
import asyncio
import sqlite3
import hashlib
import re
import os
import sys
import signal
from datetime import datetime, timedelta
from typing import List
import logging
from dataclasses import dataclass
import json
from pathlib import Path

# --- Setup Logging ---
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_dir / "bb_bot.log")
    file_handler.setFormatter(formatter)

    error_handler = logging.FileHandler(log_dir / "bb_bot_errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)

logger = setup_logging()

# --- Config ---
class Config:
    def __init__(self):
        self.config = {
            "bot_token": os.getenv("BOT_TOKEN"),
            "update_channel_id": int(os.getenv("UPDATE_CHANNEL_ID", "0")) or None,
            "rss_check_interval": int(os.getenv("RSS_CHECK_INTERVAL", "2")),
            "database_path": "bb_updates.db",
            "summary_interval_minutes": 30,
            "summary_trigger_threshold": 10
        }

    def get(self, key, default=None):
        return self.config.get(key, default)

@dataclass
class BBUpdate:
    title: str
    description: str
    link: str
    pub_date: datetime
    content_hash: str
    author: str = ""

# --- Analyzer ---
class BBAnalyzer:
    def __init__(self):
        self.categories = {
            "🏆 Competition": ['hoh', 'veto', 'eviction', 'competition', 'challenge'],
            "🎯 Strategy": ['alliance', 'vote', 'plan', 'target'],
            "💥 Drama": ['fight', 'argue', 'screamed', 'blowup'],
            "💕 Romance": ['flirt', 'kiss', 'cuddle', 'showmance']
        }

    def categorize(self, text: str):
        content = text.lower()
        tags = [cat for cat, keywords in self.categories.items() if any(k in content for k in keywords)]
        return tags or ["📝 General"]

    def importance_score(self, text: str):
        score = 1
        score += sum(k in text.lower() for k in ['eviction', 'hoh', 'backdoor', 'veto', 'target']) * 2
        return min(score, 10)

# --- Database ---
class BBDatabase:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS updates (
            content_hash TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            link TEXT,
            pub_date TEXT,
            author TEXT
        )""")
        self.conn.commit()

    def is_duplicate(self, content_hash):
        cur = self.conn.execute("SELECT 1 FROM updates WHERE content_hash = ?", (content_hash,))
        return cur.fetchone() is not None

    def insert(self, update: BBUpdate):
        self.conn.execute("""
        INSERT INTO updates (content_hash, title, description, link, pub_date, author)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (update.content_hash, update.title, update.description, update.link, update.pub_date.isoformat(), update.author))
        self.conn.commit()

    def get_updates_since(self, since: datetime):
        cur = self.conn.execute("SELECT title, description, link, pub_date, content_hash, author FROM updates WHERE pub_date > ?", (since.isoformat(),))
        return [BBUpdate(*row) for row in cur.fetchall()]

# --- Bot ---
class BBDiscordBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!bb", intents=discord.Intents.default())
        self.config = Config()
        self.db = BBDatabase(self.config.get("database_path"))
        self.analyzer = BBAnalyzer()
        self.tree = app_commands.CommandTree(self)
        self.updates_buffer: List[BBUpdate] = []
        self.last_summary_sent = datetime.utcnow()

    async def setup_hook(self):
        self.tree.add_command(self.summary)
        self.tree.add_command(self.status)
        await self.tree.sync()
        logger.info("Slash commands registered and synced")

    async def on_ready(self):
        logger.info(f"{self.user} has connected to Discord!")
        self.check_rss_feed.start()

    def content_hash(self, title, description):
        clean = re.sub(r'\s+', '', title + description).lower()
        return hashlib.md5(clean.encode()).hexdigest()

    def build_embed(self, update: BBUpdate):
        tags = self.analyzer.categorize(update.title + " " + update.description)
        score = self.analyzer.importance_score(update.title + " " + update.description)
        embed = discord.Embed(
            title=update.title,
            description=update.description,
            url=update.link,
            color=0x3498db,
            timestamp=update.pub_date
        )
        embed.add_field(name="Tags", value=", ".join(tags), inline=False)
        embed.add_field(name="Importance", value=f"{score}/10 ⭐", inline=True)
        return embed

    @tasks.loop(minutes=2)
    async def check_rss_feed(self):
        logger.info("Checking RSS feed for new updates...")
        try:
            feed = feedparser.parse("https://rss.jokersupdates.com/ubbthreads/rss/bbusaupdates/rss.php")
            new_updates = []

            for entry in feed.entries:
                title = entry.get("title", "")
                description = entry.get("description", "")
                link = entry.get("link", "")
                pub_date = datetime.utcnow()
                content_hash = self.content_hash(title, description)

                if self.db.is_duplicate(content_hash):
                    continue

                update = BBUpdate(
                    title=title,
                    description=description,
                    link=link,
                    pub_date=pub_date,
                    content_hash=content_hash,
                    author=entry.get("author", "")
                )

                self.db.insert(update)
                self.updates_buffer.append(update)
                new_updates.append(update)

            if new_updates:
                logger.info(f"Found {len(new_updates)} new updates")

            # Trigger summary if conditions met
            if (
                len(self.updates_buffer) >= self.config.get("summary_trigger_threshold", 10) or
                (datetime.utcnow() - self.last_summary_sent).total_seconds() >= self.config.get("summary_interval_minutes", 30) * 60
            ):
                await self.send_batch_summary()

        except Exception as e:
            logger.error(f"Error checking RSS feed: {e}")

    async def send_batch_summary(self):
        logger.info("Sending batch summary...")
        channel_id = self.config.get("update_channel_id")
        if not channel_id:
            logger.warning("No update channel configured")
            return
        channel = self.get_channel(channel_id)
        if not channel:
            logger.warning("Channel not found")
            return
        embed = discord.Embed(title="🧠 Big Brother Update Summary", timestamp=datetime.utcnow(), color=0x2ecc71)
        embed.description = f"Last {len(self.updates_buffer)} updates:\n"
        for update in self.updates_buffer[-10:]:
            embed.add_field(name=update.title, value=f"[Link]({update.link}) | ⭐ {self.analyzer.importance_score(update.title + update.description)}", inline=False)

        await channel.send(embed=embed)
        self.updates_buffer.clear()
        self.last_summary_sent = datetime.utcnow()

    # --- Slash Commands ---
    @app_commands.command(name="summary", description="Get a summary of updates in last 24 hours")
    async def summary(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        since = datetime.utcnow() - timedelta(hours=24)
        updates = self.db.get_updates_since(since)
        embed = discord.Embed(title="Big Brother Feed Summary", color=0x5865f2, timestamp=datetime.utcnow())
        embed.description = f"Updates in last 24 hours: {len(updates)}"

        if not updates:
            await interaction.followup.send("No updates found.", ephemeral=True)
            return

        for update in updates[:10]:
            embed.add_field(name=update.title, value=f"[Link]({update.link})", inline=False)

        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="status", description="Show bot status and config")
    async def status(self, interaction: discord.Interaction):
        await interaction.response.send_message(
            f"Bot is running. Feed check every {self.config.get('rss_check_interval')} min. "
            f"Batch every {self.config.get('summary_interval_minutes')} min or {self.config.get('summary_trigger_threshold')} updates.",
            ephemeral=True
        )

# --- Main ---
def main():
    bot = BBDiscordBot()
    token = bot.config.get("bot_token")
    if not token:
        logger.error("BOT_TOKEN not set in environment.")
        return
    bot.run(token)

if __name__ == "__main__":
    main()
