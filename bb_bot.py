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
from datetime import datetime, timedelta, time as dtime
from typing import List
import logging
from dataclasses import dataclass
import json
import traceback
from pathlib import Path
import pytz

# Configure logging
logger = logging.getLogger("bb_bot")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class BBUpdate:
    title: str
    description: str
    link: str
    pub_date: datetime
    content_hash: str
    author: str = ""

class Config:
    def __init__(self):
        self.bot_token = os.getenv("BOT_TOKEN")
        self.update_channel_id = int(os.getenv("UPDATE_CHANNEL_ID", "0"))
        self.database_path = os.getenv("DATABASE_PATH", "bb_updates.db")

class BBDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS updates (
                content_hash TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                link TEXT,
                pub_date TEXT,
                author TEXT
            )
        """)

    def store_update(self, update: BBUpdate):
        with self.conn:
            self.conn.execute("""
                INSERT OR IGNORE INTO updates (content_hash, title, description, link, pub_date, author)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (update.content_hash, update.title, update.description, update.link, update.pub_date.isoformat(), update.author))

    def get_recent_updates(self, start_time: datetime, end_time: datetime) -> List[BBUpdate]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT title, description, link, pub_date, content_hash, author
            FROM updates
            WHERE pub_date BETWEEN ? AND ?
        """, (start_time.isoformat(), end_time.isoformat()))
        rows = cursor.fetchall()
        return [BBUpdate(*row) for row in rows]

class BBDiscordBot(commands.Bot):
    def __init__(self):
        self.config = Config()
        intents = discord.Intents.default()
        super().__init__(command_prefix="!bb", intents=intents, application_id=os.getenv("APPLICATION_ID"))

        self.tree = app_commands.CommandTree(self)
        self.db = BBDatabase(self.config.database_path)
        self.rss_url = "https://rss.jokersupdates.com/ubbthreads/rss/bbusaupdates/rss.php"
        self.daily_summary_time = dtime(hour=13, minute=0)  # 6:00 AM PT in UTC (13:00 UTC)
        self.tz = pytz.utc

        signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(self.close()))

    async def setup_hook(self):
        self.tree.clear_commands(guild=None)

        @self.tree.command(name="summary", description="Get today's Big Brother update summary")
        async def summary(interaction: discord.Interaction):
            now = datetime.now(self.tz)
            start = datetime.combine(now.date(), dtime(hour=13, minute=1), tzinfo=self.tz)  # 6:01 AM PT
            if now < start:
                start -= timedelta(days=1)
            updates = self.db.get_recent_updates(start, now)
            if not updates:
                await interaction.response.send_message("No updates found for today.", ephemeral=True)
                return

            embed = discord.Embed(title="Big Brother Daily Summary", color=0x3498db)
            for update in updates[:5]:
                embed.add_field(name=update.title[:256], value=update.description[:512], inline=False)
            await interaction.response.send_message(embed=embed, ephemeral=True)

        await self.tree.sync()
        self.daily_summary.start()

    async def on_ready(self):
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info("------")

    @tasks.loop(minutes=1.0)
    async def daily_summary(self):
        now = datetime.now(self.tz)
        if now.time().hour == self.daily_summary_time.hour and now.time().minute == self.daily_summary_time.minute:
            updates = self.db.get_recent_updates(
                now.replace(hour=self.daily_summary_time.hour, minute=1, second=0, microsecond=0) - timedelta(days=1),
                now.replace(hour=self.daily_summary_time.hour, minute=0, second=0, microsecond=0)
            )
            if not updates:
                return
            channel = self.get_channel(self.config.update_channel_id)
            if channel:
                embed = discord.Embed(title="📋 Big Brother Daily Summary", color=0x3498db)
                for update in updates[:5]:
                    embed.add_field(name=update.title[:256], value=update.description[:512], inline=False)
                await channel.send(embed=embed)

    def create_content_hash(self, title: str, description: str) -> str:
        return hashlib.md5(f"{title}|{description}".encode()).hexdigest()

    async def check_rss_loop(self):
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                feed = feedparser.parse(self.rss_url)
                for entry in feed.entries:
                    title = entry.get("title", "")
                    description = entry.get("description", "")
                    pub_date = datetime.now(self.tz)
                    if hasattr(entry, "published_parsed"):
                        pub_date = datetime(*entry.published_parsed[:6], tzinfo=self.tz)
                    update = BBUpdate(
                        title=title,
                        description=description,
                        link=entry.get("link", ""),
                        pub_date=pub_date,
                        content_hash=self.create_content_hash(title, description),
                        author=entry.get("author", "")
                    )
                    self.db.store_update(update)
                logger.info("RSS checked and updates stored.")
            except Exception as e:
                logger.error(f"Error in RSS fetch: {e}")
            await asyncio.sleep(120)

def main():
    bot = BBDiscordBot()
    asyncio.create_task(bot.check_rss_loop())
    bot.run(bot.config.bot_token)

if __name__ == "__main__":
    main()
