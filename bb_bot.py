import discord
from discord.ext import tasks, commands
from discord import app_commands
import feedparser
import asyncio
import sqlite3
import hashlib
import re
import os
import sys
import signal
from datetime import datetime, timedelta, time as dt_time
from typing import List
import logging
from dataclasses import dataclass
import json
import traceback
from pathlib import Path

# Configure comprehensive logging
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler = logging.FileHandler(log_dir / "bb_bot.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)

    error_handler = logging.FileHandler(log_dir / "bb_bot_errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)

logger = setup_logging()

class Config:
    def __init__(self):
        self.config_file = Path("config.json")
        self.default_config = {
            "bot_token": "",
            "update_channel_id": None,
            "rss_check_interval": 2,
            "max_retries": 3,
            "retry_delay": 5,
            "database_path": "bb_updates.db",
            "enable_heartbeat": True,
            "heartbeat_interval": 300,
            "max_update_age_hours": 168,
            "enable_auto_restart": True,
            "max_consecutive_errors": 10
        }
        self.config = self.load_config()

    def load_config(self) -> dict:
        config = {
            "bot_token": os.getenv('BOT_TOKEN', ''),
            "update_channel_id": int(os.getenv('UPDATE_CHANNEL_ID', '0')) or None,
            "rss_check_interval": int(os.getenv('RSS_CHECK_INTERVAL', '2')),
            "max_retries": int(os.getenv('MAX_RETRIES', '3')),
            "retry_delay": int(os.getenv('RETRY_DELAY', '5')),
            "database_path": os.getenv('DATABASE_PATH', 'bb_updates.db'),
            "enable_heartbeat": os.getenv('ENABLE_HEARTBEAT', 'true').lower() == 'true',
            "heartbeat_interval": int(os.getenv('HEARTBEAT_INTERVAL', '300')),
            "max_update_age_hours": int(os.getenv('MAX_UPDATE_AGE_HOURS', '168')),
            "enable_auto_restart": os.getenv('ENABLE_AUTO_RESTART', 'true').lower() == 'true',
            "max_consecutive_errors": int(os.getenv('MAX_CONSECUTIVE_ERRORS', '10'))
        }
        if not config["bot_token"] and self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                logger.error(f"Error loading config file: {e}")

        return config

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def set(self, key: str, value):
        self.config[key] = value

@dataclass
class BBUpdate:
    title: str
    description: str
    link: str
    pub_date: datetime
    content_hash: str
    author: str = ""

class BBAnalyzer:
    def __init__(self):
        self.competition_keywords = [
            'hoh', 'head of household', 'power of veto', 'pov', 'nomination',
            'eviction', 'ceremony', 'competition', 'challenge', 'immunity'
        ]
        self.strategy_keywords = [
            'alliance', 'backdoor', 'target', 'scheme', 'plan', 'strategy',
            'vote', 'voting', 'campaigning', 'deal', 'promise', 'betrayal'
        ]
        self.drama_keywords = [
            'argument', 'fight', 'confrontation', 'drama', 'tension',
            'called out', 'blowup', 'heated', 'angry', 'upset'
        ]
        self.relationship_keywords = [
            'showmance', 'romance', 'flirting', 'cuddle', 'kiss',
            'relationship', 'attracted', 'feelings'
        ]

    def categorize_update(self, update: BBUpdate) -> List[str]:
        content = f"{update.title} {update.description}".lower()
        categories = []
        if any(k in content for k in self.competition_keywords):
            categories.append("🏆 Competition")
        if any(k in content for k in self.strategy_keywords):
            categories.append("🎯 Strategy")
        if any(k in content for k in self.drama_keywords):
            categories.append("💥 Drama")
        if any(k in content for k in self.relationship_keywords):
            categories.append("💕 Romance")
        return categories if categories else ["📝 General"]

    def extract_houseguests(self, text: str) -> List[str]:
        houseguest_pattern = r'\b[A-Z][a-z]+\b'
        potential_names = re.findall(houseguest_pattern, text)
        exclude_words = {'The', 'This', 'That', 'They', 'Some', 'Many', 'Other', 'First', 'Last'}
        return [name for name in potential_names if name not in exclude_words]

    def analyze_strategic_importance(self, update: BBUpdate) -> int:
        content = f"{update.title} {update.description}".lower()
        score = 1
        if any(word in content for word in ['eviction', 'nomination', 'backdoor']):
            score += 4
        if any(word in content for word in ['hoh', 'head of household', 'power of veto']):
            score += 3
        if any(word in content for word in ['alliance', 'target', 'strategy']):
            score += 2
        if any(word in content for word in ['vote', 'voting', 'campaign']):
            score += 2
        return min(score, 10)

class BBDatabase:
    def __init__(self, db_path: str = "bb_updates.db"):
        self.db_path = db_path
        self.connection_timeout = 30
        self.init_database()

    def get_connection(self):
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.connection_timeout,
                check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    def init_database(self):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT UNIQUE,
                    title TEXT,
                    description TEXT,
                    link TEXT,
                    pub_date TIMESTAMP,
                    author TEXT,
                    importance_score INTEGER DEFAULT 1,
                    categories TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON updates(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pub_date ON updates(pub_date)")
            conn.commit()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
        finally:
            conn.close()

    def is_duplicate(self, content_hash: str) -> bool:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM updates WHERE content_hash = ?", (content_hash,))
            result = cursor.fetchone()
            conn.close()
            return result is not None
        except Exception as e:
            logger.error(f"Database duplicate check error: {e}")
            return False

    def store_update(self, update: BBUpdate, importance_score: int = 1, categories: List[str] = None):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            categories_str = ",".join(categories) if categories else ""
            cursor.execute("""
                INSERT INTO updates (content_hash, title, description, link, pub_date, author, importance_score, categories)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (update.content_hash, update.title, update.description,
                  update.link, update.pub_date, update.author, importance_score, categories_str))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database store error: {e}")
            raise

    def get_recent_updates(self, hours: int = 24) -> List[BBUpdate]:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            cursor.execute("""
                SELECT title, description, link, pub_date, content_hash, author
                FROM updates 
                WHERE pub_date > ?
                ORDER BY pub_date DESC
            """, (cutoff_time,))
            results = cursor.fetchall()
            conn.close()
            return [BBUpdate(*row) for row in results]
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []

class BBDiscordBot(commands.Bot):
    def __init__(self):
        self.config = Config()

        if not self.config.get('bot_token'):
            logger.error("Bot token not configured! Please set BOT_TOKEN environment variable")
            sys.exit(1)

        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!bb', intents=intents)

        # DO NOT manually create a CommandTree here, use built-in self.tree

        self.rss_url = "https://rss.jokersupdates.com/ubbthreads/rss/bbusaupdates/rss.php"
        self.db = BBDatabase(self.config.get('database_path', 'bb_updates.db'))
        self.analyzer = BBAnalyzer()

        self.is_shutting_down = False
        self.last_successful_check = datetime.utcnow()
        self.total_updates_processed = 0
        self.consecutive_errors = 0

        self.daily_summary_time_utc = dt_time(hour=13, minute=0, second=0)  
        # 6:00 AM Pacific = 13:00 UTC during PDT (summer)

        self.daily_summary_task = None

        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_shutting_down = True
        asyncio.create_task(self.close())

    async def setup_hook(self):
        # Register slash commands here
        self.tree.add_command(self.daily_summary)
        self.tree.add_command(self.set_update_channel)
        self.tree.add_command(self.bot_status)
        self.tree.add_command(self.commands_help)

        # Sync commands globally or to guilds if testing
        await self.tree.sync()

        # Start background tasks
        self.check_rss_feed.start()
        self.daily_summary_task = self.loop.create_task(self.schedule_daily_summary())

    async def schedule_daily_summary(self):
        await self.wait_until_ready()
        while not self.is_shutting_down:
            now = datetime.utcnow()
            target_dt = datetime.combine(now.date(), self.daily_summary_time_utc)
            if now > target_dt:
                target_dt += timedelta(days=1)
            wait_seconds = (target_dt - now).total_seconds()
            logger.info(f"Waiting {wait_seconds:.1f}s until next daily summary")
            await asyncio.sleep(wait_seconds)

            if not self.is_shutting_down:
                await self.send_daily_summary()

    async def send_daily_summary(self):
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            logger.warning("Update channel not configured, skipping daily summary")
            return
        channel = self.get_channel(channel_id)
        if not channel:
            logger.error(f"Channel {channel_id} not found, cannot send daily summary")
            return

        # Define the daily summary period from 6:01AM UTC previous day to 6:00AM UTC today
        now = datetime.utcnow()
        end_time = datetime.combine(now.date(), dt_time(6, 0))
        start_time = end_time - timedelta(days=1) + timedelta(minutes=1)

        updates = self.db.get_recent_updates(24*2)  # fetch last 48 hours to filter by date range manually

        # Filter updates for the period start_time <= pub_date < end_time
        filtered_updates = [u for u in updates if start_time <= u.pub_date < end_time]

        if not filtered_updates:
            await channel.send("No updates found for today's summary.")
            return

        categories = {}
        for update in filtered_updates:
            update_categories = self.analyzer.categorize_update(update)
            for category in update_categories:
                categories.setdefault(category, []).append(update)

        embed = discord.Embed(
            title=f"Big Brother Daily Summary ({start_time.strftime('%Y-%m-%d')})",
            description=f"**{len(filtered_updates)} total updates**",
            color=0x3498db,
            timestamp=datetime.utcnow()
        )

        for category, cat_updates in categories.items():
            top_updates = sorted(cat_updates,
                               key=lambda x: self.analyzer.analyze_strategic_importance(x),
                               reverse=True)[:3]
            summary_text = "\n".join([f"• {update.title[:100]}..." if len(update.title) > 100 else f"• {update.title}"
                                      for update in top_updates])
            embed.add_field(
                name=f"{category} ({len(cat_updates)} updates)",
                value=summary_text or "No updates",
                inline=False
            )

        await channel.send(embed=embed)

    def create_content_hash(self, title: str, description: str) -> str:
        content = f"{title}|{description}".lower()
        content = re.sub(r'\d{1,2}:\d{2}[ap]m', '', content)
        content = re.sub(r'\d{1,2}/\d{1,2}', '', content)
        return hashlib.md5(content.encode()).hexdigest()

    def process_rss_entries(self, entries) -> List[BBUpdate]:
        updates = []
        for entry in entries:
            try:
                title = entry.get('title', 'No title')
                description = entry.get('description', 'No description')
                link = entry.get('link', '')
                pub_date = datetime.utcnow()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                content_hash = self.create_content_hash(title, description)
                author = entry.get('author', '')
                updates.append(BBUpdate(title, description, link, pub_date, content_hash, author))
            except Exception as e:
                logger.error(f"Error processing RSS entry: {e}")
        return updates

    def filter_duplicates(self, updates: List[BBUpdate]) -> List[BBUpdate]:
        new_updates = []
        seen_hashes = set()
        for update in updates:
            if not self.db.is_duplicate(update.content_hash) and update.content_hash not in seen_hashes:
                new_updates.append(update)
                seen_hashes.add(update.content_hash)
        return new_updates

    def create_update_embed(self, update: BBUpdate) -> discord.Embed:
        categories = self.analyzer.categorize_update(update)
        importance = self.analyzer.analyze_strategic_importance(update)
        colors = {1: 0x95a5a6, 2: 0x3498db, 3: 0x2ecc71, 4: 0xf39c12, 5: 0xe74c3c}
        color = colors.get(min(importance // 2 + 1, 5), 0x95a5a6)

        title = update.title if len(update.title) <= 256 else update.title[:253] + "..."
        description = update.description if len(update.description) <= 2048 else update.description[:2045] + "..."

        embed = discord.Embed(title=title, description=description, color=color, url=update.link, timestamp=update.pub_date)
        if categories:
            embed.add_field(name="Categories", value=" | ".join(categories), inline=True)
        importance_stars = "⭐" * importance
        embed.add_field(name="Strategic Importance", value=f"{importance_stars} ({importance}/10)", inline=True)
        houseguests = self.analyzer.extract_houseguests(f"{update.title} {update.description}")
        if houseguests:
            houseguests_text = ", ".join(houseguests[:5])
            if len(houseguests) > 5:
                houseguests_text += f" +{len(houseguests) - 5} more"
            embed.add_field(name="Houseguests Mentioned", value=houseguests_text, inline=False)
        if update.author:
            embed.set_footer(text=f"Reported by: {update.author}")
        return embed

    async def send_update_to_channel(self, update: BBUpdate):
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            logger.warning("Update channel not configured")
            return
        channel = self.get_channel(channel_id)
        if not channel:
            logger.error(f"Channel {channel_id} not found")
            return
        embed = self.create_update_embed(update)
        try:
            await channel.send(embed=embed)
        except Exception as e:
            logger.error(f"Failed to send update: {e}")

    @tasks.loop(minutes=2)
    async def check_rss_feed(self):
        if self.is_shutting_down:
            return
        try:
            feed = feedparser.parse(self.rss_url)
            if feed.bozo:
                logger.warning(f"RSS feed parse error: {feed.bozo_exception}")
                self.consecutive_errors += 1
                if self.consecutive_errors >= self.config.get('max_consecutive_errors', 10):
                    logger.error("Too many consecutive errors, shutting down bot")
                    await self.close()
                return
            self.consecutive_errors = 0

            updates = self.process_rss_entries(feed.entries)
            new_updates = self.filter_duplicates(updates)
            for update in new_updates:
                categories = self.analyzer.categorize_update(update)
                importance = self.analyzer.analyze_strategic_importance(update)
                self.db.store_update(update, importance, categories)
                await self.send_update_to_channel(update)
                self.total_updates_processed += 1
            self.last_successful_check = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error during RSS check: {traceback.format_exc()}")
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.config.get('max_consecutive_errors', 10):
                logger.error("Too many consecutive errors, shutting down bot")
                await self.close()

    @app_commands.command(name="summary", description="Get a summary of updates for the last N hours (max 48).")
    async def daily_summary(self, interaction: discord.Interaction, hours: int = 24):
        hours = min(max(hours, 1), 48)
        updates = self.db.get_recent_updates(hours)
        if not updates:
            await interaction.response.send_message(f"No updates found in the last {hours} hours.", ephemeral=True)
            return

        categories = {}
        for update in updates:
            update_categories = self.analyzer.categorize_update(update)
            for cat in update_categories:
                categories.setdefault(cat, []).append(update)

        embed = discord.Embed(
            title=f"Big Brother Summary (last {hours} hours)",
            description=f"Total updates: {len(updates)}",
            color=0x3498db,
            timestamp=datetime.utcnow()
        )
        for category, cat_updates in categories.items():
            top_updates = sorted(cat_updates,
                               key=lambda u: self.analyzer.analyze_strategic_importance(u),
                               reverse=True)[:3]
            summary_text = "\n".join([f"• {u.title[:100]}{'...' if len(u.title) > 100 else ''}" for u in top_updates])
            embed.add_field(name=f"{category} ({len(cat_updates)})", value=summary_text or "No updates", inline=False)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="setchannel", description="Set the channel to post RSS updates (Admin only).")
    @app_commands.checks.has_permissions(administrator=True)
    async def set_update_channel(self, interaction: discord.Interaction, channel: discord.TextChannel):
        self.config.set('update_channel_id', channel.id)
        # Ideally, persist this change to config.json here if you want permanence
        await interaction.response.send_message(f"Update channel set to {channel.mention}", ephemeral=True)

    @app_commands.command(name="status", description="Get the bot's current status.")
    async def bot_status(self, interaction: discord.Interaction):
        uptime = datetime.utcnow() - self.last_successful_check
        embed = discord.Embed(title="Big Brother Bot Status", color=0x2ecc71)
        embed.add_field(name="Updates Processed", value=str(self.total_updates_processed))
        embed.add_field(name="Last RSS Check", value=self.last_successful_check.strftime("%Y-%m-%d %H:%M UTC"))
        embed.add_field(name="Uptime Since Last Successful Check", value=str(uptime))
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="commands", description="Show available bot commands.")
    async def commands_help(self, interaction: discord.Interaction):
        commands_list = (
            "/summary [hours] - Get a summary of updates for the last N hours (default 24).\n"
            "/setchannel [channel] - Set the channel for RSS update posts (admin only).\n"
            "/status - Show bot status.\n"
            "/commands - Show this help message."
        )
        await interaction.response.send_message(f"**Big Brother Bot Commands:**\n{commands_list}", ephemeral=True)

def main():
    bot = BBDiscordBot()
    bot.run(bot.config.get('bot_token'))

if __name__ == "__main__":
    main()
