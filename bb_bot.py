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
import traceback
from pathlib import Path

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging for 24/7 operation"""
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
        
        if any(keyword in content for keyword in self.competition_keywords):
            categories.append("🏆 Competition")
        
        if any(keyword in content for keyword in self.strategy_keywords):
            categories.append("🎯 Strategy")
        
        if any(keyword in content for keyword in self.drama_keywords):
            categories.append("💥 Drama")
        
        if any(keyword in content for keyword in self.relationship_keywords):
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
            cutoff_time = datetime.now() - timedelta(hours=hours)
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
        intents.guilds = True
        super().__init__(command_prefix='!bb', intents=intents)
        
        self.rss_url = "https://rss.jokersupdates.com/ubbthreads/rss/bbusaupdates/rss.php"
        self.db = BBDatabase(self.config.get('database_path', 'bb_updates.db'))
        self.analyzer = BBAnalyzer()
        
        self.is_shutting_down = False
        self.last_successful_check = datetime.now()
        self.total_updates_processed = 0
        self.consecutive_errors = 0
        
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_shutting_down = True
        asyncio.create_task(self.close())
    
    async def setup_hook(self):
        # Register all slash commands explicitly
        self.tree.add_command(self.status_slash)
        self.tree.add_command(self.summary_slash)
        self.tree.add_command(self.setchannel_slash)
        self.tree.add_command(self.commands_slash)
        self.tree.add_command(self.test_updates)
        await self.tree.sync()
        logger.info("Slash commands registered and synced")
    
    async def on_ready(self):
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')
        try:
            self.check_rss_feed.start()
            logger.info("RSS feed monitoring started")
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
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
                pub_date = datetime.now()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                content_hash = self.create_content_hash(title, description)
                author = entry.get('author', '')
                updates.append(BBUpdate(
                    title=title,
                    description=description,
                    link=link,
                    pub_date=pub_date,
                    content_hash=content_hash,
                    author=author
                ))
            except Exception as e:
                logger.error(f"Error processing RSS entry: {e}")
                continue
        return updates
    
    def filter_duplicates(self, updates: List[BBUpdate]) -> List[BBUpdate]:
        new_updates = []
        seen_hashes = set()
        for update in updates:
            if not self.db.is_duplicate(update.content_hash):
                if update.content_hash not in seen_hashes:
                    new_updates.append(update)
                    seen_hashes.add(update.content_hash)
        return new_updates
    
    def create_update_embed(self, update: BBUpdate) -> discord.Embed:
        categories = self.analyzer.categorize_update(update)
        importance = self.analyzer.analyze_strategic_importance(update)
        colors = {
            1: 0x95a5a6, 2: 0x3498db, 3: 0x2ecc71, 4: 0xf39c12, 5: 0xe74c3c
        }
        color = colors.get(min(importance // 2 + 1, 5), 0x95a5a6)
        title = update.title[:256] if len(update.title) <= 256 else update.title[:253] + "..."
        description = update.description[:2048] if len(update.description) <= 2048 else update.description[:2045] + "..."
        embed = discord.Embed(
            title=title,
            description=description,
            color=color,
            url=update.link,
            timestamp=update.pub_date
        )
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
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found")
                return
            embed = self.create_update_embed(update)
            await channel.send(embed=embed)
            logger.info(f"Posted update '{update.title}' to channel {channel_id}")
        except Exception as e:
            logger.error(f"Error sending update to channel: {e}")
    
    @tasks.loop(minutes=2)
    async def check_rss_feed(self):
        try:
            logger.info("Checking RSS feed for new updates...")
            feed = feedparser.parse(self.rss_url)
            entries = feed.entries
            updates = self.process_rss_entries(entries)
            new_updates = self.filter_duplicates(updates)
            logger.info(f"Found {len(new_updates)} new updates")
            for update in new_updates:
                categories = self.analyzer.categorize_update(update)
                importance = self.analyzer.analyze_strategic_importance(update)
                self.db.store_update(update, importance_score=importance, categories=categories)
                await self.send_update_to_channel(update)
                self.total_updates_processed += 1
            self.consecutive_errors = 0
            self.last_successful_check = datetime.now()
        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Error checking RSS feed: {e}")
            if self.consecutive_errors >= self.config.get('max_consecutive_errors', 10):
                logger.critical("Too many consecutive errors, shutting down for safety")
                await self.close()
    
    # Slash commands below
    
    @app_commands.command(name="status", description="Check the bot's status")
    async def status_slash(self, interaction: discord.Interaction):
        uptime = datetime.now() - self.last_successful_check
        await interaction.response.send_message(
            f"Big Brother Updates Bot is online.\n"
            f"Total updates processed: {self.total_updates_processed}\n"
            f"Last successful feed check: {self.last_successful_check.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Uptime since last successful check: {str(uptime).split('.')[0]}",
            ephemeral=True
        )
    
    @app_commands.command(name="summary", description="Get today's summary of updates")
    async def summary_slash(self, interaction: discord.Interaction):
        try:
            updates = self.db.get_recent_updates(hours=24)
            if not updates:
                await interaction.response.send_message("No updates found for the last 24 hours.", ephemeral=True)
                return
            messages = []
            for update in updates[:10]:
                messages.append(f"• [{update.title}]({update.link})")
            summary_text = "\n".join(messages)
            await interaction.response.send_message(f"**Today's Big Brother Updates Summary:**\n{summary_text}", ephemeral=True)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            await interaction.response.send_message("Error retrieving summary.", ephemeral=True)
    
    @app_commands.command(name="setchannel", description="Set the updates channel")
    async def setchannel_slash(self, interaction: discord.Interaction, channel: discord.TextChannel):
        try:
            self.config.set("update_channel_id", channel.id)
            # Save config to file or environment here if desired
            await interaction.response.send_message(f"Update channel set to {channel.mention}", ephemeral=True)
        except Exception as e:
            logger.error(f"Error setting channel: {e}")
            await interaction.response.send_message("Failed to set update channel.", ephemeral=True)
    
    @app_commands.command(name="commands", description="List all available commands")
    async def commands_slash(self, interaction: discord.Interaction):
        commands_list = [
            "/status - Check the bot's status",
            "/summary - Get today's update summary",
            "/setchannel - Set the channel for update posts",
            "/commands - List all commands"
        ]
        await interaction.response.send_message("Available commands:\n" + "\n".join(commands_list), ephemeral=True)
    
    @app_commands.command(name="testbatch", description="Test batch posting of updates")
    async def test_updates(self, interaction: discord.Interaction):
        await interaction.response.send_message("Test batch command received. Functionality not implemented yet.", ephemeral=True)

def main():
    bot = BBDiscordBot()
    bot.run(bot.config.get('bot_token'))

if __name__ == "__main__":
    main()
