import discord
from discord.ext import commands, tasks
import feedparser
import asyncio
import sqlite3
import hashlib
import re
import os
import sys
import signal
from datetime import datetime, timedelta
from typing import List, Dict, Set
import logging
from dataclasses import dataclass
import json
import time
import traceback
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging for 24/7 operation"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # File handler for all logs
    file_handler = logging.FileHandler(log_dir / "bb_bot.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Error file handler
    error_handler = logging.FileHandler(log_dir / "bb_bot_errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

logger = setup_logging()

class Config:
    """Configuration management for the bot"""
    
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
        """Load configuration from environment variables or file"""
        # Try environment variables first (for cloud deployment)
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
        
        # If no bot token from environment, try config file
        if not config["bot_token"] and self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        
        return config
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value"""
        self.config[key] = value

@dataclass
class BBUpdate:
    """Represents a Big Brother update"""
    title: str
    description: str
    link: str
    pub_date: datetime
    content_hash: str
    author: str = ""

class BBAnalyzer:
    """Analyzes Big Brother updates for strategic insights"""
    
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
        """Categorize an update based on its content"""
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
        """Extract houseguest names from text"""
        houseguest_pattern = r'\b[A-Z][a-z]+\b'
        potential_names = re.findall(houseguest_pattern, text)
        
        exclude_words = {'The', 'This', 'That', 'They', 'Some', 'Many', 'Other', 'First', 'Last'}
        return [name for name in potential_names if name not in exclude_words]
    
    def analyze_strategic_importance(self, update: BBUpdate) -> int:
        """Rate strategic importance from 1-10"""
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
    """Handles database operations with connection pooling and error recovery"""
    
    def __init__(self, db_path: str = "bb_updates.db"):
        self.db_path = db_path
        self.connection_timeout = 30
        self.init_database()
    
    def get_connection(self):
        """Get database connection with proper error handling"""
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
        """Initialize the database schema with proper indexing"""
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
        """Check if update already exists"""
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
        """Store a new update"""
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
        """Get updates from the last N hours"""
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
    """Main Discord bot class with 24/7 reliability features"""
    
    def __init__(self):
        self.config = Config()
        
        if not self.config.get('bot_token'):
            logger.error("Bot token not configured! Please set BOT_TOKEN environment variable")
            sys.exit(1)
        
        intents = discord.Intents.default()
        intents.message_content = True
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
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_shutting_down = True
        asyncio.create_task(self.close())
    
    async def on_ready(self):
        """Bot startup event"""
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')
        
        try:
            self.check_rss_feed.start()
            logger.info("RSS feed monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    def create_content_hash(self, title: str, description: str) -> str:
        """Create a unique hash for content deduplication"""
        content = f"{title}|{description}".lower()
        content = re.sub(r'\d{1,2}:\d{2}[ap]m', '', content)
        content = re.sub(r'\d{1,2}/\d{1,2}', '', content)
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_rss_entries(self, entries) -> List[BBUpdate]:
        """Process RSS entries into BBUpdate objects"""
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
        """Filter out duplicate updates"""
        new_updates = []
        seen_hashes = set()
        
        for update in updates:
            if not self.db.is_duplicate(update.content_hash):
                if update.content_hash not in seen_hashes:
                    new_updates.append(update)
                    seen_hashes.add(update.content_hash)
        
        return new_updates
    
    def create_update_embed(self, update: BBUpdate) -> discord.Embed:
        """Create a Discord embed for an update"""
        categories = self.analyzer.categorize_update(update)
        importance = self.analyzer.analyze_strategic_importance(update)
        
        colors = {
            1: 0x95a5a6, 2: 0x3498db, 3: 0x2ecc71, 4: 0xf39c12, 5: 0xe74c3c
        }
        color = colors.get(min(importance // 2 + 1, 5), 0x95a5a6)
        
        title = update.title[:256] if len(update.title) <= 256 else update.title[:253] + "..."
        description = update.description[:2048] if len(update.description) <= 2048 else update.description[:2045] + "..."
        
        embed = discord.Embed(
            title="📊 Big Brother Updates Summary (Test)",
            description=f"**{len(mock_updates)} total updates** (simulated data)",
            color=0x3498db,
            timestamp=datetime.now()
        )
        
        for category, cat_updates in categories.items():
            top_updates = sorted(cat_updates, 
                               key=lambda x: self.analyzer.analyze_strategic_importance(x), 
                               reverse=True)[:3]
            
            summary_text = "\n".join([f"• {update.title}" for update in top_updates])
            
            embed.add_field(
                name=f"{category} ({len(cat_updates)} updates)",
                value=summary_text or "No updates",
                inline=False
            )
        
        embed.set_footer(text="This is a test summary with mock data")
        await ctx.send(embed=embed)
    
    @commands.command(name='testfeed')
    async def test_feed_parsing(self, ctx):
        """Test RSS feed parsing (if available)"""
        
        try:
            await ctx.send("🔍 Testing RSS feed connection...")
            
            feed = feedparser.parse(self.rss_url)
            
            embed = discord.Embed(
                title="📡 RSS Feed Test Results",
                color=0x2ecc71 if feed.entries else 0xe74c3c,
                timestamp=datetime.now()
            )
            
            embed.add_field(name="Feed URL", value=self.rss_url, inline=False)
            embed.add_field(name="Feed Title", value=feed.feed.get('title', 'No title'), inline=True)
            embed.add_field(name="Total Entries", value=str(len(feed.entries)), inline=True)
            
            if feed.entries:
                embed.add_field(name="Status", value="✅ Feed accessible", inline=True)
                
                sample_entries = feed.entries[:3]
                for i, entry in enumerate(sample_entries, 1):
                    title = entry.get('title', 'No title')[:100]
                    description = entry.get('description', 'No description')[:150]
                    
                    embed.add_field(
                        name=f"Sample Entry #{i}",
                        value=f"**Title:** {title}...\n**Description:** {description}...",
                        inline=False
                    )
            else:
                embed.add_field(name="Status", value="❌ No entries found", inline=True)
                embed.add_field(name="Note", value="This is normal when Big Brother is not currently airing", inline=False)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            error_embed = discord.Embed(
                title="❌ RSS Feed Test Failed",
                description=f"Error: {str(e)}",
                color=0xe74c3c,
                timestamp=datetime.now()
            )
            await ctx.send(embed=error_embed)
    
    @commands.command(name='testhelp')
    async def test_help(self, ctx):
        """Show all available test commands"""
        
        embed = discord.Embed(
            title="🧪 Big Brother Bot Test Commands",
            description="Test the AI analysis features before Big Brother season starts",
            color=0x9b59b6
        )
        
        embed.add_field(
            name="!bbtestanalyzer",
            value="Test AI analysis with sample Big Brother updates",
            inline=False
        )
        
        embed.add_field(
            name="!bbtestembeds", 
            value="See how different update types will look in Discord",
            inline=False
        )
        
        embed.add_field(
            name="!bbtestsummary",
            value="Test summary generation with mock data",
            inline=False
        )
        
        embed.add_field(
            name="!bbtestfeed",
            value="Test RSS feed connection and parsing",
            inline=False
        )
        
        embed.set_footer(text="These commands work without needing live Big Brother updates!")
        
        await ctx.send(embed=embed)

def main():
    """Main function to run the bot"""
    try:
        bot = BBDiscordBot()
        
        @bot.event
        async def on_command_error(ctx, error):
            if isinstance(error, commands.MissingPermissions):
                await ctx.send("You don't have permission to use this command.")
            elif isinstance(error, commands.CommandNotFound):
                await ctx.send("Command not found. Use `!bbcommands` or `!bbtesthelp` for available commands.")
            else:
                logger.error(f"Command error: {error}")
                await ctx.send("An error occurred while processing the command.")
        
        @bot.command(name='commands')
        async def commands_help(ctx):
            """Show main bot commands"""
            embed = discord.Embed(
                title="🏠 Big Brother Bot Commands",
                description="Monitor Jokers Updates RSS feed with intelligent analysis",
                color=0x3498db
            )
            
            embed.add_field(name="**Main Commands**", value="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", inline=False)
            embed.add_field(name="!bbsummary [hours]", value="Generate summary of updates (default: 24 hours)", inline=False)
            embed.add_field(name="!bbstatus", value="Show bot status and performance metrics", inline=False)
            embed.add_field(name="!bbsetchannel [ID]", value="Set update channel (Admin only)", inline=False)
            
            embed.add_field(name="**Test Commands**", value="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", inline=False)
            embed.add_field(name="!bbtesthelp", value="Show all test commands for trying out AI features", inline=False)
            embed.add_field(name="!bbtestanalyzer", value="Test AI analysis with sample updates", inline=False)
            embed.add_field(name="!bbtestembeds", value="See how updates will look in Discord", inline=False)
            
            embed.set_footer(text="Use test commands to try out features before Big Brother season starts!")
            
            await ctx.send(embed=embed)
        
        bot_token = bot.config.get('bot_token')
        if not bot_token:
            logger.error("No bot token found!")
            return
        
        logger.info("Starting Big Brother Discord Bot...")
        bot.run(bot_token, reconnect=True)
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    main()Embed(
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
        """Send an update to the configured channel"""
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
            
        except Exception as e:
            logger.error(f"Error sending update to channel: {e}")
    
    @tasks.loop(minutes=2)
    async def check_rss_feed(self):
        """Check RSS feed for new updates"""
        if self.is_shutting_down:
            return
        
        try:
            feed = feedparser.parse(self.rss_url)
            
            if not feed.entries:
                logger.warning("No entries returned from RSS feed")
                return
            
            updates = self.process_rss_entries(feed.entries)
            new_updates = self.filter_duplicates(updates)
            
            for update in new_updates:
                try:
                    categories = self.analyzer.categorize_update(update)
                    importance = self.analyzer.analyze_strategic_importance(update)
                    
                    self.db.store_update(update, importance, categories)
                    await self.send_update_to_channel(update)
                    
                    self.total_updates_processed += 1
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing update: {e}")
                    self.consecutive_errors += 1
            
            self.last_successful_check = datetime.now()
            self.consecutive_errors = 0
            
            if new_updates:
                logger.info(f"Processed {len(new_updates)} new updates")
                
        except Exception as e:
            logger.error(f"Error in RSS check: {e}")
            self.consecutive_errors += 1
    
    @commands.command(name='summary')
    async def daily_summary(self, ctx, hours: int = 24):
        """Generate a summary of updates"""
        try:
            if hours < 1 or hours > 168:
                await ctx.send("Hours must be between 1 and 168")
                return
                
            updates = self.db.get_recent_updates(hours)
            
            if not updates:
                await ctx.send(f"No updates found in the last {hours} hours.")
                return
            
            categories = {}
            for update in updates:
                update_categories = self.analyzer.categorize_update(update)
                for category in update_categories:
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(update)
            
            embed = discord.Embed(
                title=f"Big Brother Updates Summary ({hours}h)",
                description=f"**{len(updates)} total updates**",
                color=0x3498db,
                timestamp=datetime.now()
            )
            
            for category, cat_updates in categories.items():
                top_updates = sorted(cat_updates, 
                                   key=lambda x: self.analyzer.analyze_strategic_importance(x), 
                                   reverse=True)[:3]
                
                summary_text = "\n".join([f"• {update.title[:100]}..." 
                                        if len(update.title) > 100 
                                        else f"• {update.title}" 
                                        for update in top_updates])
                
                embed.add_field(
                    name=f"{category} ({len(cat_updates)} updates)",
                    value=summary_text or "No updates",
                    inline=False
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            await ctx.send("Error generating summary. Please try again.")
    
    @commands.command(name='setchannel')
    @commands.has_permissions(administrator=True)
    async def set_update_channel(self, ctx, channel_id: int):
        """Set the channel for RSS updates"""
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                await ctx.send(f"Channel with ID {channel_id} not found.")
                return
            
            if not channel.permissions_for(ctx.guild.me).send_messages:
                await ctx.send(f"I don't have permission to send messages in <#{channel_id}>")
                return
            
            self.config.set('update_channel_id', channel_id)
            await ctx.send(f"Update channel set to <#{channel_id}>")
            logger.info(f"Update channel set to {channel_id}")
            
        except Exception as e:
            logger.error(f"Error setting channel: {e}")
            await ctx.send("Error setting channel. Please try again.")
    
    @commands.command(name='status')
    async def bot_status(self, ctx):
        """Show bot status"""
        try:
            embed = discord.Embed(
                title="Big Brother Bot Status",
                color=0x2ecc71 if self.consecutive_errors == 0 else 0xe74c3c,
                timestamp=datetime.now()
            )
            
            embed.add_field(name="RSS Feed", value=self.rss_url, inline=False)
            embed.add_field(name="Update Channel", 
                           value=f"<#{self.config.get('update_channel_id')}>" if self.config.get('update_channel_id') else "Not set", 
                           inline=True)
            embed.add_field(name="Updates Processed", value=str(self.total_updates_processed), inline=True)
            embed.add_field(name="Consecutive Errors", value=str(self.consecutive_errors), inline=True)
            
            time_since_check = datetime.now() - self.last_successful_check
            embed.add_field(name="Last RSS Check", value=f"{time_since_check.total_seconds():.0f} seconds ago", inline=True)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating status: {e}")
            await ctx.send("Error generating status.")
    
    # TEST COMMANDS - These work without live Big Brother updates
    @commands.command(name='testanalyzer')
    async def test_analyzer(self, ctx):
        """Test the AI analyzer with sample Big Brother updates"""
        
        sample_updates = [
            {
                "title": "HOH Competition Results - Week 3",
                "description": "Sarah wins Head of Household competition after intense endurance challenge. She's already talking about backdooring Michael who has been getting too close to everyone. The house is buzzing with speculation about nominations.",
                "author": "BBUpdater1"
            },
            {
                "title": "Late Night Strategy Talk",
                "description": "Alliance meeting in the backyard between Sarah, Jessica, and David. They're planning to nominate Michael and Lisa, with Michael being the real target. They want to keep the backdoor plan secret until after veto ceremony.",
                "author": "NightOwlWatcher"
            },
            {
                "title": "Showmance Alert",
                "description": "Things are heating up between Jake and Amanda. They were spotted cuddling in the hammock and had a long romantic conversation under the stars. Other houseguests are starting to notice their connection.",
                "author": "LoveWatcher"
            },
            {
                "title": "Kitchen Blowup",
                "description": "Huge argument between Lisa and Michael over dirty dishes. Lisa called Michael out for being messy and not doing his share. Michael got defensive and things escalated quickly with other houseguests taking sides.",
                "author": "DramaAlert"
            },
            {
                "title": "Power of Veto Competition",
                "description": "Michael wins the Power of Veto in a puzzle competition, completely ruining Sarah's backdoor plan. The house dynamics are about to shift dramatically as Michael now has the power to save himself.",
                "author": "CompWatcher"
            }
        ]
        
        embed = discord.Embed(
            title="🧪 AI Analyzer Test Results",
            description="Testing Big Brother update analysis with sample data",
            color=0x9b59b6,
            timestamp=datetime.now()
        )
        
        for i, sample in enumerate(sample_updates, 1):
            mock_update = BBUpdate(
                title=sample["title"],
                description=sample["description"],
                link=f"https://example.com/update{i}",
                pub_date=datetime.now(),
                content_hash=f"test_hash_{i}",
                author=sample["author"]
            )
            
            categories = self.analyzer.categorize_update(mock_update)
            importance = self.analyzer.analyze_strategic_importance(mock_update)
            houseguests = self.analyzer.extract_houseguests(f"{mock_update.title} {mock_update.description}")
            
            analysis = []
            analysis.append(f"**Categories:** {' | '.join(categories)}")
            analysis.append(f"**Importance:** {'⭐' * importance} ({importance}/10)")
            if houseguests:
                analysis.append(f"**Houseguests:** {', '.join(houseguests[:3])}")
            
            embed.add_field(
                name=f"#{i}: {sample['title'][:50]}...",
                value="\n".join(analysis),
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='testembeds')
    async def test_embeds(self, ctx):
        """Test how Discord embeds will look for different update types"""
        
        strategic_update = BBUpdate(
            title="BREAKING: Massive Backdoor Plan Revealed",
            description="Sarah wins HOH and immediately starts planning to backdoor Michael. She's been building this plan for weeks with her secret alliance. Jessica and David are fully on board. They plan to nominate pawns initially, then use veto ceremony to put Michael on the block. This could completely flip the house dynamics and break up the strongest duo.",
            link="https://jokersupdates.com/strategic-update",
            pub_date=datetime.now(),
            content_hash="strategic_test",
            author="StrategyWatcher"
        )
        
        strategic_embed = self.create_update_embed(strategic_update)
        await ctx.send("**🎯 High-Stakes Strategic Update Example:**", embed=strategic_embed)
        
        await asyncio.sleep(2)
        
        drama_update = BBUpdate(
            title="Kitchen Confrontation Escalates",
            description="Lisa and Michael's argument about dishes turned into a house-wide conflict. Lisa accused Michael of being lazy and entitled. Michael fired back calling Lisa controlling and dramatic. Other houseguests are picking sides with Jessica backing Lisa and Jake defending Michael. The tension is thick and this could affect upcoming votes.",
            link="https://jokersupdates.com/drama-update",
            pub_date=datetime.now(),
            content_hash="drama_test",
            author="DramaAlert"
        )
        
        drama_embed = self.create_update_embed(drama_update)
        await ctx.send("**💥 Drama Update Example:**", embed=drama_embed)
        
        await asyncio.sleep(2)
        
        romance_update = BBUpdate(
            title="Showmance Heating Up",
            description="Jake and Amanda's relationship is getting serious. They spent the entire evening talking privately in the backyard, sharing personal stories and getting very cozy. Other houseguests are starting to notice their connection. Amanda mentioned feeling 'butterflies' when talking to the cameras. This showmance could become a powerful duo or a big target.",
            link="https://jokersupdates.com/romance-update",
            pub_date=datetime.now(),
            content_hash="romance_test",
            author="LoveWatcher"
        )
        
        romance_embed = self.create_update_embed(romance_update)
        await ctx.send("**💕 Romance Update Example:**", embed=romance_embed)
    
    @commands.command(name='testsummary')
    async def test_summary(self, ctx):
        """Test the summary generation with mock data"""
        
        mock_updates = [
            BBUpdate("HOH Competition Results", "Sarah wins endurance challenge, planning backdoor strategy", "https://example.com/1", datetime.now(), "hash1", "User1"),
            BBUpdate("Veto Competition", "Michael wins veto, saves himself from backdoor plan", "https://example.com/2", datetime.now(), "hash2", "User2"),
            BBUpdate("Alliance Meeting", "Secret alliance discusses new targets after veto win", "https://example.com/3", datetime.now(), "hash3", "User3"),
            BBUpdate("Kitchen Drama", "Lisa and Michael argue about house responsibilities", "https://example.com/4", datetime.now(), "hash4", "User4"),
            BBUpdate("Showmance Update", "Jake and Amanda getting closer, other houseguests notice", "https://example.com/5", datetime.now(), "hash5", "User5"),
            BBUpdate("Late Night Strategy", "Jessica and David plan their next moves", "https://example.com/6", datetime.now(), "hash6", "User6"),
        ]
        
        categories = {}
        for update in mock_updates:
            update_categories = self.analyzer.categorize_update(update)
            for category in update_categories:
                if category not in categories:
                    categories[category] = []
                categories[category].append(update)
        
        embed = discord.
