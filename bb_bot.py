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
        
        # Enhanced features
        self.bb27_houseguests = [
            "Angela", "Tucker", "Makensy", "Cam", "Chelsie", 
            "Rubina", "Kimo", "Leah", "Quinn", "Joseph",
            "T'Kor", "Cedric", "Brooklyn", "Kenney", "Lisa"
        ]
        
        self.sentiment_positive = [
            'happy', 'excited', 'love', 'amazing', 'great', 'wonderful', 
            'perfect', 'fantastic', 'awesome', 'good', 'fun', 'laugh'
        ]
        
        self.sentiment_negative = [
            'angry', 'hate', 'mad', 'upset', 'annoyed', 'frustrated',
            'terrible', 'awful', 'bad', 'worst', 'annoying', 'drama'
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
        mentioned_houseguests = []
        text_lower = text.lower()
        
        for houseguest in self.bb27_houseguests:
            if houseguest.lower() in text_lower:
                mentioned_houseguests.append(houseguest)
        
        return mentioned_houseguests
    
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
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        text_lower = text.lower()
        
        positive_score = sum(1 for word in self.sentiment_positive if word in text_lower)
        negative_score = sum(1 for word in self.sentiment_negative if word in text_lower)
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        positive_ratio = positive_score / total_words
        negative_ratio = negative_score / total_words
        neutral_ratio = 1 - (positive_ratio + negative_ratio)
        
        return {
            "positive": positive_ratio,
            "negative": negative_ratio, 
            "neutral": max(0, neutral_ratio)
        }

class BBDatabase:
    """Handles database operations"""
    
    def __init__(self, db_path: str = "bb_updates.db"):
        self.db_path = db_path
        self.connection_timeout = 30
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
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
        """Initialize the database schema"""
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
    """Main Discord bot class"""
    
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
        houseguests = self.analyzer.extract_houseguests(f"{update.title} {update.description}")
        sentiment = self.analyzer.analyze_sentiment(f"{update.title} {update.description}")
        
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
        
        # Add sentiment indicator
        if sentiment["positive"] > 0.1:
            embed.add_field(name="House Mood", value="😊 Positive", inline=True)
        elif sentiment["negative"] > 0.1:
            embed.add_field(name="House Mood", value="😤 Tense", inline=True)
        else:
            embed.add_field(name="House Mood", value="😐 Neutral", inline=True)
        
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
                title=f"📊 Big Brother Updates Summary ({hours}h)",
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
                title="🤖 Big Brother Bot Status",
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
    
    # TEST COMMANDS
    @commands.command(name='testanalyzer')
    async def test_analyzer(self, ctx):
        """Test the AI analyzer with sample updates"""
        
        sample_updates = [
            {
                "title": "HOH Competition Results - Week 3",
                "description": "Sarah wins Head of Household competition after intense endurance challenge. She's already talking about backdooring Michael who has been getting too close to everyone. The house is buzzing with speculation about nominations.",
                "author": "BBUpdater1"
            },
            {
                "title": "Kitchen Blowup",
                "description": "Huge argument between Lisa and Michael over dirty dishes. Lisa called Michael out for being messy and not doing his share. Michael got defensive and things escalated quickly with other houseguests taking sides.",
                "author": "DramaAlert"
            },
            {
                "title": "Showmance Alert",
                "description": "Things are heating up between Jake and Amanda. They were spotted cuddling in the hammock and had a long romantic conversation under the stars. Other houseguests are starting to notice their connection.",
                "author": "LoveWatcher"
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
            sentiment = self.analyzer.analyze_sentiment(f"{mock_update.title} {mock_update.description}")
            
            analysis = []
            analysis.append(f"**Categories:** {' | '.join(categories)}")
            analysis.append(f"**Importance:** {'⭐' * importance} ({importance}/10)")
            
            if sentiment["positive"] > 0.1:
                analysis.append(f"**Sentiment:** 😊 Positive")
            elif sentiment["negative"] > 0.1:
                analysis.append(f"**Sentiment:** 😤 Negative")
            else:
                analysis.append(f"**Sentiment:** 😐 Neutral")
            
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
        """Test how Discord embeds will look"""
        
        strategic_update = BBUpdate(
            title="BREAKING: Massive Backdoor Plan Revealed",
            description="Sarah wins HOH and immediately starts planning to backdoor Michael. She's been building this plan for weeks with her secret alliance. Jessica and David are fully on board. They plan to nominate pawns initially, then use veto ceremony to put Michael on the block.",
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
            description="Lisa and Michael's argument about dishes turned into a house-wide conflict. Lisa accused Michael of being lazy and entitled. Michael fired back calling Lisa controlling and dramatic. Other houseguests are picking sides and the tension is thick.",
            link="https://jokersupdates.com/drama-update",
            pub_date=datetime.now(),
            content_hash="drama_test",
            author="DramaAlert"
        )
        
        drama_embed = self.create_update_embed(drama_update)
        await ctx.send("**💥 Drama Update Example:**", embed=drama_embed)
    
    @commands.command(name='testhelp')
    async def test_help(self, ctx):
        """Show all available test commands"""
        
        embed = discord.Embed(
            title="🧪 Big Brother Bot Test Commands",
            description="Test the AI analysis features",
            color=0x9b59b6
        )
        
        embed.add_field(
            name="!bbtestembeds", 
            value="See how different update types will look in Discord",
            inline=False
        )
        
        embed.add_field(
            name="!bbsummary [hours]",
            value="Generate summary of updates (works with test data)",
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
                await ctx.send("Command not found. Use `!bbcommands` for available commands.")
            else:
                logger.error(f"Command error: {error}")
                await ctx.send("An error occurred while processing the command.")
        
        @bot.command(name='commands')
        async def commands_help(ctx):
            """Show available commands"""
            embed = discord.Embed(
                title="🏠 Big Brother Bot Commands",
                description="AI-powered Big Brother analysis",
                color=0x3498db
            )
            
            embed.add_field(name="**Main Commands**", value="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", inline=False)
            embed.add_field(name="!bbsummary [hours]", value="Generate summary of updates", inline=False)
            embed.add_field(name="!bbstatus", value="Show bot status", inline=False)
            embed.add_field(name="!bbsetchannel [ID]", value="Set update channel (Admin only)", inline=False)
            
            embed.add_field(name="**Test Commands**", value="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", inline=False)
            embed.add_field(name="!bbtesthelp", value="Show all test commands", inline=False)
            embed.add_field(name="!bbtestanalyzer", value="Test AI analysis with sample updates", inline=False)
            embed.add_field(name="!bbtestembeds", value="See how updates will look in Discord", inline=False)
            
            embed.set_footer(text="Use test commands to try out features before Big Brother starts!")
            
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
    main()!bbtestanalyzer",
            value="Test AI analysis with sample Big Brother updates",
            inline=False
        )
        
        embed.add_field(
            name="
