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
from datetime import datetime, timedelta, timezone
from typing import List
import logging
from dataclasses import dataclass
import json
import traceback
from pathlib import Path

# Suppress PyNaCl warning since voice is not used
import warnings
warnings.filterwarnings("ignore", message="PyNaCl is not installed, voice will NOT be supported")

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
                  update.link, update.pub_date.isoformat(), update.author, importance_score, categories_str))
            
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
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            cursor.execute("""
                SELECT title, description, link, pub_date, content_hash, author
                FROM updates 
                WHERE pub_date > ?
                ORDER BY pub_date DESC
            """, (cutoff_time.isoformat(),))
            
            results = cursor.fetchall()
            conn.close()
            
            updates = []
            for row in results:
                # row: title, description, link, pub_date, content_hash, author
                pub_date = datetime.fromisoformat(row[3]).replace(tzinfo=timezone.utc)
                updates.append(BBUpdate(
                    title=row[0],
                    description=row[1],
                    link=row[2],
                    pub_date=pub_date,
                    content_hash=row[4],
                    author=row[5]
                ))
            
            return updates
            
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
        self.last_successful_check = datetime.now(timezone.utc)
        self.total_updates_processed = 0
        self.consecutive_errors = 0
        
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Use the built-in app_commands tree, no need to reassign (avoid duplicate error)
        self.tree = self.tree
        
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
            
            # Sync commands on startup
            await self.tree.sync()
            logger.info("Slash commands synced")
            
        except Exception as e:
            logger.error(f"Error starting background tasks or syncing commands: {e}")
    
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
                
                pub_date = datetime.now(timezone.utc)
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                
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
        
        title = update.title[:256] if len(update.title) <= 256 else update.title[:
