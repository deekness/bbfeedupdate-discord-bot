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
            "rss_check_interval": 2,  # minutes
            "max_retries": 3,
            "retry_delay": 5,  # seconds
            "database_path": "bb_updates.db",
            "enable_heartbeat": True,
            "heartbeat_interval": 300,  # seconds (5 minutes)
            "max_update_age_hours": 168,  # 1 week
            "enable_auto_restart": True,
            "max_consecutive_errors": 10
        }
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged_config = self.default_config.copy()
                merged_config.update(config)
                return merged_config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return self.default_config.copy()
        else:
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def save_config(self, config: dict = None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value"""
        self.config[key] = value
        self.save_config()

class HealthMonitor:
    """Health monitoring and auto-recovery system"""
    
    def __init__(self, bot):
        self.bot = bot
        self.consecutive_errors = 0
        self.last_heartbeat = datetime.now()
        self.last_rss_success = datetime.now()
        self.total_updates_processed = 0
        self.start_time = datetime.now()
        self.error_log = []
        
    def record_error(self, error: Exception, context: str = ""):
        """Record an error for monitoring"""
        self.consecutive_errors += 1
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        self.error_log.append(error_entry)
        
        # Keep only last 100 errors
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]
        
        logger.error(f"Error in {context}: {error}")
        
        # Auto-restart if too many consecutive errors
        if (self.consecutive_errors >= self.bot.config.get('max_consecutive_errors', 10) and 
            self.bot.config.get('enable_auto_restart', True)):
            logger.critical("Too many consecutive errors, attempting restart...")
            self.restart_bot()
    
    def record_success(self, context: str = ""):
        """Record successful operation"""
        self.consecutive_errors = 0
        if context == "rss_check":
            self.last_rss_success = datetime.now()
        logger.debug(f"Success in {context}")
    
    def heartbeat(self):
        """Update heartbeat timestamp"""
        self.last_heartbeat = datetime.now()
        logger.info(f"Heartbeat: Bot healthy, processed {self.total_updates_processed} updates")
    
    def get_health_status(self) -> dict:
        """Get comprehensive health status"""
        now = datetime.now()
        uptime = now - self.start_time
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'uptime_human': str(uptime),
            'consecutive_errors': self.consecutive_errors,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'last_rss_success': self.last_rss_success.isoformat(),
            'total_updates_processed': self.total_updates_processed,
            'recent_errors': self.error_log[-5:] if self.error_log else [],
            'memory_usage': self.get_memory_usage()
        }
    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not installed'}
    
    def restart_bot(self):
        """Restart the bot process"""
        logger.critical("Restarting bot...")
        os.execv(sys.executable, ['python'] + sys.argv)

class ResilientRSSParser:
    """RSS parser with retry logic and error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=config.get('max_retries', 3),
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set timeout and user agent
        self.session.headers.update({
            'User-Agent': 'BB-Discord-Bot/1.0'
        })
    
    def parse_feed(self, url: str) -> List[dict]:
        """Parse RSS feed with resilient error handling"""
        max_retries = self.config.get('max_retries', 3)
        retry_delay = self.config.get('retry_delay', 5)
        
        for attempt in range(max_retries):
            try:
                # First try with requests session
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Parse with feedparser
                feed = feedparser.parse(response.content)
                
                if feed.bozo:
                    logger.warning(f"RSS feed has issues: {feed.bozo_exception}")
                
                if not feed.entries:
                    logger.warning("RSS feed returned no entries")
            @dataclass
class BBUpdate:
                
                logger.info(f"Successfully parsed {len(feed.entries)} entries from RSS feed")
                return feed.entries
                
            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
                
            except Exception as e:
                logger.error(f"Unexpected error parsing RSS feed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
        
        return []
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
        # This would be populated with actual houseguest names for the current season
        # For now, using placeholder pattern matching
        houseguest_pattern = r'\b[A-Z][a-z]+\b'
        potential_names = re.findall(houseguest_pattern, text)
        
        # Filter out common non-name words
        exclude_words = {'The', 'This', 'That', 'They', 'Some', 'Many', 'Other', 'First', 'Last'}
        return [name for name in potential_names if name not in exclude_words]
    
    def analyze_strategic_importance(self, update: BBUpdate) -> int:
        """Rate strategic importance from 1-10"""
        content = f"{update.title} {update.description}".lower()
        score = 1
        
        # High importance indicators
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
        self.setup_maintenance_tasks()
    
    def get_connection(self):
        """Get database connection with proper error handling"""
        try:
            conn = sqlite3.connect(
                self.db_path, 
                timeout=self.connection_timeout,
                check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            conn.execute("PRAGMA synchronous=NORMAL")  # Better performance
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
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_date DATE,
                    content TEXT,
                    update_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status_type TEXT,
                    status_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON updates(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pub_date ON updates(pub_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_at ON updates(processed_at)")
            
            conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
        finally:
            conn.close()
    
    def setup_maintenance_tasks(self):
        """Setup database maintenance tasks"""
        # This would be called periodically to clean up old data
        pass
    
    def cleanup_old_data(self, max_age_hours: int = 168):  # 1 week default
        """Clean up old database entries"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            # Clean old updates
            cursor.execute("DELETE FROM updates WHERE processed_at < ?", (cutoff_time,))
            deleted_updates = cursor.rowcount
            
            # Clean old summaries
            cursor.execute("DELETE FROM summaries WHERE created_at < ?", (cutoff_time,))
            deleted_summaries = cursor.rowcount
            
            # Clean old status entries
            cursor.execute("DELETE FROM bot_status WHERE created_at < ?", (cutoff_time,))
            deleted_status = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {deleted_updates} old updates, {deleted_summaries} old summaries, {deleted_status} old status entries")
            
        except Exception as e:
            logger.error(f"Database cleanup error: {e}")
    
    def is_duplicate(self, content_hash: str) -> bool:
        """Check if update already exists with better error handling"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT 1 FROM updates WHERE content_hash = ?", (content_hash,))
            result = cursor.fetchone()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            logger.error(f"Database duplicate check error: {e}")
            return False  # Assume not duplicate on error to avoid losing updates
    
    def store_update(self, update: BBUpdate, importance_score: int = 1, categories: List[str] = None):
        """Store a new update with enhanced data"""
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
            logger.debug(f"Stored update: {update.title[:50]}...")
            
        except Exception as e:
            logger.error(f"Database store error: {e}")
            raise
    
    def get_recent_updates(self, hours: int = 24) -> List[BBUpdate]:
        """Get updates from the last N hours with better error handling"""
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
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Total updates
            cursor.execute("SELECT COUNT(*) FROM updates")
            total_updates = cursor.fetchone()[0]
            
            # Updates today
            today = datetime.now().date()
            cursor.execute("SELECT COUNT(*) FROM updates WHERE DATE(processed_at) = ?", (today,))
            today_updates = cursor.fetchone()[0]
            
            # Updates this week
            week_ago = datetime.now() - timedelta(days=7)
            cursor.execute("SELECT COUNT(*) FROM updates WHERE processed_at > ?", (week_ago,))
            week_updates = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_updates': total_updates,
                'today_updates': today_updates,
                'week_updates': week_updates
            }
            
        except Exception as e:
            logger.error(f"Database stats error: {e}")
            return {'error': str(e)}

class BBDiscordBot(commands.Bot):
    """Main Discord bot class with 24/7 reliability features"""
    
    def __init__(self):
        # Load configuration
        self.config = Config()
        
        # Validate required config
        if not self.config.get('bot_token'):
            logger.error("Bot token not configured! Please set bot_token in config.json")
            sys.exit(1)
        
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!bb', intents=intents)
        
        self.rss_url = "https://rss.jokersupdates.com/ubbthreads/rss/bbusaupdates/rss.php"
        self.db = BBDatabase(self.config.get('database_path', 'bb_updates.db'))
        self.analyzer = BBAnalyzer()
        self.health_monitor = HealthMonitor(self)
        self.rss_parser = ResilientRSSParser(self.config)
        
        # Bot state
        self.is_shutting_down = False
        self.last_successful_check = datetime.now()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_shutting_down = True
        asyncio.create_task(self.close())
    
    async def on_ready(self):
        """Bot startup event with comprehensive initialization"""
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')
        
        # Start background tasks
        try:
            self.check_rss_feed.start()
            logger.info("RSS feed monitoring started")
            
            if self.config.get('enable_heartbeat', True):
                self.heartbeat_task.start()
                logger.info("Heartbeat monitoring started")
            
            # Start maintenance tasks
            self.maintenance_task.start()
            logger.info("Maintenance tasks started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
            self.health_monitor.record_error(e, "startup")
    
    async def on_disconnect(self):
        """Handle disconnect events"""
        logger.warning("Bot disconnected from Discord")
    
    async def on_resumed(self):
        """Handle resume events"""
        logger.info("Bot resumed Discord connection")
    
    async def on_error(self, event, *args, **kwargs):
        """Handle Discord API errors"""
        logger.error(f"Discord error in {event}: {traceback.format_exc()}")
        self.health_monitor.record_error(Exception(f"Discord error in {event}"), "discord_event")
    
    async def close(self):
        """Enhanced close method with cleanup"""
        logger.info("Shutting down bot...")
        
        # Stop background tasks
        if hasattr(self, 'check_rss_feed'):
            self.check_rss_feed.cancel()
        if hasattr(self, 'heartbeat_task'):
            self.heartbeat_task.cancel()
        if hasattr(self, 'maintenance_task'):
            self.maintenance_task.cancel()
        
        # Close database connections
        try:
            # Any cleanup needed for database
            pass
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
        
        # Close Discord connection
        await super().close()
        logger.info("Bot shutdown complete")
    
    @tasks.loop(minutes=1)  # Check every minute for better reliability
    async def check_rss_feed(self):
        """Periodically check the RSS feed for new updates with enhanced error handling"""
        if self.is_shutting_down:
            return
        
        try:
            # Get RSS entries
            entries = self.rss_parser.parse_feed(self.rss_url)
            
            if not entries:
                logger.warning("No entries returned from RSS feed")
                return
            
            # Process entries
            updates = self.process_rss_entries(entries)
            new_updates = self.filter_duplicates(updates)
            
            # Send updates to Discord
            for update in new_updates:
                try:
                    categories = self.analyzer.categorize_update(update)
                    importance = self.analyzer.analyze_strategic_importance(update)
                    
                    # Store in database
                    self.db.store_update(update, importance, categories)
                    
                    # Send to Discord
                    await self.send_update_to_channel(update)
                    
                    # Update counters
                    self.health_monitor.total_updates_processed += 1
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing individual update: {e}")
                    self.health_monitor.record_error(e, "update_processing")
            
            # Record successful check
            self.last_successful_check = datetime.now()
            self.health_monitor.record_success("rss_check")
            
            if new_updates:
                logger.info(f"Processed {len(new_updates)} new updates")
                
        except Exception as e:
            logger.error(f"Error in RSS check cycle: {e}")
            self.health_monitor.record_error(e, "rss_check")
    
    @tasks.loop(seconds=300)  # 5 minutes
    async def heartbeat_task(self):
        """Send periodic heartbeat and health check"""
        if self.is_shutting_down:
            return
        
        try:
            self.health_monitor.heartbeat()
            
            # Check if we haven't successfully checked RSS in too long
            time_since_success = datetime.now() - self.last_successful_check
            if time_since_success > timedelta(minutes=30):
                logger.warning(f"No successful RSS check in {time_since_success}")
                
                # Try to restart RSS task
                if self.check_rss_feed.is_running():
                    self.check_rss_feed.restart()
                    logger.info("Restarted RSS feed task")
                
        except Exception as e:
            logger.error(f"Error in heartbeat task: {e}")
            self.health_monitor.record_error(e, "heartbeat")
    
    @tasks.loop(hours=24)  # Daily maintenance
    async def maintenance_task(self):
        """Daily maintenance tasks"""
        if self.is_shutting_down:
            return
        
        try:
            logger.info("Running daily maintenance...")
            
            # Clean up old database entries
            max_age = self.config.get('max_update_age_hours', 168)
            self.db.cleanup_old_data(max_age)
            
            # Log statistics
            stats = self.db.get_stats()
            logger.info(f"Database stats: {stats}")
            
            # Reset error counters
            self.health_monitor.consecutive_errors = 0
            
            logger.info("Daily maintenance completed")
            
        except Exception as e:
            logger.error(f"Error in maintenance task: {e}")
            self.health_monitor.record_error(e, "maintenance")
    
    def process_rss_entries(self, entries) -> List[BBUpdate]:
        """Process RSS entries into BBUpdate objects"""
        updates = []
        
        for entry in entries:
            try:
                # Extract basic info
                title = entry.get('title', 'No title')
                description = entry.get('description', 'No description')
                link = entry.get('link', '')
                
                # Parse publication date
                pub_date = datetime.now()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                
                # Create content hash for deduplication
                content_hash = self.create_content_hash(title, description)
                
                # Extract author if available
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
        
        return updates2}[ap]m', '', content)
        content = re.sub(r'\d{1,2}/\d{1,2}', '', content)
        return hashlib.md5(content.encode()).hexdigest()
    
    def parse_rss_feed(self) -> List[BBUpdate]:
        """Parse the RSS feed and return new updates"""
        try:
            feed = feedparser.parse(self.rss_url)
            updates = []
            
            for entry in feed.entries:
                # Extract basic info
                title = entry.get('title', 'No title')
                description = entry.get('description', 'No description')
                link = entry.get('link', '')
                
                # Parse publication date
                pub_date = datetime.now()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                
                # Create content hash for deduplication
                content_hash = self.create_content_hash(title, description)
                
                # Extract author if available
                author = entry.get('author', '')
                
                updates.append(BBUpdate(
                    title=title,
                    description=description,
                    link=link,
                    pub_date=pub_date,
                    content_hash=content_hash,
                    author=author
                ))
            
            return updates
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed: {e}")
            return []
    
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
        
        # Choose embed color based on importance
        colors = {
            1: 0x95a5a6,  # Light gray
            2: 0x3498db,  # Blue
            3: 0x2ecc71,  # Green
            4: 0xf39c12,  # Orange
            5: 0xe74c3c   # Red
        }
        color = colors.get(min(importance // 2 + 1, 5), 0x95a5a6)
        
        embed = discord.Embed(
            title=update.title[:256],  # Discord title limit
            description=update.description[:2048],  # Discord description limit
            color=color,
            url=update.link,
            timestamp=update.pub_date
        )
        
        # Add categories
        if categories:
            embed.add_field(
                name="Categories",
                value=" | ".join(categories),
                inline=True
            )
        
        # Add strategic importance
        importance_stars = "⭐" * importance
        embed.add_field(
            name="Strategic Importance",
            value=f"{importance_stars} ({importance}/10)",
            inline=True
        )
        
        # Add houseguests mentioned
        houseguests = self.analyzer.extract_houseguests(f"{update.title} {update.description}")
        if houseguests:
            embed.add_field(
                name="Houseguests Mentioned",
                value=", ".join(houseguests[:5]),  # Limit to 5
                inline=False
            )
        
        if update.author:
            embed.set_footer(text=f"Reported by: {update.author}")
        
        return embed
    
    async def send_update_to_channel(self, update: BBUpdate):
        """Send an update to the configured channel"""
        if not self.update_channel_id:
            return
        
        channel = self.get_channel(self.update_channel_id)
        if not channel:
            logger.error(f"Channel {self.update_channel_id} not found")
            return
        
        embed = self.create_update_embed(update)
        await channel.send(embed=embed)
    
    @tasks.loop(minutes=2)  # Check every 2 minutes
    async def check_rss_feed(self):
        """Periodically check the RSS feed for new updates"""
        try:
            updates = self.parse_rss_feed()
            new_updates = self.filter_duplicates(updates)
            
            for update in new_updates:
                # Store in database
                self.db.store_update(update)
                
                # Send to Discord
                await self.send_update_to_channel(update)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(1)
            
            if new_updates:
                logger.info(f"Processed {len(new_updates)} new updates")
                
        except Exception as e:
            logger.error(f"Error checking RSS feed: {e}")
    
    @commands.command(name='summary')
    async def daily_summary(self, ctx, hours: int = 24):
        """Generate a summary of updates from the last N hours"""
        try:
            updates = self.db.get_recent_updates(hours)
            
            if not updates:
                await ctx.send(f"No updates found in the last {hours} hours.")
                return
            
            # Categorize updates
            categories = {}
            for update in updates:
                update_categories = self.analyzer.categorize_update(update)
                for category in update_categories:
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(update)
            
            # Create summary embed
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
        self.update_channel_id = channel_id
        await ctx.send(f"Update channel set to <#{channel_id}>")
    
    @commands.command(name='status')
    async def bot_status(self, ctx):
        """Show bot status"""
        recent_updates = self.db.get_recent_updates(1)  # Last hour
        
        embed = discord.Embed(
            title="Big Brother Bot Status",
            color=0x2ecc71,
            timestamp=datetime.now()
        )
        
        embed.add_field(
            name="RSS Feed",
            value=self.rss_url,
            inline=False
        )
        
        embed.add_field(
            name="Update Channel",
            value=f"<#{self.update_channel_id}>" if self.update_channel_id else "Not set",
            inline=True
        )
        
        embed.add_field(
            name="Updates (Last Hour)",
            value=str(len(recent_updates)),
            inline=True
        )
        
        embed.add_field(
            name="Next Check",
            value="Every 2 minutes",
            inline=True
        )
        
        await ctx.send(embed=embed)

# Bot setup and configuration
def main():
    """Main function to run the bot"""
    bot = BBDiscordBot()
    
    # Add error handling
    @bot.event
    async def on_command_error(ctx, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.send("You don't have permission to use this command.")
        elif isinstance(error, commands.CommandNotFound):
            await ctx.send("Command not found. Use `!bbhelp` for available commands.")
        else:
            logger.error(f"Command error: {error}")
            await ctx.send("An error occurred while processing the command.")
    
    # Custom help command
    @bot.command(name='help')
    async def help_command(ctx):
        """Show available commands"""
        embed = discord.Embed(
            title="Big Brother Bot Commands",
            description="Monitor Jokers Updates RSS feed with intelligent analysis",
            color=0x3498db
        )
        
        embed.add_field(
            name="!bbsummary [hours]",
            value="Generate summary of updates (default: 24 hours)",
            inline=False
        )
        
        embed.add_field(
            name="!bbstatus",
            value="Show bot status and configuration",
            inline=False
        )
        
        embed.add_field(
            name="!bbsetchannel [channel_id]",
            value="Set channel for RSS updates (Admin only)",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    # Run the bot
    # bot.run('YOUR_BOT_TOKEN')  # Replace with your actual token

if __name__ == "__main__":
    main()
