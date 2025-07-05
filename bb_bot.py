import discord
from discord.ext import commands, tasks
import feedparser
import asyncio
import aiosqlite
import hashlib
import re
import os
import sys
import signal
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
import logging
from dataclasses import dataclass, field
import json
import time
import traceback
from pathlib import Path
import aiohttp
from collections import defaultdict, Counter
import difflib

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging for 24/7 operation"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # File handler for all logs with rotation
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_dir / "bb_bot.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Error file handler
    error_handler = RotatingFileHandler(
        log_dir / "bb_bot_errors.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
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
            "max_consecutive_errors": 10,
            "enable_smart_names": True,
            "min_name_confidence": 0.7,
            "cache_ttl": 3600
        }
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load configuration from environment variables or file"""
        # Try environment variables first (for cloud deployment)
        config = self.default_config.copy()
        
        # Load from environment
        for key in config:
            env_key = key.upper()
            env_value = os.getenv(env_key)
            if env_value is not None:
                if isinstance(config[key], bool):
                    config[key] = env_value.lower() == 'true'
                elif isinstance(config[key], int):
                    try:
                        config[key] = int(env_value)
                    except ValueError:
                        logger.warning(f"Invalid int value for {env_key}: {env_value}")
                elif isinstance(config[key], float):
                    try:
                        config[key] = float(env_value)
                    except ValueError:
                        logger.warning(f"Invalid float value for {env_key}: {env_value}")
                else:
                    config[key] = env_value
        
        # If no bot token from environment, try config file
        if not config["bot_token"] and self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    for key, value in file_config.items():
                        if key in config and value:
                            config[key] = value
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
    mentioned_houseguests: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    importance_score: int = 1

class HouseguestTracker:
    """Smart houseguest name recognition and tracking"""
    
    def __init__(self):
        self.known_houseguests: Dict[str, Dict[str, any]] = {}
        self.nickname_map: Dict[str, str] = {}
        self.name_patterns = {
            'full_name': re.compile(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'),
            'single_name': re.compile(r'\b([A-Z][a-z]+)\b'),
            'nickname': re.compile(r'\b([A-Z][a-z]+y|[A-Z][a-z]+ie)\b'),  # Common nickname patterns
            'quoted_name': re.compile(r'"([A-Z][a-z]+)"'),
            'possessive': re.compile(r"\b([A-Z][a-z]+)'s\b")
        }
        self.context_indicators = [
            'told', 'said', 'thinks', 'wants', 'nominated', 'won', 'lost',
            'aligned', 'targeting', 'alliance', 'showmance'
        ]
        self.exclude_words = {
            'The', 'This', 'That', 'They', 'Some', 'Many', 'Other', 'First', 'Last',
            'Big', 'Brother', 'Head', 'Household', 'Power', 'Veto', 'Have', 'Not',
            'America', 'Live', 'Feeds', 'Update', 'Quick', 'Game', 'House', 'Diary',
            'Room', 'Competition', 'Ceremony', 'Meeting', 'Night', 'Morning', 'Evening',
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        }
        self.name_frequency = Counter()
        self.name_associations = defaultdict(set)
        
    def learn_houseguest(self, name: str, context: str = "", confidence: float = 1.0):
        """Learn a new houseguest name with context"""
        canonical = name.lower()
        
        if canonical not in self.known_houseguests:
            self.known_houseguests[canonical] = {
                'display_name': name,
                'aliases': {name},
                'mention_count': 0,
                'first_seen': datetime.now(),
                'contexts': []
            }
        
        self.known_houseguests[canonical]['mention_count'] += 1
        self.known_houseguests[canonical]['aliases'].add(name)
        
        if context:
            self.known_houseguests[canonical]['contexts'].append(context[:100])
        
        # Track frequency
        self.name_frequency[canonical] += 1
        
        # Learn associations
        other_names = self.extract_potential_names(context)
        for other in other_names:
            if other.lower() != canonical:
                self.name_associations[canonical].add(other.lower())
    
    def extract_potential_names(self, text: str) -> List[str]:
        """Extract potential houseguest names using multiple patterns"""
        potential_names = set()
        
        # Check for full names
        for match in self.name_patterns['full_name'].findall(text):
            full_name = f"{match[0]} {match[1]}"
            if match[0] not in self.exclude_words and match[1] not in self.exclude_words:
                potential_names.add(match[0])  # First name
                potential_names.add(full_name)  # Full name
        
        # Check other patterns
        for pattern_name, pattern in self.name_patterns.items():
            if pattern_name != 'full_name':
                for match in pattern.findall(text):
                    name = match.strip("'s") if pattern_name == 'possessive' else match
                    if name not in self.exclude_words and len(name) > 2:
                        potential_names.add(name)
        
        # Verify names have context indicators nearby
        verified_names = []
        text_lower = text.lower()
        
        for name in potential_names:
            name_pos = text_lower.find(name.lower())
            if name_pos != -1:
                # Check surrounding context (50 chars before and after)
                context_start = max(0, name_pos - 50)
                context_end = min(len(text), name_pos + len(name) + 50)
                context = text_lower[context_start:context_end]
                
                # Check if any context indicators are present
                if any(indicator in context for indicator in self.context_indicators):
                    verified_names.append(name)
                    self.learn_houseguest(name, text[context_start:context_end])
                elif self.is_likely_houseguest(name):
                    verified_names.append(name)
        
        return verified_names
    
    def is_likely_houseguest(self, name: str) -> bool:
        """Determine if a name is likely a houseguest based on learned patterns"""
        canonical = name.lower()
        
        # Already known
        if canonical in self.known_houseguests:
            return True
        
        # Check frequency threshold
        if self.name_frequency[canonical] >= 3:
            return True
        
        # Check similarity to known names
        for known in self.known_houseguests:
            similarity = difflib.SequenceMatcher(None, canonical, known).ratio()
            if similarity > 0.8:  # 80% similar
                self.add_alias(known, name)
                return True
        
        return False
    
    def add_alias(self, canonical_name: str, alias: str):
        """Add an alias for a known houseguest"""
        canonical = canonical_name.lower()
        if canonical in self.known_houseguests:
            self.known_houseguests[canonical]['aliases'].add(alias)
            self.nickname_map[alias.lower()] = canonical
    
    def resolve_name(self, name: str) -> str:
        """Resolve a name to its canonical form"""
        name_lower = name.lower()
        
        # Direct match
        if name_lower in self.known_houseguests:
            return self.known_houseguests[name_lower]['display_name']
        
        # Nickname match
        if name_lower in self.nickname_map:
            canonical = self.nickname_map[name_lower]
            return self.known_houseguests[canonical]['display_name']
        
        # Fuzzy match
        best_match = None
        best_ratio = 0
        
        for known in self.known_houseguests:
            ratio = difflib.SequenceMatcher(None, name_lower, known).ratio()
            if ratio > best_ratio and ratio > 0.8:
                best_ratio = ratio
                best_match = known
        
        if best_match:
            self.add_alias(best_match, name)
            return self.known_houseguests[best_match]['display_name']
        
        return name
    
    def get_houseguest_stats(self) -> Dict[str, any]:
        """Get statistics about tracked houseguests"""
        stats = {
            'total_houseguests': len(self.known_houseguests),
            'total_mentions': sum(hg['mention_count'] for hg in self.known_houseguests.values()),
            'most_mentioned': None,
            'associations': {}
        }
        
        if self.known_houseguests:
            most_mentioned = max(self.known_houseguests.items(), 
                               key=lambda x: x[1]['mention_count'])
            stats['most_mentioned'] = {
                'name': most_mentioned[1]['display_name'],
                'count': most_mentioned[1]['mention_count']
            }
        
        # Get top associations
        for name, associates in self.name_associations.items():
            if associates and name in self.known_houseguests:
                display_name = self.known_houseguests[name]['display_name']
                stats['associations'][display_name] = [
                    self.known_houseguests.get(a, {}).get('display_name', a.title()) 
                    for a in list(associates)[:3]
                ]
        
        return stats

class BBAnalyzer:
    """Enhanced analyzer with smart name recognition"""
    
    def __init__(self, houseguest_tracker: HouseguestTracker):
        self.houseguest_tracker = houseguest_tracker
        
        # Enhanced keyword categories with weights
        self.keyword_categories = {
            '🏆 Competition': {
                'keywords': ['hoh', 'head of household', 'power of veto', 'pov', 'veto',
                           'nomination', 'ceremony', 'competition', 'comp', 'challenge',
                           'immunity', 'safety', 'won', 'wins', 'winner', 'lost', 'loser'],
                'weight': 1.5
            },
            '🎯 Strategy': {
                'keywords': ['alliance', 'backdoor', 'target', 'targeting', 'scheme', 
                           'plan', 'strategy', 'vote', 'voting', 'flip', 'campaigning',
                           'deal', 'final', 'jury', 'evict', 'save', 'use veto'],
                'weight': 1.3
            },
            '💥 Drama': {
                'keywords': ['argument', 'fight', 'confrontation', 'drama', 'tension',
                           'called out', 'blowup', 'heated', 'angry', 'upset', 'crying',
                           'tears', 'yelling', 'screaming', 'clash', 'conflict'],
                'weight': 1.2
            },
            '💕 Romance': {
                'keywords': ['showmance', 'romance', 'flirting', 'cuddle', 'kiss',
                           'relationship', 'attracted', 'feelings', 'crush', 'date',
                           'love', 'couple', 'bed', 'sleeping together'],
                'weight': 1.0
            },
            '🎮 Twist': {
                'keywords': ['twist', 'power', 'secret', 'america', 'vote', 'special',
                           'announcement', 'surprise', 'revealed', 'hidden'],
                'weight': 1.4
            }
        }
        
        # Time-based importance modifiers
        self.time_modifiers = {
            'immediate': ['now', 'just', 'breaking', 'happening'],
            'recent': ['earlier', 'today', 'tonight'],
            'ongoing': ['continues', 'still', 'ongoing']
        }
    
    def categorize_update(self, update: BBUpdate) -> List[str]:
        """Categorize with weighted scoring"""
        content = f"{update.title} {update.description}".lower()
        category_scores = {}
        
        for category, data in self.keyword_categories.items():
            score = 0
            for keyword in data['keywords']:
                if keyword in content:
                    # Give more weight to keywords in title
                    if keyword in update.title.lower():
                        score += 2 * data['weight']
                    else:
                        score += data['weight']
            
            if score > 0:
                category_scores[category] = score
        
        # Sort by score and return top categories
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        categories = [cat for cat, score in sorted_categories[:3] if score > 0]
        
        return categories if categories else ["📝 General"]
    
    def extract_houseguests(self, update: BBUpdate) -> List[str]:
        """Extract and resolve houseguest names"""
        text = f"{update.title} {update.description}"
        potential_names = self.houseguest_tracker.extract_potential_names(text)
        
        # Resolve names to canonical forms
        resolved_names = []
        seen = set()
        
        for name in potential_names:
            resolved = self.houseguest_tracker.resolve_name(name)
            canonical = resolved.lower()
            if canonical not in seen:
                resolved_names.append(resolved)
                seen.add(canonical)
        
        return resolved_names
    
    def analyze_strategic_importance(self, update: BBUpdate) -> int:
        """Enhanced importance scoring"""
        content = f"{update.title} {update.description}".lower()
        score = 1
        
        # Category-based scoring
        categories = self.categorize_update(update)
        category_multiplier = 1.0
        
        for category in categories:
            if 'Competition' in category:
                category_multiplier = max(category_multiplier, 1.5)
            elif 'Strategy' in category:
                category_multiplier = max(category_multiplier, 1.3)
            elif 'Twist' in category:
                category_multiplier = max(category_multiplier, 1.4)
        
        # Content-based scoring
        critical_events = {
            'eviction': 4, 'evicted': 4, 'eliminated': 4,
            'nomination': 3, 'nominated': 3, 'renom': 3,
            'backdoor': 4, 'blindside': 4,
            'hoh': 3, 'head of household': 3,
            'veto': 3, 'power of veto': 3, 'pov': 3,
            'alliance': 2, 'final 2': 3, 'final two': 3,
            'vote': 2, 'flip': 3, 'house flip': 4,
            'twist': 3, 'power': 3, 'advantage': 3
        }
        
        for event, points in critical_events.items():
            if event in content:
                score += points
        
        # Time modifier
        for modifier_type, keywords in self.time_modifiers.items():
            if any(keyword in content for keyword in keywords):
                if modifier_type == 'immediate':
                    score += 2
                elif modifier_type == 'recent':
                    score += 1
        
        # Number of houseguests mentioned (more people = more important)
        houseguest_count = len(update.mentioned_houseguests)
        if houseguest_count >= 4:
            score += 2
        elif houseguest_count >= 2:
            score += 1
        
        # Apply category multiplier
        score = int(score * category_multiplier)
        
        return min(score, 10)

class BBDatabase:
    """Async database operations with connection pooling"""
    
    def __init__(self, db_path: str = "bb_updates.db"):
        self.db_path = db_path
        self._connection = None
        self._lock = asyncio.Lock()
    
    async def get_connection(self):
        """Get or create database connection"""
        async with self._lock:
            if self._connection is None:
                self._connection = await aiosqlite.connect(self.db_path)
                await self._connection.execute("PRAGMA journal_mode=WAL")
                await self._connection.execute("PRAGMA synchronous=NORMAL")
            return self._connection
    
    async def init_database(self):
        """Initialize the database schema"""
        conn = await self.get_connection()
        try:
            await conn.execute("""
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
                    mentioned_houseguests TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS houseguests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_name TEXT UNIQUE,
                    display_name TEXT,
                    aliases TEXT,
                    mention_count INTEGER DEFAULT 0,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    associations TEXT
                )
            """)
            
            # Create indices
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON updates(content_hash)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_pub_date ON updates(pub_date)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON updates(importance_score)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_houseguest_mentions ON houseguests(mention_count)")
            
            await conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        async with self._lock:
            if self._connection:
                await self._connection.close()
                self._connection = None
    
    async def is_duplicate(self, content_hash: str) -> bool:
        """Check if update already exists"""
        try:
            conn = await self.get_connection()
            async with conn.execute(
                "SELECT 1 FROM updates WHERE content_hash = ?", 
                (content_hash,)
            ) as cursor:
                result = await cursor.fetchone()
                return result is not None
            
        except Exception as e:
            logger.error(f"Database duplicate check error: {e}")
            return False
    
    async def store_update(self, update: BBUpdate):
        """Store a new update"""
        try:
            conn = await self.get_connection()
            
            categories_str = "|".join(update.categories)
            houseguests_str = "|".join(update.mentioned_houseguests)
            
            await conn.execute("""
                INSERT INTO updates (
                    content_hash, title, description, link, pub_date, 
                    author, importance_score, categories, mentioned_houseguests
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                update.content_hash, update.title, update.description, 
                update.link, update.pub_date, update.author, 
                update.importance_score, categories_str, houseguests_str
            ))
            
            await conn.commit()
            
        except Exception as e:
            logger.error(f"Database store error: {e}")
            raise
    
    async def save_houseguest_data(self, houseguest_tracker: HouseguestTracker):
        """Save houseguest tracking data"""
        try:
            conn = await self.get_connection()
            
            for canonical, data in houseguest_tracker.known_houseguests.items():
                aliases_str = "|".join(data['aliases'])
                associations = houseguest_tracker.name_associations.get(canonical, set())
                associations_str = "|".join(associations)
                
                await conn.execute("""
                    INSERT OR REPLACE INTO houseguests (
                        canonical_name, display_name, aliases, mention_count,
                        first_seen, last_seen, associations
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    canonical, data['display_name'], aliases_str,
                    data['mention_count'], data['first_seen'],
                    datetime.now(), associations_str
                ))
            
            await conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving houseguest data: {e}")
    
    async def load_houseguest_data(self) -> Dict[str, any]:
        """Load houseguest data from database"""
        try:
            conn = await self.get_connection()
            houseguests = {}
            
            async with conn.execute("""
                SELECT canonical_name, display_name, aliases, mention_count,
                       first_seen, associations
                FROM houseguests
            """) as cursor:
                async for row in cursor:
                    canonical = row[0]
                    houseguests[canonical] = {
                        'display_name': row[1],
                        'aliases': set(row[2].split('|')) if row[2] else set(),
                        'mention_count': row[3],
                        'first_seen': datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                        'contexts': [],
                        'associations': set(row[5].split('|')) if row[5] else set()
                    }
            
            return houseguests
            
        except Exception as e:
            logger.error(f"Error loading houseguest data: {e}")
            return {}
    
    async def get_recent_updates(self, hours: int = 24) -> List[BBUpdate]:
        """Get updates from the last N hours"""
        try:
            conn = await self.get_connection()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            async with conn.execute("""
                SELECT title, description, link, pub_date, content_hash, author,
                       categories, mentioned_houseguests, importance_score
                FROM updates 
                WHERE pub_date > ?
                ORDER BY pub_date DESC
            """, (cutoff_time,)) as cursor:
                
                updates = []
                async for row in cursor:
                    update = BBUpdate(
                        title=row[0],
                        description=row[1],
                        link=row[2],
                        pub_date=datetime.fromisoformat(row[3]) if isinstance(row[3], str) else row[3],
                        content_hash=row[4],
                        author=row[5] or "",
                        categories=row[6].split('|') if row[6] else [],
                        mentioned_houseguests=row[7].split('|') if row[7] else [],
                        importance_score=row[8]
                    )
                    updates.append(update)
                
                return updates
            
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []

class BBDiscordBot(commands.Bot):
    """Enhanced Discord bot with smart features"""
    
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
        self.houseguest_tracker = HouseguestTracker()
        self.analyzer = BBAnalyzer(self.houseguest_tracker)
        
        # Performance optimizations
        self.session: Optional[aiohttp.ClientSession] = None
        self.rss_cache = {'data': None, 'timestamp': 0}
        self.cache_ttl = self.config.get('cache_ttl', 3600)
        
        # State tracking
        self.is_shutting_down = False
        self.last_successful_check = datetime.now()
        self.total_updates_processed = 0
        self.consecutive_errors = 0
        self.update_queue = asyncio.Queue(maxsize=100)
        
        # Signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_shutting_down = True
        asyncio.create_task(self.cleanup())
    
    async def setup_hook(self):
        """Setup bot resources"""
        await self.db.init_database()
        
        # Load houseguest data
        saved_houseguests = await self.db.load_houseguest_data()
        self.houseguest_tracker.known_houseguests = saved_houseguests
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Start background tasks
        self.process_update_queue.start()
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Save houseguest data
            await self.db.save_houseguest_data(self.houseguest_tracker)
            
            # Close connections
            if self.session:
                await self.session.close()
            await self.db.close()
            
            await self.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def on_ready(self):
        """Bot startup event"""
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')
        
        try:
            self.check_rss_feed.start()
            if self.config.get('enable_heartbeat'):
                self.heartbeat.start()
            
            logger.info("Background tasks started successfully")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    def create_content_hash(self, title: str, description: str) -> str:
        """Create a unique hash for content deduplication"""
        # Normalize content
        content = f"{title}|{description}".lower()
        # Remove timestamps and dates for better deduplication
        content = re.sub(r'\d{1,2}:\d{2}\s*[ap]m', '', content)
        content = re.sub(r'\d{1,2}/\d{1,2}(/\d{2,4})?', '', content)
        content = re.sub(r'\s+', ' ', content).strip()
        return hashlib.md5(content.encode()).hexdigest()
    
    async def fetch_rss_feed(self) -> Optional[feedparser.FeedParserDict]:
        """Fetch RSS feed with caching and error handling"""
        try:
            # Check cache
            current_time = time.time()
            if (self.rss_cache['data'] and 
                current_time - self.rss_cache['timestamp'] < 60):  # 1 minute cache
                return self.rss_cache['data']
            
            # Fetch fresh data
            async with self.session.get(self.rss_url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    # Update cache
                    self.rss_cache['data'] = feed
                    self.rss_cache['timestamp'] = current_time
                    
                    return feed
                else:
                    logger.error(f"RSS feed returned status {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("RSS feed fetch timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching RSS feed: {e}")
            return None
    
    def process_rss_entries(self, entries) -> List[BBUpdate]:
        """Process RSS entries into BBUpdate objects"""
        updates = []
        
        for entry in entries[:20]:  # Limit processing to prevent overload
            try:
                title = entry.get('title', 'No title')
                description = entry.get('description', 'No description')
                link = entry.get('link', '')
                
                # Parse publication date
                pub_date = datetime.now()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])
                
                # Skip old updates
                age = datetime.now() - pub_date
                max_age = self.config.get('max_update_age_hours', 168)
                if age.total_seconds() > max_age * 3600:
                    continue
                
                content_hash = self.create_content_hash(title, description)
                author = entry.get('author', '')
                
                update = BBUpdate(
                    title=title,
                    description=description,
                    link=link,
                    pub_date=pub_date,
                    content_hash=content_hash,
                    author=author
                )
                
                updates.append(update)
                
            except Exception as e:
                logger.error(f"Error processing RSS entry: {e}")
                continue
        
        return updates
    
    async def process_update(self, update: BBUpdate):
        """Process a single update with analysis"""
        try:
            # Extract houseguests
            update.mentioned_houseguests = self.analyzer.extract_houseguests(update)
            
            # Categorize
            update.categories = self.analyzer.categorize_update(update)
            
            # Calculate importance
            update.importance_score = self.analyzer.analyze_strategic_importance(update)
            
            # Store in database
            await self.db.store_update(update)
            
            # Send to Discord
            await self.send_update_to_channel(update)
            
            self.total_updates_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing update: {e}")
            raise
    
    def create_update_embed(self, update: BBUpdate) -> discord.Embed:
        """Create a rich Discord embed for an update"""
        # Color based on importance
        colors = {
            range(1, 3): 0x95a5a6,  # Gray for low importance
            range(3, 5): 0x3498db,  # Blue for medium
            range(5, 7): 0x2ecc71,  # Green for medium-high
            range(7, 9): 0xf39c12,  # Orange for high
            range(9, 11): 0xe74c3c  # Red for critical
        }
        
        color = 0x95a5a6
        for range_obj, color_value in colors.items():
            if update.importance_score in range_obj:
                color = color_value
                break
        
        # Truncate if needed
        title = update.title[:256] if len(update.title) <= 256 else update.title[:253] + "..."
        description = update.description[:2048] if len(update.description) <= 2048 else update.description[:2045] + "..."
        
        embed = discord.Embed(
            title=title,
            description=description,
            color=color,
            url=update.link if update.link else None,
            timestamp=update.pub_date
        )
        
        # Categories field
        if update.categories:
            embed.add_field(
                name="Categories", 
                value=" | ".join(update.categories), 
                inline=True
            )
        
        # Importance field
        importance_bar = "█" * update.importance_score + "░" * (10 - update.importance_score)
        embed.add_field(
            name="Strategic Importance", 
            value=f"{importance_bar} {update.importance_score}/10", 
            inline=True
        )
        
        # Houseguests field
        if update.mentioned_houseguests:
            houseguests_text = ", ".join(update.mentioned_houseguests[:8])
            if len(update.mentioned_houseguests) > 8:
                houseguests_text += f" +{len(update.mentioned_houseguests) - 8} more"
            embed.add_field(
                name="Houseguests Mentioned", 
                value=houseguests_text, 
                inline=False
            )
        
        # Footer
        footer_parts = []
        if update.author:
            footer_parts.append(f"By: {update.author}")
        footer_parts.append(f"Update #{self.total_updates_processed}")
        embed.set_footer(text=" | ".join(footer_parts))
        
        return embed
    
    async def send_update_to_channel(self, update: BBUpdate):
        """Send update to configured channel with rate limiting"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            logger.warning("Update channel not configured")
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                channel = await self.fetch_channel(channel_id)
            
            if not channel:
                logger.error(f"Channel {channel_id} not found")
                return
            
            embed = self.create_update_embed(update)
            
            # Add buttons for high importance updates
            if update.importance_score >= 7:
                view = UpdateActionView(update)
                await channel.send(embed=embed, view=view)
            else:
                await channel.send(embed=embed)
            
        except discord.errors.Forbidden:
            logger.error(f"No permission to send to channel {channel_id}")
        except Exception as e:
            logger.error(f"Error sending update to channel: {e}")
    
    @tasks.loop(seconds=1)
    async def process_update_queue(self):
        """Process queued updates with rate limiting"""
        if self.is_shutting_down:
            return
        
        try:
            # Process up to 5 updates per second
            for _ in range(5):
                if self.update_queue.empty():
                    break
                    
                update = await self.update_queue.get()
                await self.process_update(update)
                
        except Exception as e:
            logger.error(f"Error processing update queue: {e}")
    
    @tasks.loop(minutes=2)
    async def check_rss_feed(self):
        """Check RSS feed for new updates"""
        if self.is_shutting_down:
            return
        
        try:
            feed = await self.fetch_rss_feed()
            
            if not feed or not feed.entries:
                logger.warning("No entries returned from RSS feed")
                self.consecutive_errors += 1
                return
            
            updates = self.process_rss_entries(feed.entries)
            new_updates = []
            
            # Check for duplicates
            for update in updates:
                if not await self.db.is_duplicate(update.content_hash):
                    new_updates.append(update)
            
            # Queue new updates
            for update in new_updates:
                try:
                    await self.update_queue.put(update)
                except asyncio.QueueFull:
                    logger.warning("Update queue full, skipping update")
            
            self.last_successful_check = datetime.now()
            self.consecutive_errors = 0
            
            if new_updates:
                logger.info(f"Found {len(new_updates)} new updates")
            
            # Save houseguest data periodically
            if self.total_updates_processed % 50 == 0:
                await self.db.save_houseguest_data(self.houseguest_tracker)
                
        except Exception as e:
            logger.error(f"Error in RSS check: {e}")
            self.consecutive_errors += 1
            
            # Auto-restart if too many errors
            if (self.config.get('enable_auto_restart') and 
                self.consecutive_errors >= self.config.get('max_consecutive_errors', 10)):
                logger.critical("Too many consecutive errors, restarting...")
                await self.cleanup()
                os.execv(sys.executable, [sys.executable] + sys.argv)
    
    @tasks.loop(minutes=5)
    async def heartbeat(self):
        """Send heartbeat to monitoring service"""
        if self.is_shutting_down:
            return
        
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'updates_processed': self.total_updates_processed,
                'consecutive_errors': self.consecutive_errors,
                'houseguests_tracked': len(self.houseguest_tracker.known_houseguests),
                'queue_size': self.update_queue.qsize()
            }
            
            logger.info(f"Heartbeat: {stats}")
            
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
    
    # Commands
    @commands.command(name='summary')
    async def daily_summary(self, ctx, hours: int = 24):
        """Generate a comprehensive summary of updates"""
        try:
            if hours < 1 or hours > 168:
                await ctx.send("Hours must be between 1 and 168")
                return
            
            await ctx.typing()
            
            updates = await self.db.get_recent_updates(hours)
            
            if not updates:
                await ctx.send(f"No updates found in the last {hours} hours.")
                return
            
            # Group by categories
            category_groups = defaultdict(list)
            for update in updates:
                for category in update.categories:
                    category_groups[category].append(update)
            
            # Create paginated embeds
            embeds = []
            
            # Overview embed
            overview_embed = discord.Embed(
                title=f"Big Brother Updates Summary ({hours}h)",
                description=f"**{len(updates)} total updates**",
                color=0x3498db,
                timestamp=datetime.now()
            )
            
            # Add houseguest stats
            hg_stats = self.houseguest_tracker.get_houseguest_stats()
            if hg_stats['most_mentioned']:
                overview_embed.add_field(
                    name="Most Mentioned Houseguest",
                    value=f"{hg_stats['most_mentioned']['name']} ({hg_stats['most_mentioned']['count']} times)",
                    inline=True
                )
            
            overview_embed.add_field(
                name="Houseguests Tracked",
                value=str(hg_stats['total_houseguests']),
                inline=True
            )
            
            embeds.append(overview_embed)
            
            # Category summaries
            for category, cat_updates in sorted(category_groups.items(), 
                                              key=lambda x: len(x[1]), reverse=True):
                if len(embeds) >= 10:  # Discord embed limit
                    break
                
                cat_embed = discord.Embed(
                    title=f"{category} Updates",
                    description=f"{len(cat_updates)} updates in this category",
                    color=0x2ecc71
                )
                
                # Top 5 most important updates
                top_updates = sorted(cat_updates, 
                                   key=lambda x: x.importance_score, 
                                   reverse=True)[:5]
                
                for i, update in enumerate(top_updates, 1):
                    time_str = update.pub_date.strftime("%m/%d %I:%M%p")
                    value = f"[{time_str}] {update.title[:100]}"
                    if len(update.title) > 100:
                        value += "..."
                    
                    # Add houseguests if any
                    if update.mentioned_houseguests:
                        value += f"\n*{', '.join(update.mentioned_houseguests[:3])}*"
                    
                    cat_embed.add_field(
                        name=f"#{i} (Importance: {update.importance_score}/10)",
                        value=value,
                        inline=False
                    )
                
                embeds.append(cat_embed)
            
            # Send embeds
            for embed in embeds:
                await ctx.send(embed=embed)
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            await ctx.send("Error generating summary. Please try again.")
    
    @commands.command(name='houseguests', aliases=['hgs'])
    async def houseguest_list(self, ctx):
        """Show tracked houseguests and their stats"""
        try:
            if not self.houseguest_tracker.known_houseguests:
                await ctx.send("No houseguests tracked yet.")
                return
            
            embed = discord.Embed(
                title="Tracked Houseguests",
                color=0x9b59b6,
                timestamp=datetime.now()
            )
            
            # Sort by mention count
            sorted_hgs = sorted(
                self.houseguest_tracker.known_houseguests.items(),
                key=lambda x: x[1]['mention_count'],
                reverse=True
            )
            
            for canonical, data in sorted_hgs[:25]:  # Limit to 25 fields
                aliases = [a for a in data['aliases'] if a != data['display_name']]
                value_parts = [f"Mentions: {data['mention_count']}"]
                
                if aliases:
                    value_parts.append(f"Also: {', '.join(aliases[:3])}")
                
                # Add associations
                associations = self.houseguest_tracker.name_associations.get(canonical, set())
                if associations:
                    assoc_names = []
                    for assoc in list(associations)[:3]:
                        if assoc in self.houseguest_tracker.known_houseguests:
                            assoc_names.append(
                                self.houseguest_tracker.known_houseguests[assoc]['display_name']
                            )
                    if assoc_names:
                        value_parts.append(f"Often with: {', '.join(assoc_names)}")
                
                embed.add_field(
                    name=data['display_name'],
                    value="\n".join(value_parts),
                    inline=True
                )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error showing houseguests: {e}")
            await ctx.send("Error displaying houseguest list.")
    
    @commands.command(name='addname')
    @commands.has_permissions(manage_messages=True)
    async def add_houseguest_alias(self, ctx, houseguest: str, *, aliases: str):
        """Add aliases for a houseguest
        Usage: !bbaddname Matt matthew matty"""
        try:
            aliases_list = aliases.split()
            
            # Find or create houseguest
            canonical = houseguest.lower()
            if canonical not in self.houseguest_tracker.known_houseguests:
                self.houseguest_tracker.learn_houseguest(houseguest)
            
            # Add aliases
            for alias in aliases_list:
                self.houseguest_tracker.add_alias(canonical, alias)
            
            # Save to database
            await self.db.save_houseguest_data(self.houseguest_tracker)
            
            embed = discord.Embed(
                title="Aliases Added",
                description=f"Added {len(aliases_list)} aliases for {houseguest}",
                color=0x2ecc71
            )
            embed.add_field(name="Aliases", value=", ".join(aliases_list))
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error adding aliases: {e}")
            await ctx.send("Error adding aliases.")
    
    @commands.command(name='search')
    async def search_updates(self, ctx, *, query: str):
        """Search for updates containing specific text"""
        try:
            await ctx.typing()
            
            # Search in recent updates (last 7 days)
            updates = await self.db.get_recent_updates(168)
            
            # Filter updates
            query_lower = query.lower()
            matching_updates = []
            
            for update in updates:
                if (query_lower in update.title.lower() or 
                    query_lower in update.description.lower() or
                    any(query_lower in hg.lower() for hg in update.mentioned_houseguests)):
                    matching_updates.append(update)
            
            if not matching_updates:
                await ctx.send(f"No updates found containing '{query}'")
                return
            
            # Create embed
            embed = discord.Embed(
                title=f"Search Results for '{query}'",
                description=f"Found {len(matching_updates)} matching updates",
                color=0x3498db,
                timestamp=datetime.now()
            )
            
            # Show top 10 results
            for update in matching_updates[:10]:
                time_str = update.pub_date.strftime("%m/%d %I:%M%p")
                value = f"[{time_str}] {update.description[:150]}"
                if len(update.description) > 150:
                    value += "..."
                
                embed.add_field(
                    name=update.title[:100],
                    value=value,
                    inline=False
                )
            
            if len(matching_updates) > 10:
                embed.set_footer(text=f"Showing 10 of {len(matching_updates)} results")
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error searching updates: {e}")
            await ctx.send("Error searching updates.")
    
    @commands.command(name='setchannel')
    @commands.has_permissions(administrator=True)
    async def set_update_channel(self, ctx, channel: discord.TextChannel):
        """Set the channel for RSS updates"""
        try:
            if not channel.permissions_for(ctx.guild.me).send_messages:
                await ctx.send(f"I don't have permission to send messages in {channel.mention}")
                return
            
            self.config.set('update_channel_id', channel.id)
            
            embed = discord.Embed(
                title="Update Channel Set",
                description=f"Updates will now be sent to {channel.mention}",
                color=0x2ecc71
            )
            
            await ctx.send(embed=embed)
            logger.info(f"Update channel set to {channel.id}")
            
        except Exception as e:
            logger.error(f"Error setting channel: {e}")
            await ctx.send("Error setting channel.")
    
    @commands.command(name='status')
    async def bot_status(self, ctx):
        """Show detailed bot status"""
        try:
            # Calculate uptime
            uptime = datetime.now() - self.last_successful_check
            
            # Get system stats
            import psutil
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            embed = discord.Embed(
                title="Big Brother Bot Status",
                color=0x2ecc71 if self.consecutive_errors == 0 else 0xe74c3c,
                timestamp=datetime.now()
            )
            
            # Bot info
            embed.add_field(
                name="Bot Info",
                value=f"Version: 2.0\nGuilds: {len(self.guilds)}\nUptime: {uptime}",
                inline=True
            )
            
            # RSS info
            embed.add_field(
                name="RSS Feed",
                value=f"URL: [Jokers Updates]({self.rss_url})\nLast Check: {self.last_successful_check.strftime('%I:%M%p')}",
                inline=True
            )
            
            # Channel info
            channel_id = self.config.get('update_channel_id')
            channel_mention = f"<#{channel_id}>" if channel_id else "Not set"
            embed.add_field(
                name="Update Channel",
                value=channel_mention,
                inline=True
            )
            
            # Statistics
            embed.add_field(
                name="Statistics",
                value=f"Updates Processed: {self.total_updates_processed}\n"
                      f"Houseguests Tracked: {len(self.houseguest_tracker.known_houseguests)}\n"
                      f"Queue Size: {self.update_queue.qsize()}",
                inline=True
            )
            
            # System info
            embed.add_field(
                name="System",
                value=f"Memory: {memory_usage:.1f} MB\n"
                      f"Errors: {self.consecutive_errors}",
                inline=True
            )
            
            # Health status
            if self.consecutive_errors == 0:
                health = "✅ Healthy"
            elif self.consecutive_errors < 5:
                health = "⚠️ Warning"
            else:
                health = "❌ Critical"
            
            embed.add_field(
                name="Health",
                value=health,
                inline=True
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating status: {e}")
            await ctx.send("Error generating status.")

class UpdateActionView(discord.ui.View):
    """Interactive buttons for important updates"""
    
    def __init__(self, update: BBUpdate):
        super().__init__(timeout=3600)  # 1 hour timeout
        self.update = update
    
    @discord.ui.button(label="Mark as Read", style=discord.ButtonStyle.primary, emoji="✅")
    async def mark_read(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("Update marked as read!", ephemeral=True)
        self.stop()
    
    @discord.ui.button(label="Get Details", style=discord.ButtonStyle.secondary, emoji="📋")
    async def get_details(self, interaction: discord.Interaction, button: discord.ui.Button):
        detail_embed = discord.Embed(
            title="Update Details",
            color=0x3498db
        )
        
        detail_embed.add_field(
            name="Full Description",
            value=self.update.description[:1024],
            inline=False
        )
        
        if self.update.mentioned_houseguests:
            detail_embed.add_field(
                name="All Houseguests Mentioned",
                value=", ".join(self.update.mentioned_houseguests),
                inline=False
            )
        
        detail_embed.add_field(
            name="Link",
            value=self.update.link or "No link available",
            inline=False
        )
        
        await interaction.response.send_message(embed=detail_embed, ephemeral=True)

def main():
    """Main function to run the bot"""
    try:
        bot = BBDiscordBot()
        
        @bot.event
        async def on_command_error(ctx, error):
            if isinstance(error, commands.MissingPermissions):
                await ctx.send("You don't have permission to use this command.")
            elif isinstance(error, commands.CommandNotFound):
                # Ignore command not found
                pass
            elif isinstance(error, commands.MissingRequiredArgument):
                await ctx.send(f"Missing required argument: {error.param.name}")
            else:
                logger.error(f"Command error: {error}")
                await ctx.send("An error occurred while processing the command.")
        
        @bot.command(name='help')
        async def help_command(ctx):
            """Show help for all commands"""
            embed = discord.Embed(
                title="Big Brother Bot Commands",
                description="Monitor Big Brother updates with smart analysis",
                color=0x3498db
            )
            
            commands_list = [
                ("!bbsummary [hours]", "Get a summary of recent updates (default: 24h)"),
                ("!bbhouseguests", "Show all tracked houseguests and stats"),
                ("!bbsearch <query>", "Search for updates containing specific text"),
                ("!bbaddname <name> <aliases>", "Add aliases for a houseguest (Mod only)"),
                ("!bbstatus", "Show bot status and statistics"),
                ("!bbsetchannel #channel", "Set update channel (Admin only)"),
                ("!bbhelp", "Show this help message")
            ]
            
            for name, description in commands_list:
                embed.add_field(name=name, value=description, inline=False)
            
            embed.set_footer(text="Big Brother Bot v2.0 | Smart houseguest tracking enabled")
            
            await ctx.send(embed=embed)
        
        bot_token = bot.config.get('bot_token')
        if not bot_token:
            logger.error("No bot token found! Set BOT_TOKEN environment variable")
            return
        
        logger.info("Starting Big Brother Discord Bot v2.0...")
        bot.run(bot_token, reconnect=True)
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    main()
