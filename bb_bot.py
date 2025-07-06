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
import pytz
from typing import List, Dict, Set, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import time
import traceback
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict, Counter, deque, OrderedDict
import anthropic
from enum import Enum

# Configure SQLite to handle datetime properly
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter("timestamp", lambda b: datetime.fromisoformat(b.decode()))
sqlite3.register_converter("TIMESTAMP", lambda b: datetime.fromisoformat(b.decode()))

# Configuration constants
DEFAULT_CONFIG = {
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
    "anthropic_api_key": "",
    "enable_llm_summaries": True,
    "llm_model": "claude-3-haiku-20240307",
    "batch_mode": "intelligent",
    "min_batch_size": 3,
    "max_batch_wait_minutes": 30,
    "urgent_batch_threshold": 2,
    "timeline_mode": "smart",
    "max_timeline_embeds": 3,
    "show_importance_timeline": True,
    # Rate limiting settings
    "llm_requests_per_minute": 10,
    "llm_requests_per_hour": 100,
    "max_processed_hashes": 10000
}

# Environment variable mappings
ENV_MAPPINGS = {
    'BOT_TOKEN': 'bot_token',
    'UPDATE_CHANNEL_ID': 'update_channel_id',
    'ANTHROPIC_API_KEY': 'anthropic_api_key',
    'RSS_CHECK_INTERVAL': 'rss_check_interval',
    'LLM_MODEL': 'llm_model',
    'BATCH_MODE': 'batch_mode',
    'OWNER_ID': 'owner_id'
}

# Prediction categories with point values
PREDICTION_CATEGORIES = {
    'season_winner': {'points': 100, 'name': 'Season Winner'},
    'weekly_hoh': {'points': 10, 'name': 'Weekly HOH'},
    'weekly_veto': {'points': 8, 'name': 'Weekly Veto'},
    'weekly_eviction': {'points': 15, 'name': 'Weekly Eviction'},
    'jury_vote': {'points': 5, 'name': 'Jury Vote'},
    'special_event': {'points': 20, 'name': 'Special Event'}
}

# Zing constants
PG_ZINGS = [
    # 60 PG zings (30%)
    "{target}, you're so forgettable, production had to label your microphone twice!",
    "{target}, your diary room sessions are so boring, even the cameras fall asleep!",
    "{target}, you float so much, you should come with a pool noodle!",
    "{target}, your game moves are like a sloth doing yoga - slow and pointless!",
    "{target}, you're playing Big Brother like it's Big Sleeper!",
    "{target}, your strategy is so invisible, it should come with a magnifying glass!",
    "{target}, you're about as threatening as a rubber duck in the HOH competition!",
    "{target}, your alliance building skills are like a house of cards in a tornado!",
    "{target}, you blend into the background so well, you should be wallpaper!",
    "{target}, your competition record is like a phone with no signal - zero bars!",
    "{target}, you're playing chess while everyone else is playing... well, chess. You're just bad at it!",
    "{target}, your social game is like Wi-Fi in the basement - barely connected!",
    "{target}, you're so out of the loop, you think POV stands for 'Point of Vanilla'!",
    "{target}, your jury management is like a GPS with no satellites - completely lost!",
    "{target}, you're coasting so hard, you should be wearing roller skates!",
    "{target}, your HOH reign was shorter than a commercial break!",
    "{target}, you're playing Big Brother like it's Little Sister!",
    "{target}, your competition wins are rarer than a unicorn sighting!",
    "{target}, you're so predictable, we could replace you with a magic 8-ball!",
    "{target}, your strategy talks are like bedtime stories - they put everyone to sleep!",
    "{target}, you're about as intimidating as a kitten in mittens!",
    "{target}, your game resume is shorter than a haiku!",
    "{target}, you're playing hide and seek while everyone else plays Big Brother!",
    "{target}, your big moves are smaller than a Tic Tac!",
    "{target}, you're so far behind in the game, you're still in sequester!",
    "{target}, your diary room confessions are like reading a blank book!",
    "{target}, you're floating so much, NASA wants to study you!",
    "{target}, your competition performance is like dial-up internet - outdated and slow!",
    "{target}, you're playing Big Brother like it's a library - too quiet!",
    "{target}, your strategic mind is like a maze with no exit!",
    "{target}, you're so passive, you make a sloth look hyperactive!",
    "{target}, your game awareness is like a bat without sonar!",
    "{target}, you're riding coattails so hard, you should be a fashion designer!",
    "{target}, your threat level is lower than the basement!",
    "{target}, you're playing 4D chess... badly... in a checkers game!",
    "{target}, your social connections are like a phone on airplane mode!",
    "{target}, you're so forgettable, your own alliance forgets you exist!",
    "{target}, your competition skills need more training than a newborn giraffe!",
    "{target}, you're playing Big Brother like it's Big Naptime!",
    "{target}, your strategic planning is like a GPS going in circles!",
    "{target}, you're so under the radar, submarines are jealous!",
    "{target}, your game moves are like a snail racing a cheetah!",
    "{target}, you're playing so safe, you should wear a helmet in the diary room!",
    "{target}, your big brother game is more like little cousin energy!",
    "{target}, you're coasting harder than a bike going downhill!",
    "{target}, your competition record is like a broken scoreboard - all zeros!",
    "{target}, you're so out of touch, you think the veto meeting is a coffee date!",
    "{target}, your strategy is like a book with all blank pages!",
    "{target}, you're playing Big Brother like it's Big Background Character!",
    "{target}, your game influence is like a whisper in a hurricane!",
    "{target}, you're so forgettable, the memory wall forgot to light up your picture!",
    "{target}, your HOH letters are probably just participation certificates!",
    "{target}, you're playing so quietly, closed captions can't even pick you up!",
    "{target}, your competition training must have been watching paint dry!",
    "{target}, you're floating so professionally, you should charge admission!",
    "{target}, your strategic conversations are like silent movies without subtitles!",
    "{target}, you're so far from winning, you need a telescope to see the prize!",
    "{target}, your game awareness is like a security camera that's unplugged!",
    "{target}, you're playing Big Brother like it's Big Spectator!",
    "{target}, your influence in the house is like a fan with no blades!"
]

PG13_ZINGS = [
    # 140 PG-13 zings (70%)
    "{target}, you're so boring in bed, your showmance partner counts sheep... while awake!",
    "{target}, your kissing technique is like CPR - technically correct but nobody's enjoying it!",
    "{target}, you're so thirsty, the Have-Not cold showers are jealous!",
    "{target}, your flirting is so bad, even the cameras switch to fish!",
    "{target}, you shower so rarely, Febreze is considering you for a sponsorship!",
    "{target}, your game is like your love life - all talk, no action!",
    "{target}, you're so desperate for attention, you'd showmance with a mannequin!",
    "{target}, your HOH room action is like a nature documentary - rare and disappointing!",
    "{target}, you're trying so hard to be America's Favorite, you'd kiss a cactus for votes!",
    "{target}, your strategy is like your hygiene - questionable at best!",
    "{target}, you're so clingy, your showmance needs a restraining order!",
    "{target}, your diary room sessions are like your dating profile - full of lies!",
    "{target}, you're so bad at comps, you'd lose a staring contest to a blind person!",
    "{target}, your social game is like your personal hygiene - it stinks!",
    "{target}, you're showmancing so hard, production needs a cold shower!",
    "{target}, your game is weaker than your pull-out game... from competitions!",
    "{target}, you're so fake, your tears come with a warranty!",
    "{target}, your loyalty flip-flops more than your shower shoes!",
    "{target}, you're riding [name] harder than a mechanical bull at a bar!",
    "{target}, your competition performance is like your dating history - brief and embarrassing!",
    "{target}, you're so delusional, you think your showmance actually likes you!",
    "{target}, your strategic mind is like your love life - completely fictional!",
    "{target}, you're trying to play puppet master but you can't even master basic hygiene!",
    "{target}, your game is so messy, it needs a hazmat team!",
    "{target}, you're so desperate for camera time, you'd streak through the backyard!",
    "{target}, your HOH reign was shorter than your last relationship!",
    "{target}, you kiss so much ass, you should carry mouthwash!",
    "{target}, your game is like a bad hookup - regrettable and forgettable!",
    "{target}, you're so paranoid, you check under the bed for hidden vetoes!",
    "{target}, your showmance strategy is like your personality - shallow and transparent!",
    "{target}, you're playing everyone like a fiddle... a broken fiddle... that nobody wants to hear!",
    "{target}, your DR sessions are faker than your showmance feelings!",
    "{target}, you're so obsessed with camera time, you'd make out with your own reflection!",
    "{target}, your competition beast mode is more like competition deceased mode!",
    "{target}, you're riding coattails so hard, you should be charged rent!",
    "{target}, your flirting technique makes everyone want to self-evict!",
    "{target}, you're so two-faced, you need two microphones!",
    "{target}, your game is like your shower schedule - inconsistent and concerning!",
    "{target}, you're playing so dirty, the backyard needs to be power-washed!",
    "{target}, your strategic talks are like foreplay - awkward and nobody finishes satisfied!",
    "{target}, you're so thirsty for attention, the pool is jealous!",
    "{target}, your social game is like a bad date - everyone's looking for an exit!",
    "{target}, you've been in more beds than a hotel maid!",
    "{target}, your competition record is like your dating record - a lot of participation, no victories!",
    "{target}, you're so desperate for allies, you'd align with the ants!",
    "{target}, your game moves are like your dance moves - embarrassing and ineffective!",
    "{target}, you're playing so sloppy, you need a bib!",
    "{target}, your showmance is so forced, it needs a safe word!",
    "{target}, you're backstabbing so much, you should be a chiropractor!",
    "{target}, your HOH room saw less action than the storage room!",
    "{target}, you're so fake, your diary room needs a fact-checker!",
    "{target}, your strategy is like your skincare routine - non-existent!",
    "{target}, you're trying to be a mastermind but you can't even master basic math!",
    "{target}, your game is so weak, it needs Viagra!",
    "{target}, you're so desperate for screen time, you'd shower with the door open!",
    "{target}, your loyalty changes faster than your underwear... which isn't saying much!",
    "{target}, you're playing Big Brother like it's The Bachelor - wrong show!",
    "{target}, your competition performance is like your pickup lines - a total flop!",
    "{target}, you're so clingy, your alliance needs therapy!",
    "{target}, your game is dirtier than the kitchen after taco night!",
    "{target}, you're trying to be a villain but you're more like a Disney Channel antagonist!",
    "{target}, your strategic mind is like your dating standards - set way too low!",
    "{target}, you're so thirsty, you make the Have-Nots look hydrated!",
    "{target}, your DR sessions are more scripted than a soap opera!",
    "{target}, you're playing everyone but the only thing you're playing is yourself!",
    "{target}, your game is like a bad Tinder date - all swipe, no substance!",
    "{target}, you're so desperate for attention, you'd cuddle with the cameras!",
    "{target}, your competition skills are like your flirting skills - non-existent!",
    "{target}, you're riding [name] so hard, you should pay for gas!",
    "{target}, your game is messier than the bathroom after slop week!",
    "{target}, you're so fake, your emotions come with a director's cut!",
    "{target}, your strategic planning is like your shower planning - it doesn't happen!",
    "{target}, you're playing so many sides, you're basically a geometry lesson!",
    "{target}, your showmance is so awkward, it makes the feeds switch to fish!",
    "{target}, you're so bad at comps, you'd lose strip poker fully clothed!",
    "{target}, your social game is like a bad rash - irritating and spreading!",
    "{target}, you're trying to be memorable but you're as forgettable as last night's slop!",
    "{target}, your game is like your hygiene habits - everyone's talking about how bad it is!",
    "{target}, you're so desperate for a showmance, you'd date a Have-Not restriction!",
    "{target}, your competition record is like your love life - a series of unfortunate events!",
    "{target}, you're playing puppet master but your strings are more tangled than earbuds!",
    "{target}, your strategy sessions are like bad foreplay - confusing and unsatisfying!",
    "{target}, you're so thirsty, you're dehydrating the other houseguests!",
    "{target}, your game moves are like your bowel movements - irregular and concerning!",
    "{target}, you're trying to be Brad Pitt but you're more like arm pit!",
    "{target}, your alliance loyalty is like your deodorant - it doesn't last long!",
    "{target}, you're so fake, your tears need a stunt double!",
    "{target}, your HOH reign was like bad sex - over before anyone noticed it started!",
    "{target}, you're playing so many people, you need a spreadsheet!",
    "{target}, your social game is like your shower game - avoided by everyone!",
    "{target}, you're so desperate for votes, you'd promise your firstborn!",
    "{target}, your competition performance is like your personality - a total letdown!",
    "{target}, you're clinging to [name] like a bad STD!",
    "{target}, your game is dirtier than the hot tub after a showmance session!",
    "{target}, you're so delusional, you think production likes you!",
    "{target}, your strategic mind is like your sex appeal - imaginary!",
    "{target}, you're backstabbing so much, you need a spinal surgeon!",
    "{target}, your DR sessions are faker than your orgasms... I mean, emotions!",
    "{target}, you're so bad at this game, you make first boots look like masterminds!",
    "{target}, your showmance chemistry is like oil and water - it doesn't mix!",
    "{target}, you're playing so sloppy, you need a mop and bucket!",
    "{target}, your game is like a bad hookup - everyone regrets it happened!",
    "{target}, you're so thirsty for attention, you'd streak through the diary room!",
    "{target}, your competition skills are like your kissing skills - all teeth!",
    "{target}, you're riding coattails harder than a rodeo champion!",
    "{target}, your strategy is like your underwear - it needs to be changed!",
    "{target}, you're so fake, your showmance thinks you're CGI!",
    "{target}, your game moves are like your sexual moves - predictable and disappointing!",
    "{target}, you're playing everyone but yourself - and badly!",
    "{target}, your social game is like body odor - offensive and hard to ignore!",
    "{target}, you're so desperate for allies, you'd align with production!",
    "{target}, your HOH room saw less action than a nun's bedroom!",
    "{target}, you're trying to be a mastermind but you can't even mind your own business!",
    "{target}, your game is weaker than your bladder during a lockdown!",
    "{target}, you're so clingy, your showmance filed for emotional support!",
    "{target}, your competition record is like your hygiene record - needs improvement!",
    "{target}, you're playing dirty but not in the fun way!",
    "{target}, your strategic planning is like your family planning - accidental at best!",
    "{target}, you're so thirsty, the pool lost water!",
    "{target}, your game is messier than the sheets after your HOH reign!",
    "{target}, you're so fake, your diary room needs subtitles for the truth!",
    "{target}, your loyalty flip-flops more than a fish out of water!",
    "{target}, you're riding [name] like it's your job - and you're working overtime!",
    "{target}, your social game is like a communicable disease - everyone's trying to avoid it!",
    "{target}, you're so bad at comps, you'd lose a sleeping contest to an insomniac!",
    "{target}, your game moves are like your romantic moves - desperate and cringeworthy!",
    "{target}, you're playing so many sides, you're basically a rubik's cube!",
    "{target}, your showmance is so forced, it needs a lubricant!",
    "{target}, you're so desperate for screen time, you'd do diary rooms naked!",
    "{target}, your strategy is like your sex life - all in your head!",
    "{target}, you're backstabbing so much, you should open a sushi restaurant!",
    "{target}, your game is dirtier than the thoughts you have in the HOH shower!",
    "{target}, you're so delusional, you think your edit is accurate!",
    "{target}, your competition performance is like erectile dysfunction - a failure to rise to the occasion!",
    "{target}, you're clinging to power like you cling to your showmance - desperately!",
    "{target}, your DR sessions have more fiction than the Bible!",
    "{target}, you're so thirsty, you're sponsored by Gatorade!",
    "{target}, your game is like a wet dream - seemed good at the time but embarrassing in daylight!",
    "{target}, you're playing everyone like a cheap kazoo - badly and annoying everyone!"
]

# Combine all zings
ALL_ZINGS = PG_ZINGS + PG13_ZINGS

# Game-specific constants
COMPETITION_KEYWORDS = [
    'hoh', 'head of household', 'power of veto', 'pov', 'nomination', 
    'eviction', 'ceremony', 'competition', 'challenge', 'immunity'
]

STRATEGY_KEYWORDS = [
    'alliance', 'backdoor', 'target', 'scheme', 'plan', 'strategy',
    'vote', 'voting', 'campaigning', 'deal', 'promise', 'betrayal'
]

DRAMA_KEYWORDS = [
    'argument', 'fight', 'confrontation', 'drama', 'tension',
    'called out', 'blowup', 'heated', 'angry', 'upset'
]

RELATIONSHIP_KEYWORDS = [
    'showmance', 'romance', 'flirting', 'cuddle', 'kiss',
    'relationship', 'attracted', 'feelings', 'friendship', 'bond'
]

ENTERTAINMENT_KEYWORDS = [
    'funny', 'joke', 'laugh', 'prank', 'hilarious', 'comedy',
    'entertaining', 'memorable', 'quirky', 'silly'
]

URGENT_KEYWORDS = [
    'evicted', 'eliminated', 'wins hoh', 'wins pov', 'backdoor', 
    'self-evict', 'expelled', 'quit', 'medical', 'pandora', 'coup',
    'diamond veto', 'secret power', 'battle back', 'return', 'winner',
    'final', 'jury vote', 'finale night'
]

EXCLUDE_WORDS = {
    'The', 'This', 'That', 'They', 'Some', 'Many', 'Other', 'First', 'Last',
    'Big', 'Brother', 'Julie', 'Host', 'Diary', 'Room', 'Have', 'Not',
    'America', 'Favorite', 'Winner', 'Finale', 'Vote', 'Jury', 'Eviction',
    'Competition', 'Ceremony', 'Veto', 'Power', 'Head', 'Household'
}

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

# Alliance tracking enums
class AllianceStatus(Enum):
    ACTIVE = "active"
    BROKEN = "broken"
    SUSPECTED = "suspected"
    DISSOLVED = "dissolved"

class AllianceEventType(Enum):
    FORMED = "formed"
    EXPANDED = "expanded"
    BETRAYAL = "betrayal"
    DISSOLVED = "dissolved"
    SUSPECTED = "suspected"

class LRUCache:
    """Thread-safe LRU Cache implementation for tracking processed hashes"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self._lock = asyncio.Lock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    async def contains(self, key: str) -> bool:
        """Check if key exists in cache (async for thread safety)"""
        async with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats['hits'] += 1
                return True
            self.stats['misses'] += 1
            return False
    
    async def add(self, key: str, value: Optional[float] = None) -> None:
        """Add key to cache with optional timestamp value"""
        async with self._lock:
            if key in self.cache:
                # Update existing and move to end
                self.cache.move_to_end(key)
                self.cache[key] = value or time.time()
            else:
                # Add new entry
                self.cache[key] = value or time.time()
                
                # Check capacity and evict if necessary
                if len(self.cache) > self.capacity:
                    evicted_key, _ = self.cache.popitem(last=False)  # Remove oldest
                    self.stats['evictions'] += 1
                    logger.debug(f"Evicted hash from cache: {evicted_key[:8]}...")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'capacity': self.capacity,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'hit_rate': f"{hit_rate:.1f}%"
        }
    
    async def clear(self) -> None:
        """Clear the cache"""
        async with self._lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    async def get_oldest_timestamp(self) -> Optional[float]:
        """Get timestamp of oldest entry"""
        async with self._lock:
            if self.cache:
                first_key = next(iter(self.cache))
                return self.cache[first_key]
            return None

class RateLimiter:
    """Rate limiter for API calls to prevent excessive costs"""
    
    def __init__(self, max_requests_per_minute: int = 10, max_requests_per_hour: int = 100):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.minute_requests = deque()
        self.hour_requests = deque()
        self.total_requests = 0
        
    async def wait_if_needed(self) -> bool:
        """Wait if rate limit would be exceeded. Returns True if request can proceed."""
        now = time.time()
        
        # Clean old requests
        self._clean_old_requests(now)
        
        # Check minute limit
        if len(self.minute_requests) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.minute_requests[0])
            if wait_time > 0:
                logger.warning(f"Rate limit: waiting {wait_time:.1f}s for minute window")
                await asyncio.sleep(wait_time)
                self._clean_old_requests(time.time())
        
        # Check hour limit
        if len(self.hour_requests) >= self.max_requests_per_hour:
            wait_time = 3600 - (now - self.hour_requests[0])
            if wait_time > 0:
                logger.warning(f"Rate limit: waiting {wait_time:.1f}s for hour window")
                await asyncio.sleep(wait_time)
                self._clean_old_requests(time.time())
        
        # Record the request
        current_time = time.time()
        self.minute_requests.append(current_time)
        self.hour_requests.append(current_time)
        self.total_requests += 1
        
        return True
    
    def _clean_old_requests(self, now: float):
        """Remove requests older than the window"""
        # Remove requests older than 1 minute
        while self.minute_requests and self.minute_requests[0] < now - 60:
            self.minute_requests.popleft()
        
        # Remove requests older than 1 hour
        while self.hour_requests and self.hour_requests[0] < now - 3600:
            self.hour_requests.popleft()
    
    def get_stats(self) -> Dict[str, int]:
        """Get current rate limit statistics"""
        now = time.time()
        self._clean_old_requests(now)
        
        return {
            'requests_this_minute': len(self.minute_requests),
            'requests_this_hour': len(self.hour_requests),
            'total_requests': self.total_requests,
            'minute_limit': self.max_requests_per_minute,
            'hour_limit': self.max_requests_per_hour
        }

class Config:
    """Enhanced configuration management with validation"""
    
    def __init__(self):
        self.config_file = Path("config.json")
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> dict:
        """Load configuration with better validation"""
        config = DEFAULT_CONFIG.copy()
        
        # Environment variables (priority 1)
        for env_var, config_key in ENV_MAPPINGS.items():
            env_value = os.getenv(env_var)
            if env_value:
                config[config_key] = self._convert_env_value(config_key, env_value)
        
        # Config file (priority 2)
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    # Only use file config for missing env vars
                    for key, value in file_config.items():
                        if key in config and not os.getenv(key.upper()):
                            config[key] = value
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        
        return config
    
    def _convert_env_value(self, config_key: str, env_value: str):
        """Convert environment variable to proper type"""
        if config_key == 'update_channel_id':
            try:
                return int(env_value) if env_value != '0' else None
            except ValueError:
                logger.warning(f"Invalid channel ID: {env_value}")
                return None
        elif config_key in ['rss_check_interval', 'max_retries', 'retry_delay', 'owner_id',
                           'llm_requests_per_minute', 'llm_requests_per_hour', 'max_processed_hashes']:
            try:
                return int(env_value)
            except ValueError:
                logger.warning(f"Invalid integer for {config_key}: {env_value}")
                return DEFAULT_CONFIG.get(config_key, 0)
        elif config_key in ['enable_heartbeat', 'enable_llm_summaries']:
            return env_value.lower() == 'true'
        else:
            return env_value
    
    def _validate_config(self):
        """Validate critical configuration values"""
        if not self.config.get('bot_token'):
            raise ValueError("BOT_TOKEN is required")
        
        # Validate rate limiting settings
        if self.config.get('llm_requests_per_minute', 0) <= 0:
            self.config['llm_requests_per_minute'] = DEFAULT_CONFIG['llm_requests_per_minute']
        
        if self.config.get('llm_requests_per_hour', 0) <= 0:
            self.config['llm_requests_per_hour'] = DEFAULT_CONFIG['llm_requests_per_hour']
        
        logger.info("Configuration validated successfully")
    
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
    """Analyzes Big Brother updates for strategic insights and social dynamics"""
    
    def categorize_update(self, update: BBUpdate) -> List[str]:
        """Categorize an update based on its content"""
        content = f"{update.title} {update.description}".lower()
        categories = []
        
        if any(keyword in content for keyword in COMPETITION_KEYWORDS):
            categories.append("🏆 Competition")
        
        if any(keyword in content for keyword in STRATEGY_KEYWORDS):
            categories.append("🎯 Strategy")
        
        if any(keyword in content for keyword in DRAMA_KEYWORDS):
            categories.append("💥 Drama")
        
        if any(keyword in content for keyword in RELATIONSHIP_KEYWORDS):
            categories.append("💕 Romance")
        
        if any(keyword in content for keyword in ENTERTAINMENT_KEYWORDS):
            categories.append("🎬 Entertainment")
        
        return categories if categories else ["📝 General"]
    
    def extract_houseguests(self, text: str) -> List[str]:
        """Extract houseguest names from text"""
        houseguest_pattern = r'\b[A-Z][a-z]+\b'
        potential_names = re.findall(houseguest_pattern, text)
        return [name for name in potential_names if name not in EXCLUDE_WORDS]
    
    def analyze_strategic_importance(self, update: BBUpdate) -> int:
        """Rate importance from 1-10 with balanced strategic/social weighting"""
        content = f"{update.title} {update.description}".lower()
        score = 1
        
        # Strategic moments (high importance)
        if any(word in content for word in ['eviction', 'nomination', 'backdoor']):
            score += 4
        if any(word in content for word in ['hoh', 'head of household', 'power of veto']):
            score += 3
        if any(word in content for word in ['alliance', 'target', 'strategy']):
            score += 2
        if any(word in content for word in ['vote', 'voting', 'campaign']):
            score += 2
        
        # Social moments (medium-high importance for superfans)
        if any(word in content for word in ['showmance', 'kiss', 'cuddle', 'romance']):
            score += 3
        if any(word in content for word in ['fight', 'argument', 'confrontation', 'blowup']):
            score += 3
        if any(word in content for word in ['friendship', 'bond', 'close', 'trust']):
            score += 2
        
        # Entertainment moments (medium importance)
        if any(word in content for word in ['funny', 'joke', 'laugh', 'prank']):
            score += 2
        if any(word in content for word in ['crying', 'emotional', 'breakdown']):
            score += 2
        
        # House culture (low-medium importance)
        if any(word in content for word in ['tradition', 'routine', 'habit', 'inside joke']):
            score += 1
        
        # Finale night special scoring
        if any(word in content for word in ['finale', 'winner', 'crowned', 'julie']):
            if any(word in content for word in ['america', 'favorite']):
                score += 4
        
        return min(score, 10)

class AllianceTracker:
    """Tracks and analyzes Big Brother alliances"""
    
    # Alliance detection patterns
    ALLIANCE_FORMATION_PATTERNS = [
        (r"([\w\s]+) and ([\w\s]+) make a final (\d+)", "final_deal"),
        (r"([\w\s]+) forms? an? alliance with ([\w\s]+)", "alliance"),
        (r"([\w\s]+) and ([\w\s]+) agree to work together", "agreement"),
        (r"([\w\s]+) and ([\w\s]+) shake on it", "handshake"),
        (r"([\w\s]+), ([\w\s]+),? and ([\w\s]+) form an? alliance", "group_alliance"),
        (r"([\w\s]+) joins? forces with ([\w\s]+)", "joining_forces"),
        (r"([\w\s]+) wants? to work with ([\w\s]+)", "wants_work"),
        (r"([\w\s]+) trusts? ([\w\s]+) completely", "trust"),
    ]
    
    BETRAYAL_PATTERNS = [
        (r"([\w\s]+) wants? to backdoor ([\w\s]+)", "backdoor"),
        (r"([\w\s]+) throws? ([\w\s]+) under the bus", "bus"),
        (r"([\w\s]+) is now targeting ([\w\s]+)", "targeting"),
        (r"([\w\s]+) turns? on ([\w\s]+)", "turns"),
        (r"([\w\s]+) betrays? ([\w\s]+)", "betrays"),
        (r"([\w\s]+) flips? on ([\w\s]+)", "flips"),
        (r"([\w\s]+) wants? ([\w\s]+) out", "wants_out"),
    ]
    
    ALLIANCE_NAME_PATTERNS = [
        r"(?:The|the) ([\w\s]+) alliance",
        r"alliance (?:called|named) (?:The|the)? ?([\w\s]+)",
        r"(?:The|the) ([\w\s]+) \([\w\s,]+\)",  # The Core (Chelsie, Cam, etc)
    ]
    
    def __init__(self, db_path: str = "bb_updates.db"):
        self.db_path = db_path
        self.init_alliance_tables()
        self._alliance_cache = {}
        self._member_cache = defaultdict(set)  # houseguest -> alliance_ids
    
    def get_connection(self):
        """Get database connection with datetime support"""
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        return conn
    
    def init_alliance_tables(self):
        """Initialize alliance tracking tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Main alliances table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alliances (
                alliance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                formed_date TIMESTAMP,
                dissolved_date TIMESTAMP,
                status TEXT DEFAULT 'active',
                confidence_level INTEGER DEFAULT 50,
                last_activity TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Alliance members table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alliance_members (
                alliance_id INTEGER,
                houseguest_name TEXT,
                joined_date TIMESTAMP,
                left_date TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (alliance_id) REFERENCES alliances(alliance_id),
                UNIQUE(alliance_id, houseguest_name)
            )
        """)
        
        # Alliance events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alliance_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alliance_id INTEGER,
                event_type TEXT,
                description TEXT,
                involved_houseguests TEXT,
                timestamp TIMESTAMP,
                update_hash TEXT,
                FOREIGN KEY (alliance_id) REFERENCES alliances(alliance_id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alliance_status ON alliances(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alliance_members ON alliance_members(houseguest_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alliance_events_type ON alliance_events(event_type)")
        
        conn.commit()
        conn.close()
        
        logger.info("Alliance tracking tables initialized")
    
    def analyze_update_for_alliances(self, update: BBUpdate) -> List[Dict]:
        """Analyze an update for alliance information"""
        content = f"{update.title} {update.description}".strip()
        detected_events = []
        
        # Skip finale/voting/competition winner updates
        skip_phrases = [
            'votes for', 'voted for', 'to be the winner', 'winner of big brother',
            'jury vote', 'crown the winner', 'wins bb', 'wins hoh', 'wins pov',
            'wins the power', 'eviction vote', 'evicted', 'julie pulls the keys',
            'america\'s favorite', 'afp', 'finale', 'final vote', 'cast their vote',
            'announces the winner', 'wins big brother', 'jury votes', 'key to vote',
            'official with a vote', 'makes it official'
        ]
        
        content_lower = content.lower()
        if any(phrase in content_lower for phrase in skip_phrases):
            return []  # Don't process these updates for alliances
        
        # Also skip if it's clearly a competition/ceremony result
        if re.search(r'wins?\s+(the\s+)?(hoh|pov|veto|competition|challenge|big\s+brother)', content, re.IGNORECASE):
            return []
        
        # Skip eviction and ceremony updates
        if re.search(r'(eviction|nomination|veto)\s+ceremony', content, re.IGNORECASE):
            return []
        
        # Check for alliance formations
        for pattern, pattern_type in self.ALLIANCE_FORMATION_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                houseguests = [g.strip() for g in groups if g and not g.isdigit()]
                
                # Filter out common words that aren't houseguests
                houseguests = [hg for hg in houseguests if hg not in EXCLUDE_WORDS]
                
                # Additional filtering for common false positives
                houseguests = [hg for hg in houseguests if len(hg) > 2 and not hg.lower() in ['big', 'brother', 'julie', 'host', 'america']]
                
                if len(houseguests) >= 2:
                    detected_events.append({
                        'type': AllianceEventType.FORMED,
                        'houseguests': houseguests,
                        'pattern_type': pattern_type,
                        'confidence': self._calculate_confidence(pattern_type),
                        'update': update
                    })
        
        # Check for betrayals
        for pattern, pattern_type in self.BETRAYAL_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    betrayer = groups[0].strip()
                    betrayed = groups[1].strip()
                    
                    if betrayer not in EXCLUDE_WORDS and betrayed not in EXCLUDE_WORDS:
                        if len(betrayer) > 2 and len(betrayed) > 2:  # Additional length check
                            detected_events.append({
                                'type': AllianceEventType.BETRAYAL,
                                'betrayer': betrayer,
                                'betrayed': betrayed,
                                'pattern_type': pattern_type,
                                'update': update
                            })
        
        # Check for named alliances
        for pattern in self.ALLIANCE_NAME_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                alliance_name = match.group(1).strip()
                # Skip if alliance name contains finale/winner phrases
                if alliance_name and len(alliance_name) > 2:
                    if not any(skip in alliance_name.lower() for skip in ['winner', 'big brother', 'finale']):
                        # Try to extract members mentioned nearby
                        members = self._extract_nearby_houseguests(content, match.start(), match.end())
                        detected_events.append({
                            'type': AllianceEventType.FORMED,
                            'alliance_name': alliance_name,
                            'houseguests': members,
                            'pattern_type': 'named_alliance',
                            'confidence': 80,  # Named alliances have higher confidence
                            'update': update
                        })
        
        return detected_events
    
    def _calculate_confidence(self, pattern_type: str) -> int:
        """Calculate confidence level based on pattern type"""
        confidence_map = {
            'final_deal': 90,
            'alliance': 85,
            'handshake': 80,
            'group_alliance': 85,
            'agreement': 75,
            'joining_forces': 70,
            'wants_work': 50,
            'trust': 60,
            'named_alliance': 80
        }
        return confidence_map.get(pattern_type, 50)
    
    def _extract_nearby_houseguests(self, content: str, start: int, end: int, window: int = 100) -> List[str]:
        """Extract houseguest names near an alliance mention"""
        # Look in a window around the alliance name
        search_start = max(0, start - window)
        search_end = min(len(content), end + window)
        search_text = content[search_start:search_end]
        
        # Find capitalized words that could be names
        potential_names = re.findall(r'\b[A-Z][a-z]+\b', search_text)
        
        # Filter out common words
        houseguests = [name for name in potential_names if name not in EXCLUDE_WORDS]
        
        return list(set(houseguests))  # Remove duplicates
    
    def process_alliance_event(self, event: Dict) -> Optional[int]:
        """Process a detected alliance event and update database"""
        try:
            if event['type'] == AllianceEventType.FORMED:
                return self._handle_alliance_formation(event)
            elif event['type'] == AllianceEventType.BETRAYAL:
                return self._handle_betrayal(event)
            
        except Exception as e:
            logger.error(f"Error processing alliance event: {e}")
            return None
    
    def _handle_alliance_formation(self, event: Dict) -> Optional[int]:
        """Handle alliance formation event"""
        houseguests = event.get('houseguests', [])
        if len(houseguests) < 2:
            return None
        
        # Check if these houseguests already have an alliance together
        existing_alliance = self._find_existing_alliance(houseguests)
        
        if existing_alliance:
            # Update confidence and last activity
            self._update_alliance_confidence(existing_alliance['alliance_id'], event['confidence'])
            return existing_alliance['alliance_id']
        else:
            # Create new alliance
            alliance_name = event.get('alliance_name')
            if not alliance_name:
                # Generate a name if not provided
                alliance_name = f"{houseguests[0]}/{houseguests[1]}"
                if len(houseguests) > 2:
                    alliance_name += f" +{len(houseguests)-2}"
            
            return self._create_alliance(
                name=alliance_name,
                members=houseguests,
                confidence=event['confidence'],
                formed_date=event['update'].pub_date
            )
    
    def _handle_betrayal(self, event: Dict) -> None:
        """Handle betrayal event"""
        betrayer = event['betrayer']
        betrayed = event['betrayed']
        
        # Find alliances containing both houseguests
        shared_alliances = self._find_shared_alliances(betrayer, betrayed)
        
        for alliance in shared_alliances:
            # Record betrayal event
            self._record_alliance_event(
                alliance_id=alliance['alliance_id'],
                event_type=AllianceEventType.BETRAYAL,
                description=f"{betrayer} betrayed {betrayed}",
                involved=[betrayer, betrayed],
                timestamp=event['update'].pub_date,
                update_hash=event['update'].content_hash
            )
            
            # Consider marking alliance as broken if confidence drops
            self._update_alliance_confidence(alliance['alliance_id'], -30)
    
    def _create_alliance(self, name: str, members: List[str], confidence: int, formed_date: datetime) -> int:
        """Create a new alliance in the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Insert alliance
            cursor.execute("""
                INSERT INTO alliances (name, formed_date, status, confidence_level, last_activity)
                VALUES (?, ?, ?, ?, ?)
            """, (name, formed_date, AllianceStatus.ACTIVE.value, confidence, formed_date))
            
            alliance_id = cursor.lastrowid
            
            # Insert members
            for member in members:
                cursor.execute("""
                    INSERT OR IGNORE INTO alliance_members (alliance_id, houseguest_name, joined_date)
                    VALUES (?, ?, ?)
                """, (alliance_id, member, formed_date))
            
            # Record formation event
            cursor.execute("""
                INSERT INTO alliance_events (alliance_id, event_type, description, involved_houseguests, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (alliance_id, AllianceEventType.FORMED.value, 
                  f"Alliance '{name}' formed", ",".join(members), formed_date))
            
            conn.commit()
            logger.info(f"Created new alliance: {name} with members {members}")
            
            return alliance_id
            
        except Exception as e:
            logger.error(f"Error creating alliance: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_active_alliances(self) -> List[Dict]:
        """Get all currently active alliances"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT a.alliance_id, a.name, a.confidence_level, a.last_activity,
                       GROUP_CONCAT(am.houseguest_name) as members
                FROM alliances a
                JOIN alliance_members am ON a.alliance_id = am.alliance_id
                WHERE a.status = ? AND am.is_active = 1
                GROUP BY a.alliance_id
                ORDER BY a.confidence_level DESC, a.last_activity DESC
            """, (AllianceStatus.ACTIVE.value,))
            
            alliances = []
            for row in cursor.fetchall():
                alliances.append({
                    'alliance_id': row[0],
                    'name': row[1],
                    'confidence': row[2],
                    'last_activity': row[3],
                    'members': row[4].split(',') if row[4] else []
                })
            
            return alliances
            
        except Exception as e:
            logger.error(f"Error getting active alliances: {e}")
            return []
        finally:
            conn.close()
    
    def create_alliance_map_embed(self) -> discord.Embed:
        """Create an embed showing current alliance relationships"""
        alliances = self.get_active_alliances()
        broken_alliances = self.get_recently_broken_alliances()
        
        if not alliances and not broken_alliances:
            embed = discord.Embed(
                title="🤝 Big Brother Alliance Map",
                description="No active alliances detected yet",
                color=0x95a5a6,
                timestamp=datetime.now()
            )
            return embed
        
        embed = discord.Embed(
            title="🤝 Big Brother Alliance Map",
            description=f"**{len(alliances)} Active Alliances**",
            color=0x3498db,
            timestamp=datetime.now()
        )
        
        # Group alliances by confidence
        high_conf = [a for a in alliances if a['confidence'] >= 70]
        med_conf = [a for a in alliances if 40 <= a['confidence'] < 70]
        low_conf = [a for a in alliances if a['confidence'] < 40]
        
        if high_conf:
            alliance_text = []
            for alliance in high_conf[:5]:  # Limit to 5
                members_str = " + ".join([f"**{m}**" for m in alliance['members']])
                alliance_text.append(f"🔗 {alliance['name']}\n{members_str}")
            
            embed.add_field(
                name="💪 Strong Alliances",
                value="\n\n".join(alliance_text),
                inline=False
            )
        
        if med_conf:
            alliance_text = []
            for alliance in med_conf[:3]:  # Limit to 3
                members_str = " + ".join(alliance['members'])
                alliance_text.append(f"🤝 {alliance['name']}: {members_str}")
            
            embed.add_field(
                name="🤔 Suspected Alliances",
                value="\n".join(alliance_text),
                inline=False
            )
        
        # Add recently broken strong alliances
        if broken_alliances:
            broken_text = []
            for alliance in broken_alliances[:3]:  # Limit to 3
                members_str = " + ".join(alliance['members'])
                days_ago = (datetime.now() - datetime.fromisoformat(alliance['broken_date'])).days
                broken_text.append(f"💔 {alliance['name']}: {members_str}\n   *Broke {days_ago}d ago after {alliance['days_strong']} days*")
            
            embed.add_field(
                name="⚰️ Recently Broken Alliances",
                value="\n\n".join(broken_text),
                inline=False
            )
        
        # Add recent betrayals
        recent_betrayals = self.get_recent_betrayals(days=3)
        if recent_betrayals:
            betrayal_text = []
            for betrayal in recent_betrayals[:3]:
                betrayal_text.append(f"⚡ {betrayal['description']}")
            
            embed.add_field(
                name="💥 Recent Betrayals",
                value="\n".join(betrayal_text),
                inline=False
            )
        
        embed.set_footer(text="Alliance confidence based on feed activity")
        
        return embed
    
    def get_recently_broken_alliances(self, days: int = 7) -> List[Dict]:
        """Get alliances that were strong but recently broke"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT DISTINCT a.alliance_id, a.name, a.formed_date, 
                       MAX(ae.timestamp) as broken_date,
                       GROUP_CONCAT(DISTINCT am.houseguest_name) as members
                FROM alliances a
                JOIN alliance_members am ON a.alliance_id = am.alliance_id
                JOIN alliance_events ae ON a.alliance_id = ae.alliance_id
                WHERE a.status IN (?, ?)
                  AND ae.event_type = ? 
                  AND datetime(ae.timestamp) > datetime(?)
                  AND a.confidence_level >= 70
                GROUP BY a.alliance_id
                ORDER BY broken_date DESC
            """, (AllianceStatus.BROKEN.value, AllianceStatus.DISSOLVED.value,
                  AllianceEventType.BETRAYAL.value, cutoff_date))
            
            broken_alliances = []
            for row in cursor.fetchall():
                alliance_id, name, formed_date, broken_date, members = row
                
                try:
                    formed = datetime.fromisoformat(formed_date) if isinstance(formed_date, str) else formed_date
                    broken = datetime.fromisoformat(broken_date) if isinstance(broken_date, str) else broken_date
                    days_strong = (broken - formed).days
                except:
                    days_strong = 0
                
                broken_alliances.append({
                    'alliance_id': alliance_id,
                    'name': name,
                    'members': members.split(',') if members else [],
                    'formed_date': formed_date,
                    'broken_date': broken_date,
                    'days_strong': days_strong
                })
            
            return broken_alliances
            
        except Exception as e:
            logger.error(f"Error getting broken alliances: {e}")
            return []
        finally:
            conn.close()
    
    def get_houseguest_loyalty_embed(self, houseguest: str) -> discord.Embed:
        """Create an embed showing a houseguest's alliance history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT a.name, a.status, am.joined_date, am.left_date, a.confidence_level,
                       GROUP_CONCAT(am2.houseguest_name) as all_members
                FROM alliance_members am
                JOIN alliances a ON am.alliance_id = a.alliance_id
                JOIN alliance_members am2 ON am2.alliance_id = a.alliance_id
                WHERE am.houseguest_name = ?
                GROUP BY a.alliance_id
                ORDER BY am.joined_date DESC
            """, (houseguest,))
            
            alliances = cursor.fetchall()
            
            cursor.execute("""
                SELECT COUNT(*) FROM alliance_events
                WHERE event_type = ? AND involved_houseguests LIKE ?
            """, (AllianceEventType.BETRAYAL.value, f"%{houseguest}%"))
            
            betrayal_count = cursor.fetchone()[0]
            
            embed = discord.Embed(
                title=f"🎭 {houseguest}'s Alliance History",
                color=0xe74c3c if betrayal_count > 2 else 0x2ecc71,
                timestamp=datetime.now()
            )
            
            if not alliances:
                embed.description = f"{houseguest} has not been detected in any alliances"
                return embed
            
            active_alliances = sum(1 for a in alliances if a[1] == AllianceStatus.ACTIVE.value)
            broken_alliances = sum(1 for a in alliances if a[1] in [AllianceStatus.BROKEN.value, AllianceStatus.DISSOLVED.value])
            
            loyalty_score = max(0, 100 - (betrayal_count * 20) - (broken_alliances * 10))
            loyalty_emoji = "🏆" if loyalty_score >= 80 else "⚠️" if loyalty_score >= 50 else "🚨"
            
            embed.description = f"**Loyalty Score**: {loyalty_emoji} {loyalty_score}/100\n"
            embed.description += f"**Betrayals**: {betrayal_count} | **Active Alliances**: {active_alliances}"
            
            alliance_text = []
            for alliance in alliances[:6]:
                name, status, joined, left, confidence, members = alliance
                status_emoji = "✅" if status == AllianceStatus.ACTIVE.value else "❌"
                
                other_members = [m for m in members.split(',') if m != houseguest]
                members_str = ", ".join(other_members[:3])
                if len(other_members) > 3:
                    members_str += f" +{len(other_members)-3}"
                
                alliance_text.append(f"{status_emoji} **{name}** (w/ {members_str})")
            
            embed.add_field(
                name="📋 Alliance History",
                value="\n".join(alliance_text) if alliance_text else "No alliances found",
                inline=False
            )
            
            return embed
            
        except Exception as e:
            logger.error(f"Error in loyalty embed: {e}")
            embed = discord.Embed(
                title=f"🎭 {houseguest}'s Alliance History",
                description="Error retrieving alliance data",
                color=0xe74c3c
            )
            return embed
        finally:
            conn.close()
    
    def get_recent_betrayals(self, days: int = 7) -> List[Dict]:
        """Get recent betrayal events"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT description, timestamp, involved_houseguests
                FROM alliance_events
                WHERE event_type = ? AND datetime(timestamp) > datetime(?)
                ORDER BY timestamp DESC
            """, (AllianceEventType.BETRAYAL.value, cutoff_date))
            
            betrayals = []
            for row in cursor.fetchall():
                betrayals.append({
                    'description': row[0],
                    'timestamp': row[1],
                    'involved': row[2].split(',') if row[2] else []
                })
            
            return betrayals
            
        except Exception as e:
            logger.error(f"Error getting betrayals: {e}")
            return []
        finally:
            conn.close()
    
    def _find_existing_alliance(self, houseguests: List[str]) -> Optional[Dict]:
        """Find if these houseguests already have an alliance"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get all active alliances for first houseguest
            cursor.execute("""
                SELECT DISTINCT a.alliance_id, a.name, a.confidence_level
                FROM alliance_members am
                JOIN alliances a ON am.alliance_id = a.alliance_id
                WHERE am.houseguest_name = ? AND a.status = ? AND am.is_active = 1
            """, (houseguests[0], AllianceStatus.ACTIVE.value))
            
            for row in cursor.fetchall():
                alliance_id = row[0]
                
                # Check if all houseguests are in this alliance
                all_in = True
                for hg in houseguests[1:]:
                    cursor.execute("""
                        SELECT 1 FROM alliance_members
                        WHERE alliance_id = ? AND houseguest_name = ? AND is_active = 1
                    """, (alliance_id, hg))
                    
                    if not cursor.fetchone():
                        all_in = False
                        break
                
                if all_in:
                    return {
                        'alliance_id': alliance_id,
                        'name': row[1],
                        'confidence': row[2]
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding existing alliance: {e}")
            return None
        finally:
            conn.close()
    
    def _find_shared_alliances(self, hg1: str, hg2: str) -> List[Dict]:
        """Find alliances containing both houseguests"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT DISTINCT a.alliance_id, a.name
                FROM alliance_members am1
                JOIN alliance_members am2 ON am1.alliance_id = am2.alliance_id
                JOIN alliances a ON am1.alliance_id = a.alliance_id
                WHERE am1.houseguest_name = ? AND am2.houseguest_name = ?
                      AND am1.is_active = 1 AND am2.is_active = 1
                      AND a.status = ?
            """, (hg1, hg2, AllianceStatus.ACTIVE.value))
            
            alliances = []
            for row in cursor.fetchall():
                alliances.append({
                    'alliance_id': row[0],
                    'name': row[1]
                })
            
            return alliances
            
        except Exception as e:
            logger.error(f"Error finding shared alliances: {e}")
            return []
        finally:
            conn.close()
    
    def _update_alliance_confidence(self, alliance_id: int, change: int):
        """Update alliance confidence level"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get current confidence and status
            cursor.execute("""
                SELECT confidence_level, status, formed_date 
                FROM alliances WHERE alliance_id = ?
            """, (alliance_id,))
            
            current_conf, status, formed_date = cursor.fetchone()
            
            # Track if this was a strong alliance before the change
            was_strong = current_conf >= 70 and status == AllianceStatus.ACTIVE.value
            
            # Calculate how long it's been active if it was strong
            if was_strong:
                formed = datetime.fromisoformat(formed_date) if isinstance(formed_date, str) else formed_date
                days_active = (datetime.now() - formed).days
            else:
                days_active = 0
            
            # Update confidence
            new_confidence = max(0, min(100, current_conf + change))
            
            cursor.execute("""
                UPDATE alliances 
                SET confidence_level = ?,
                    last_activity = ?
                WHERE alliance_id = ?
            """, (new_confidence, datetime.now(), alliance_id))
            
            # Check if confidence dropped too low
            if new_confidence < 20:
                # Mark alliance as broken
                new_status = AllianceStatus.BROKEN.value
                cursor.execute("""
                    UPDATE alliances SET status = ? WHERE alliance_id = ?
                """, (new_status, alliance_id))
                
                # If this was a strong alliance for over a week, record it as a major break
                if was_strong and days_active >= 7:
                    cursor.execute("""
                        INSERT INTO alliance_events 
                        (alliance_id, event_type, description, timestamp)
                        VALUES (?, ?, ?, ?)
                    """, (alliance_id, AllianceEventType.DISSOLVED.value,
                          f"Alliance dissolved after {days_active} days", datetime.now()))
                    
                    logger.info(f"Major alliance break: Alliance {alliance_id} was strong for {days_active} days")
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error updating alliance confidence: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _record_alliance_event(self, alliance_id: int, event_type: AllianceEventType, 
                             description: str, involved: List[str], 
                             timestamp: datetime, update_hash: str = None):
        """Record an alliance event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO alliance_events 
                (alliance_id, event_type, description, involved_houseguests, timestamp, update_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (alliance_id, event_type.value, description, 
                  ",".join(involved), timestamp, update_hash))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error recording alliance event: {e}")
            conn.rollback()
        finally:
            conn.close()

class PredictionSystem:
    """NEW: Prediction Leaderboard System for Big Brother predictions"""
    
    def __init__(self, db_path: str = "bb_updates.db"):
        self.db_path = db_path
        self.init_prediction_tables()
    
    def get_connection(self):
        """Get database connection with datetime support"""
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        return conn
    
    def init_prediction_tables(self):
        """Initialize prediction system tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Prediction categories with different point values
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_categories (
                category_id TEXT PRIMARY KEY,
                name TEXT UNIQUE,
                points_value INTEGER,
                description TEXT
            )
        """)
        
        # Individual prediction polls
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_polls (
                poll_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id TEXT,
                title TEXT,
                description TEXT,
                created_by INTEGER,
                created_at TIMESTAMP,
                closes_at TIMESTAMP,
                status TEXT DEFAULT 'open',
                correct_option_id INTEGER,
                FOREIGN KEY (category_id) REFERENCES prediction_categories(category_id)
            )
        """)
        
        # Poll options
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS poll_options (
                option_id INTEGER PRIMARY KEY AUTOINCREMENT,
                poll_id INTEGER,
                option_text TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (poll_id) REFERENCES prediction_polls(poll_id)
            )
        """)
        
        # User predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                poll_id INTEGER,
                user_id INTEGER,
                option_id INTEGER,
                predicted_at TIMESTAMP,
                points_earned INTEGER DEFAULT 0,
                FOREIGN KEY (poll_id) REFERENCES prediction_polls(poll_id),
                FOREIGN KEY (option_id) REFERENCES poll_options(option_id),
                UNIQUE(poll_id, user_id)
            )
        """)
        
        # Leaderboard cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_leaderboard (
                user_id INTEGER PRIMARY KEY,
                total_points INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                total_predictions INTEGER DEFAULT 0,
                last_updated TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_polls_status ON prediction_polls(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_polls_closes ON prediction_polls(closes_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user ON user_predictions(user_id)")
        
        # Insert default categories
        for category_id, category_data in PREDICTION_CATEGORIES.items():
            cursor.execute("""
                INSERT OR IGNORE INTO prediction_categories (category_id, name, points_value, description)
                VALUES (?, ?, ?, ?)
            """, (category_id, category_data['name'], category_data['points'], 
                  f"Predict the {category_data['name'].lower()}"))
        
        conn.commit()
        conn.close()
        
        logger.info("Prediction system tables initialized")
    
    def create_poll(self, category_id: str, title: str, description: str, 
                   created_by: int, hours_open: int, options: List[str]) -> Optional[int]:
        """Create a new prediction poll"""
        if category_id not in PREDICTION_CATEGORIES:
            return None
        
        if len(options) < 2 or len(options) > 10:
            return None
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            closes_at = datetime.now() + timedelta(hours=hours_open)
            
            # Insert poll
            cursor.execute("""
                INSERT INTO prediction_polls (category_id, title, description, created_by, created_at, closes_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (category_id, title, description, created_by, datetime.now(), closes_at))
            
            poll_id = cursor.lastrowid
            
            # Insert options
            for option_text in options:
                cursor.execute("""
                    INSERT INTO poll_options (poll_id, option_text, created_at)
                    VALUES (?, ?, ?)
                """, (poll_id, option_text.strip(), datetime.now()))
            
            conn.commit()
            logger.info(f"Created poll {poll_id}: {title}")
            return poll_id
            
        except Exception as e:
            logger.error(f"Error creating poll: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def vote_on_poll(self, poll_id: int, user_id: int, option_text: str) -> Dict[str, any]:
        """Vote on a prediction poll"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Check if poll exists and is open
            cursor.execute("""
                SELECT status, closes_at FROM prediction_polls WHERE poll_id = ?
            """, (poll_id,))
            
            poll_info = cursor.fetchone()
            if not poll_info:
                return {'success': False, 'error': 'Poll not found'}
            
            status, closes_at = poll_info
            if status != 'open':
                return {'success': False, 'error': 'Poll is closed'}
            
            if datetime.now() > closes_at:
                # Auto-close expired poll
                cursor.execute("""
                    UPDATE prediction_polls SET status = 'closed' WHERE poll_id = ?
                """, (poll_id,))
                conn.commit()
                return {'success': False, 'error': 'Poll has expired'}
            
            # Find the option
            cursor.execute("""
                SELECT option_id FROM poll_options WHERE poll_id = ? AND option_text = ?
            """, (poll_id, option_text))
            
            option_result = cursor.fetchone()
            if not option_result:
                return {'success': False, 'error': 'Option not found'}
            
            option_id = option_result[0]
            
            # Check if user already voted
            cursor.execute("""
                SELECT prediction_id FROM user_predictions WHERE poll_id = ? AND user_id = ?
            """, (poll_id, user_id))
            
            if cursor.fetchone():
                return {'success': False, 'error': 'You have already voted on this poll'}
            
            # Insert vote
            cursor.execute("""
                INSERT INTO user_predictions (poll_id, user_id, option_id, predicted_at)
                VALUES (?, ?, ?, ?)
            """, (poll_id, user_id, option_id, datetime.now()))
            
            conn.commit()
            return {'success': True, 'message': f'Vote recorded for "{option_text}"'}
            
        except Exception as e:
            logger.error(f"Error voting on poll: {e}")
            conn.rollback()
            return {'success': False, 'error': 'Database error'}
        finally:
            conn.close()
    
    def close_poll(self, poll_id: int) -> bool:
        """Manually close a poll"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE prediction_polls SET status = 'closed' WHERE poll_id = ? AND status = 'open'
            """, (poll_id,))
            
            success = cursor.rowcount > 0
            conn.commit()
            
            if success:
                logger.info(f"Poll {poll_id} manually closed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error closing poll: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def resolve_poll(self, poll_id: int, winning_option: str) -> Dict[str, any]:
        """Resolve a poll and award points"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get poll info
            cursor.execute("""
                SELECT category_id, status FROM prediction_polls WHERE poll_id = ?
            """, (poll_id,))
            
            poll_info = cursor.fetchone()
            if not poll_info:
                return {'success': False, 'error': 'Poll not found'}
            
            category_id, status = poll_info
            if status == 'resolved':
                return {'success': False, 'error': 'Poll already resolved'}
            
            # Find winning option
            cursor.execute("""
                SELECT option_id FROM poll_options WHERE poll_id = ? AND option_text = ?
            """, (poll_id, winning_option))
            
            option_result = cursor.fetchone()
            if not option_result:
                return {'success': False, 'error': 'Winning option not found'}
            
            winning_option_id = option_result[0]
            points_value = PREDICTION_CATEGORIES[category_id]['points']
            
            # Update poll status and set correct option
            cursor.execute("""
                UPDATE prediction_polls SET status = 'resolved', correct_option_id = ? WHERE poll_id = ?
            """, (winning_option_id, poll_id))
            
            # Award points to correct predictions
            cursor.execute("""
                UPDATE user_predictions SET points_earned = ? 
                WHERE poll_id = ? AND option_id = ?
            """, (points_value, poll_id, winning_option_id))
            
            # Get winners for response
            cursor.execute("""
                SELECT COUNT(*) FROM user_predictions WHERE poll_id = ? AND option_id = ?
            """, (poll_id, winning_option_id))
            
            winner_count = cursor.fetchone()[0]
            
            # Update leaderboard
            self._update_leaderboard_for_poll(cursor, poll_id)
            
            conn.commit()
            
            logger.info(f"Poll {poll_id} resolved: {winning_option} won, {winner_count} correct predictions")
            
            return {
                'success': True, 
                'message': f'Poll resolved! "{winning_option}" was correct.',
                'winner_count': winner_count,
                'points_awarded': points_value
            }
            
        except Exception as e:
            logger.error(f"Error resolving poll: {e}")
            conn.rollback()
            return {'success': False, 'error': 'Database error'}
        finally:
            conn.close()
    
    def _update_leaderboard_for_poll(self, cursor, poll_id: int):
        """Update leaderboard entries for a resolved poll"""
        # Get all predictions for this poll
        cursor.execute("""
            SELECT user_id, points_earned FROM user_predictions WHERE poll_id = ?
        """, (poll_id,))
        
        predictions = cursor.fetchall()
        
        for user_id, points_earned in predictions:
            # Update or insert leaderboard entry
            cursor.execute("""
                INSERT INTO prediction_leaderboard (user_id, total_points, correct_predictions, total_predictions, last_updated)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    total_points = total_points + ?,
                    correct_predictions = correct_predictions + ?,
                    total_predictions = total_predictions + 1,
                    last_updated = ?
            """, (user_id, points_earned, 1 if points_earned > 0 else 0, datetime.now(),
                  points_earned, 1 if points_earned > 0 else 0, datetime.now()))
    
    def get_active_polls(self) -> List[Dict]:
        """Get all currently active polls"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Auto-close expired polls first
            cursor.execute("""
                UPDATE prediction_polls SET status = 'closed' 
                WHERE status = 'open' AND closes_at < ?
            """, (datetime.now(),))
            
            # Get active polls
            cursor.execute("""
                SELECT p.poll_id, p.title, p.description, pc.name as category_name, 
                       pc.points_value, p.closes_at, p.created_at,
                       COUNT(up.prediction_id) as vote_count
                FROM prediction_polls p
                JOIN prediction_categories pc ON p.category_id = pc.category_id
                LEFT JOIN user_predictions up ON p.poll_id = up.poll_id
                WHERE p.status = 'open'
                GROUP BY p.poll_id
                ORDER BY p.closes_at ASC
            """, )
            
            polls = []
            for row in cursor.fetchall():
                polls.append({
                    'poll_id': row[0],
                    'title': row[1],
                    'description': row[2],
                    'category': row[3],
                    'points_value': row[4],
                    'closes_at': row[5],
                    'created_at': row[6],
                    'vote_count': row[7]
                })
            
            conn.commit()
            return polls
            
        except Exception as e:
            logger.error(f"Error getting active polls: {e}")
            return []
        finally:
            conn.close()
    
    def get_poll_details(self, poll_id: int) -> Optional[Dict]:
        """Get detailed information about a specific poll"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get poll info
            cursor.execute("""
                SELECT p.poll_id, p.title, p.description, pc.name as category_name,
                       pc.points_value, p.status, p.closes_at, p.created_at
                FROM prediction_polls p
                JOIN prediction_categories pc ON p.category_id = pc.category_id
                WHERE p.poll_id = ?
            """, (poll_id,))
            
            poll_info = cursor.fetchone()
            if not poll_info:
                return None
            
            # Get options with vote counts
            cursor.execute("""
                SELECT po.option_text, COUNT(up.prediction_id) as vote_count
                FROM poll_options po
                LEFT JOIN user_predictions up ON po.option_id = up.option_id
                WHERE po.poll_id = ?
                GROUP BY po.option_id, po.option_text
                ORDER BY vote_count DESC
            """, (poll_id,))
            
            options = cursor.fetchall()
            
            return {
                'poll_id': poll_info[0],
                'title': poll_info[1],
                'description': poll_info[2],
                'category': poll_info[3],
                'points_value': poll_info[4],
                'status': poll_info[5],
                'closes_at': poll_info[6],
                'created_at': poll_info[7],
                'options': [{'text': opt[0], 'votes': opt[1]} for opt in options]
            }
            
        except Exception as e:
            logger.error(f"Error getting poll details: {e}")
            return None
        finally:
            conn.close()
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get prediction leaderboard"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT user_id, total_points, correct_predictions, total_predictions,
                       CASE WHEN total_predictions > 0 
                            THEN ROUND(CAST(correct_predictions AS FLOAT) / total_predictions * 100, 1)
                            ELSE 0 END as accuracy
                FROM prediction_leaderboard
                ORDER BY total_points DESC, accuracy DESC
                LIMIT ?
            """, (limit,))
            
            leaderboard = []
            for i, row in enumerate(cursor.fetchall(), 1):
                leaderboard.append({
                    'rank': i,
                    'user_id': row[0],
                    'total_points': row[1],
                    'correct_predictions': row[2],
                    'total_predictions': row[3],
                    'accuracy': row[4]
                })
            
            return leaderboard
            
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []
        finally:
            conn.close()
    
    def get_user_predictions(self, user_id: int, category: str = None) -> List[Dict]:
        """Get a user's prediction history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if category:
                cursor.execute("""
                    SELECT p.title, pc.name as category, po.option_text, 
                           up.points_earned, p.status, up.predicted_at
                    FROM user_predictions up
                    JOIN prediction_polls p ON up.poll_id = p.poll_id
                    JOIN prediction_categories pc ON p.category_id = pc.category_id
                    JOIN poll_options po ON up.option_id = po.option_id
                    WHERE up.user_id = ? AND p.category_id = ?
                    ORDER BY up.predicted_at DESC
                """, (user_id, category))
            else:
                cursor.execute("""
                    SELECT p.title, pc.name as category, po.option_text, 
                           up.points_earned, p.status, up.predicted_at
                    FROM user_predictions up
                    JOIN prediction_polls p ON up.poll_id = p.poll_id
                    JOIN prediction_categories pc ON p.category_id = pc.category_id
                    JOIN poll_options po ON up.option_id = po.option_id
                    WHERE up.user_id = ?
                    ORDER BY up.predicted_at DESC
                """, (user_id,))
            
            predictions = []
            for row in cursor.fetchall():
                predictions.append({
                    'title': row[0],
                    'category': row[1],
                    'prediction': row[2],
                    'points_earned': row[3],
                    'status': row[4],
                    'predicted_at': row[5]
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting user predictions: {e}")
            return []
        finally:
            conn.close()

class UpdateBatcher:
    """Groups and analyzes updates like a BB superfan using LLM intelligence"""
    
    def __init__(self, analyzer: BBAnalyzer, config: Config):
        self.analyzer = analyzer
        self.config = config
        self.update_queue = []
        self.last_batch_time = datetime.now()
        
        # Replace the set with LRU cache
        max_hashes = config.get('max_processed_hashes', 10000)
        self.processed_hashes_cache = LRUCache(capacity=max_hashes)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=config.get('llm_requests_per_minute', 10),
            max_requests_per_hour=config.get('llm_requests_per_hour', 100)
        )
        
        # Initialize Anthropic client
        self.llm_client = None
        self.llm_model = config.get('llm_model', 'claude-3-haiku-20240307')
        self._init_llm_client()
    
    def _init_llm_client(self):
        """Initialize LLM client with proper error handling"""
        api_key = self.config.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY', '')
        
        if not api_key.strip():
            logger.warning("No valid Anthropic API key provided")
            return
        
        try:
            self.llm_client = anthropic.Anthropic(api_key=api_key.strip())
            logger.info("LLM integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm_client = None
    
    def should_send_batch(self) -> bool:
        """Determine if we should send a batch now"""
        if not self.update_queue:
            return False
            
        time_elapsed = (datetime.now() - self.last_batch_time).total_seconds() / 60
        
        # Check for urgent updates
        has_urgent = any(self._is_urgent(update) for update in self.update_queue)
        if has_urgent and len(self.update_queue) >= 2:
            return True
        
        # High activity mode: 10+ updates or 15 minutes
        if len(self.update_queue) >= 10 or time_elapsed >= 15:
            return True
            
        # Normal mode: 5+ updates or 30 minutes
        if len(self.update_queue) >= 5 or time_elapsed >= 30:
            return True
            
        return False
    
    def _is_urgent(self, update: BBUpdate) -> bool:
        """Check if update contains game-critical information"""
        content = f"{update.title} {update.description}".lower()
        return any(keyword in content for keyword in URGENT_KEYWORDS)
    
    async def add_update(self, update: BBUpdate):
        """Add update to queue if not already processed"""
        # Use async method for thread safety
        if not await self.processed_hashes_cache.contains(update.content_hash):
            self.update_queue.append(update)
            await self.processed_hashes_cache.add(update.content_hash)
            
            # Log cache stats periodically
            cache_stats = self.processed_hashes_cache.get_stats()
            if cache_stats['size'] % 1000 == 0:  # Log every 1000 entries
                logger.info(f"Hash cache stats: {cache_stats}")
    
    async def create_batch_summary(self) -> List[discord.Embed]:
        """Create intelligent summary embeds using LLM if available"""
        if not self.update_queue:
            return []
        
        embeds = []
        
        # Use LLM if available and rate limits allow
        if self.llm_client and await self._can_make_llm_request():
            try:
                embeds = await self._create_llm_summary()
            except Exception as e:
                logger.error(f"LLM summary failed: {e}")
                embeds = self._create_pattern_summary_with_explanation("LLM analysis failed")
        else:
            reason = "LLM unavailable" if not self.llm_client else "Rate limit reached"
            embeds = self._create_pattern_summary_with_explanation(reason)
        
        # Clear queue after processing
        self.update_queue.clear()
        self.last_batch_time = datetime.now()
        
        return embeds
    
    async def _can_make_llm_request(self) -> bool:
        """Check if we can make an LLM request without hitting rate limits"""
        stats = self.rate_limiter.get_stats()
        return (stats['requests_this_minute'] < stats['minute_limit'] and 
                stats['requests_this_hour'] < stats['hour_limit'])
    
    async def _create_llm_summary(self) -> List[discord.Embed]:
        """Use Claude to create intelligent summaries with rate limiting"""
        # Wait for rate limit if needed
        await self.rate_limiter.wait_if_needed()
        
        # Prepare update data for LLM
        updates_data = []
        for update in self.update_queue:
            time_str = update.pub_date.strftime('%I:%M %p')
            updates_data.append({
                'time': time_str,
                'title': update.title,
                'description': update.description[:200] if update.description != update.title else ""
            })
        
        # Create LLM prompt (single prompt for all scenarios)
        prompt = self._create_llm_prompt(updates_data)
        
        try:
            # Make LLM request
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.llm_model,
                max_tokens=1200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            analysis = self._parse_llm_response(response.content[0].text)
            
            # Create main embed
            embeds = self._create_embeds_from_analysis(analysis)
            
            # Add highlights embed for batches with 7+ updates if rate limits allow
            if len(self.update_queue) >= 7:
                logger.info(f"Creating highlights embed for {len(self.update_queue)} updates")
                try:
                    # Check if we can make another LLM call
                    if await self._can_make_llm_request():
                        highlights_embed = await self._create_llm_highlights_embed(analysis.get('game_phase', 'current'))
                        if highlights_embed:
                            embeds.append(highlights_embed)
                            logger.info("Added LLM-curated highlights embed")
                        else:
                            logger.warning("LLM highlights embed creation returned None")
                    else:
                        logger.warning("Rate limit reached - creating pattern-based highlights embed")
                        highlights_embed = self._create_pattern_highlights_embed()
                        if highlights_embed:
                            embeds.append(highlights_embed)
                            logger.info("Added pattern-based highlights embed")
                except Exception as e:
                    logger.error(f"Error creating highlights embed: {e}")
                    # Fallback to pattern-based highlights
                    highlights_embed = self._create_pattern_highlights_embed()
                    if highlights_embed:
                        embeds.append(highlights_embed)
                        logger.info("Added fallback highlights embed")
            
            return embeds
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    async def _create_llm_highlights_embed(self, game_phase: str) -> Optional[discord.Embed]:
        """Create a highlights embed using LLM to curate the most important moments"""
        try:
            # Wait for rate limit if needed
            await self.rate_limiter.wait_if_needed()
            
            # Prepare update data
            updates_text = "\n".join([
                f"{self._extract_correct_time(u)} - {u.title}"
                for u in self.update_queue
            ])
            
            prompt = f"""You are a Big Brother superfan curating the MOST IMPORTANT moments from these {len(self.update_queue)} updates.

{updates_text}

Select 5-8 updates that are TRUE HIGHLIGHTS - moments that stand out as particularly important, dramatic, funny, or game-changing. 

HIGHLIGHT-WORTHY updates include:
- Competition wins (HOH, POV, etc.)
- Major strategic moves or betrayals
- Dramatic fights or confrontations  
- Romantic moments (first kiss, breakup, etc.)
- Hilarious or memorable incidents
- Game-changing twists revealed
- Eviction results or surprise votes
- Alliance formations or breaks

NOT highlights (unless part of something bigger):
- Single jury votes (unless it's a crucial swing vote)
- Routine conversations
- Minor game talk
- Regular daily activities

For each selected update, provide:
{{
    "highlights": [
        {{
            "time": "exact time from update",
            "title": "exact title from update BUT REMOVE the time if it appears at the beginning",
            "importance_emoji": "🔥 for high, ⭐ for medium, 📝 for low",
            "reason": "ONLY add this field if the title needs crucial context that isn't obvious. Keep it VERY brief (under 10 words). Most updates won't need this."
        }}
    ]
}}

Be selective - these should be the updates that a superfan would want to know about if they could only see 5-8 things from this time period."""

            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.llm_model,
                max_tokens=800,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            try:
                highlights_data = self._parse_llm_response(response.content[0].text)
                
                if not highlights_data.get('highlights'):
                    logger.warning("No highlights in LLM response")
                    return self._create_pattern_highlights_embed()
                
                embed = discord.Embed(
                    title="🎯 Feed Highlights - What Mattered",
                    description=f"Chen Bot's key moments ({len(highlights_data['highlights'])} of {len(self.update_queue)} updates)",
                    color=0xe74c3c if game_phase == "final_weeks" else 0xf39c12 if game_phase == "jury_phase" else 0x3498db,
                    timestamp=datetime.now()
                )
                
                for highlight in highlights_data['highlights'][:8]:  # Max 8
                    # Clean the title - remove time if it appears at the beginning
                    title = highlight.get('title', 'Update')
                    # Remove time pattern from the beginning of the title
                    title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*-\s*', '', title)
                    
                    # Only add reason if it exists and isn't empty
                    if highlight.get('reason') and highlight['reason'].strip():
                        embed.add_field(
                            name=f"{highlight.get('importance_emoji', '📝')} {highlight.get('time', 'Time')}",
                            value=f"{title}\n*{highlight['reason']}*",
                            inline=False
                        )
                    else:
                        embed.add_field(
                            name=f"{highlight.get('importance_emoji', '📝')} {highlight.get('time', 'Time')}",
                            value=title,
                            inline=False
                        )
                
                embed.set_footer(text=f"Chen Bot's Highlights • {game_phase.replace('_', ' ').title()}")
                
                return embed
                
            except Exception as e:
                logger.error(f"Failed to parse highlights response: {e}")
                return self._create_pattern_highlights_embed()
                
        except Exception as e:
            logger.error(f"LLM highlights creation failed: {e}")
            return self._create_pattern_highlights_embed()
    
    def _create_llm_prompt(self, updates_data: List[Dict]) -> str:
        """Create LLM prompt for all scenarios"""
        updates_text = "\n".join([
            f"{u['time']} - {u['title']}" + (f" ({u['description']})" if u['description'] else "")
            for u in updates_data
        ])
        
        return f"""You are the ultimate Big Brother superfan - part Taran Armstrong's strategic genius, part live feed obsessive who loves ALL aspects of the BB experience.

Analyze these {len(updates_data)} Big Brother live feed updates:

{updates_text}

As a complete Big Brother superfan, provide analysis covering BOTH strategic gameplay AND social dynamics:

{{
    "headline": "Compelling headline that captures the most significant development (strategic OR social)",
    "summary": "3-4 sentence summary balancing strategic implications with social dynamics and entertainment value",
    "strategic_analysis": "Strategic implications - targets, power shifts, competition positioning, voting plans",
    "social_dynamics": "Alliance formations, alliance shifts, trust levels, betrayals, strategic partnerships",
    "entertainment_highlights": "Funny moments, drama, memorable quotes, personality clashes, or unique interactions",
    "key_players": ["houseguests", "involved", "in", "strategic", "and", "social", "moments"],
    "game_phase": "one of: early_game, jury_phase, final_weeks, finale_night",
    "strategic_importance": 7,
    "house_culture": "Inside jokes, daily routines, house traditions, or quirky moments that define this group",
    "relationship_updates": "Showmance developments, romantic connections, dating situations, or relationship changes"
}}

Remember: Big Brother superfans want strategic depth BUT also love the social experiment aspects. Include:
- Strategic gameplay and voting plans
- Alliance dynamics and strategic partnerships
- Entertainment value and memorable moments
- House culture and personality interactions
- Showmance and romantic developments

Focus Social Dynamics on ALLIANCES and strategic relationships, while Relationship Updates covers SHOWMANCES and romantic connections."""
    
    def _parse_llm_response(self, response_text: str) -> dict:
        """Parse LLM response with fallback handling"""
        try:
            # Try to extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_text = response_text[json_start:json_end]
                return json.loads(json_text)
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parsing failed: {e}, using text response")
            return self._parse_text_response(response_text)
    
    def _parse_text_response(self, response_text: str) -> dict:
        """Parse LLM response when JSON parsing fails"""
        # Create a basic analysis from the text response
        analysis = {
            "headline": "Big Brother Update",
            "summary": response_text[:300] + "..." if len(response_text) > 300 else response_text,
            "strategic_analysis": "Analysis available in summary above.",
            "key_players": [],
            "game_phase": "current",
            "strategic_importance": 5,
            "social_dynamics": "See summary for details.",
            "entertainment_highlights": "Various moments occurred."
        }
        
        # Try to extract key information
        try:
            names = re.findall(r'\b[A-Z][a-z]+\b', response_text)
            analysis['key_players'] = [name for name in names if name not in EXCLUDE_WORDS][:5]
        except Exception as e:
            logger.debug(f"Text parsing error: {e}")
        
        return analysis
    
    def _create_embeds_from_analysis(self, analysis: dict) -> List[discord.Embed]:
        """Create Discord embeds from LLM analysis"""
        game_phase_colors = {
            "early_game": 0x3498db,
            "jury_phase": 0xf39c12,
            "final_weeks": 0xe74c3c,
            "finale_night": 0xffd700
        }
        
        color = game_phase_colors.get(analysis.get('game_phase', 'early_game'), 0x3498db)
        
        main_embed = discord.Embed(
            title=f"🎭 {analysis['headline']}",
            description=f"**{len(self.update_queue)} feed updates** • {analysis.get('game_phase', 'Current Phase').replace('_', ' ').title()}\n\n{analysis['summary']}",
            color=color,
            timestamp=datetime.now()
        )
        
        # Add strategic analysis
        if analysis.get('strategic_analysis'):
            main_embed.add_field(
                name="🎯 Strategic Analysis",
                value=analysis['strategic_analysis'],
                inline=False
            )
        
        # Add regular season fields
        if analysis.get('social_dynamics'):
            main_embed.add_field(
                name="🤝 Alliance Dynamics",
                value=analysis['social_dynamics'],
                inline=False
            )
        
        if analysis.get('entertainment_highlights'):
            main_embed.add_field(
                name="🎬 Entertainment Highlights",
                value=analysis['entertainment_highlights'],
                inline=False
            )
        
        if analysis.get('relationship_updates'):
            main_embed.add_field(
                name="💕 Showmance Updates",
                value=analysis['relationship_updates'],
                inline=False
            )
        
        if analysis.get('house_culture'):
            main_embed.add_field(
                name="🏠 House Culture",
                value=analysis['house_culture'],
                inline=False
            )
        
        # Add common fields
        self._add_common_fields(main_embed, analysis)
        
        return [main_embed]
    
    def _add_common_fields(self, embed: discord.Embed, analysis: dict):
        """Add common fields to embed"""
        # Key players
        if analysis.get('key_players'):
            players = analysis['key_players'][:8]
            embed.add_field(
                name="⭐ Key Players",
                value=" • ".join(players),
                inline=False
            )
        
        # Strategic importance
        importance = analysis.get('strategic_importance', 5)
        importance_bar = "🔥" * min(importance, 10)
        embed.add_field(
            name="🎲 Overall Importance",
            value=f"{importance_bar} {importance}/10",
            inline=True
        )
        
        # Footer
        embed.set_footer(text="Chen Bot's Observations")
    
    def _create_pattern_summary_with_explanation(self, reason: str) -> List[discord.Embed]:
        """Enhanced pattern-based summary with explanation"""
        grouped = self._group_updates_pattern()
        
        total_updates = sum(len(updates) for updates in grouped.values())
        headline = self._get_headline_pattern(grouped)
        
        embed = discord.Embed(
            title=f"🎭 {headline}",
            description=f"**{total_updates} updates** from the last batch period\n\n*{reason} - Using pattern analysis*",
            color=self._get_batch_color_pattern(grouped),
            timestamp=datetime.now()
        )
        
        # Add grouped summaries
        for category, updates in grouped.items():
            if updates:
                summary = self._create_category_summary(category, updates)
                embed.add_field(
                    name=f"{self._get_category_emoji(category)} {category.title()} ({len(updates)})",
                    value=summary,
                    inline=False
                )
        
        return [embed]

    def get_rate_limit_stats(self) -> Dict[str, int]:
        """Get current rate limiting statistics"""
        return self.rate_limiter.get_stats()
    
    def _group_updates_pattern(self) -> Dict[str, List[BBUpdate]]:
        """Pattern-based grouping when LLM unavailable"""
        groups = defaultdict(list)
        
        for update in self.update_queue:
            content = f"{update.title} {update.description}".lower()
            
            if 'vote' in content or 'voting' in content:
                groups['votes'].append(update)
            elif any(word in content for word in ['wins', 'won', 'hoh', 'pov', 'veto']):
                groups['competitions'].append(update)
            elif any(word in content for word in ['nominat', 'ceremony']):
                groups['nominations'].append(update)
            else:
                groups['general'].append(update)
        
        return groups
    
    def _get_headline_pattern(self, grouped: Dict[str, List[BBUpdate]]) -> str:
        """Generate headline without LLM"""
        if grouped.get('votes'):
            return "Jury Votes Revealed!"
        elif grouped.get('competitions'):
            return "Competition Results!"
        elif grouped.get('nominations'):
            return "Nomination Ceremony!"
        else:
            return "Big Brother Update"
    
    def _get_batch_color_pattern(self, grouped: Dict[str, List[BBUpdate]]) -> int:
        """Determine color without LLM"""
        if grouped.get('votes'):
            return 0xe74c3c  # Red
        elif grouped.get('competitions'):
            return 0xf39c12  # Orange
        else:
            return 0x3498db  # Blue
    
    def _create_category_summary(self, category: str, updates: List[BBUpdate]) -> str:
        """Create better category summaries"""
        if len(updates) <= 3:
            return "\n".join([
                f"• {u.title[:80]}..." if len(u.title) > 80 else f"• {u.title}"
                for u in updates
            ])
        else:
            return (f"• {updates[0].title[:80]}...\n"
                   f"• {updates[1].title[:80]}...\n"
                   f"• And {len(updates)-2} more updates")
    
    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for category"""
        emoji_map = {
            'votes': '🗳️',
            'competitions': '🏆',
            'nominations': '🎯',
            'general': '📝',
            'strategy': '🧠',
            'drama': '💥'
        }
        return emoji_map.get(category, '📝')
    
    async def create_daily_recap(self, updates: List[BBUpdate], day_number: int) -> List[discord.Embed]:
        """Create a comprehensive daily recap from all updates"""
        if not updates:
            return []
        
        logger.info(f"Creating daily recap for {len(updates)} updates")
        
        # Check if we need to chunk the updates (too many for single LLM call)
        if len(updates) > 50:
            return await self._create_chunked_daily_recap(updates, day_number)
        else:
            return await self._create_single_daily_recap(updates, day_number)
    
    async def _create_single_daily_recap(self, updates: List[BBUpdate], day_number: int) -> List[discord.Embed]:
        """Create daily recap from manageable number of updates"""
        if not self.llm_client or not await self._can_make_llm_request():
            return self._create_pattern_daily_recap(updates, day_number)
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            # Prepare chronological update data
            updates_data = []
            for update in updates:
                time_str = self._extract_correct_time(update)
                updates_data.append({
                    'time': time_str,
                    'title': update.title,
                    'description': update.description[:150] if update.description != update.title else ""
                })
            
            # Create daily recap prompt
            prompt = f"""You are the ultimate Big Brother superfan creating a comprehensive DAILY RECAP for Day {day_number}.

Analyze these {len(updates)} updates from the entire day (chronological order):

{chr(10).join([f"{u['time']} - {u['title']}" + (f" ({u['description']})" if u['description'] else "") for u in updates_data])}

Create a comprehensive daily recap that tells the story of Day {day_number}:

{{
    "headline": "Day {day_number} headline capturing the most significant storyline",
    "summary": "4-5 sentence summary of the day's key developments and overall narrative",
    "strategic_analysis": "Strategic developments - key conversations, alliance shifts, target changes, power dynamics",
    "social_dynamics": "Alliance formations, trust shifts, betrayals, strategic partnerships throughout the day",
    "entertainment_highlights": "Memorable moments, drama, funny interactions, personality conflicts from the day",
    "key_players": ["houseguests", "who", "were", "central", "to", "the", "day"],
    "strategic_importance": 8,
    "house_culture": "Daily routines, inside jokes, traditions, or cultural moments that defined the day",
    "relationship_updates": "Showmance developments, romantic moments, or relationship changes",
    "day_timeline": "Brief chronological overview of how the day unfolded from morning to night"
}}

Focus on creating a comprehensive daily story that captures the full arc of Day {day_number}. This should read like a daily diary entry for superfans."""

            # Make LLM request
            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.llm_model,
                max_tokens=1500,  # Longer for daily recap
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            analysis = self._parse_llm_response(response.content[0].text)
            
            # Create daily recap embed
            return self._create_daily_recap_embed(analysis, day_number, len(updates))
            
        except Exception as e:
            logger.error(f"Daily recap LLM failed: {e}")
            return self._create_pattern_daily_recap(updates, day_number)
    
    async def _create_chunked_daily_recap(self, updates: List[BBUpdate], day_number: int) -> List[discord.Embed]:
        """Create daily recap by summarizing in chunks first"""
        chunk_size = 25
        chunks = [updates[i:i + chunk_size] for i in range(0, len(updates), chunk_size)]
        
        logger.info(f"Creating chunked daily recap: {len(chunks)} chunks of ~{chunk_size} updates each")
        
        # First pass: summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            if not await self._can_make_llm_request():
                # Fall back to pattern analysis if rate limited
                summary = self._summarize_chunk_pattern(chunk, i + 1)
            else:
                summary = await self._summarize_chunk_llm(chunk, i + 1)
            
            chunk_summaries.append(summary)
        
        # Second pass: create final daily recap from summaries
        return await self._create_final_daily_recap(chunk_summaries, day_number, len(updates))
    
    async def _summarize_chunk_llm(self, chunk: List[BBUpdate], chunk_number: int) -> str:
        """Summarize a chunk of updates using LLM"""
        try:
            await self.rate_limiter.wait_if_needed()
            
            updates_text = "\n".join([
                f"{self._extract_correct_time(u)} - {u.title}"
                for u in chunk
            ])
            
            prompt = f"""Summarize this chunk of Big Brother updates (Chunk {chunk_number}):

{updates_text}

Provide a 2-3 sentence summary capturing the key strategic and social developments in this time period. Focus on the most important events and conversations."""

            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.llm_model,
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return f"**Chunk {chunk_number}**: {response.content[0].text}"
            
        except Exception as e:
            logger.error(f"Chunk summarization failed: {e}")
            return self._summarize_chunk_pattern(chunk, chunk_number)
    
    def _summarize_chunk_pattern(self, chunk: List[BBUpdate], chunk_number: int) -> str:
        """Summarize a chunk using pattern matching"""
        # Group by importance and pick top updates
        important_updates = sorted(
            chunk, 
            key=lambda x: self.analyzer.analyze_strategic_importance(x), 
            reverse=True
        )[:3]
        
        titles = [u.title[:60] + "..." if len(u.title) > 60 else u.title for u in important_updates]
        return f"**Chunk {chunk_number}**: {' • '.join(titles)}"
    
    async def _create_final_daily_recap(self, chunk_summaries: List[str], day_number: int, total_updates: int) -> List[discord.Embed]:
        """Create final daily recap from chunk summaries"""
        if not self.llm_client or not await self._can_make_llm_request():
            return self._create_pattern_daily_recap_from_summaries(chunk_summaries, day_number, total_updates)
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            summaries_text = "\n\n".join(chunk_summaries)
            
            prompt = f"""Create a comprehensive Day {day_number} recap from these chunk summaries:

{summaries_text}

Create a daily recap that tells the story of Day {day_number} from these summaries:

{{
    "headline": "Day {day_number} headline capturing the most significant storyline",
    "summary": "4-5 sentence summary of the day's key developments and overall narrative",
    "strategic_analysis": "Strategic developments throughout the day",
    "social_dynamics": "Alliance and relationship dynamics",
    "entertainment_highlights": "Memorable moments and drama",
    "key_players": ["main", "houseguests", "from", "the", "day"],
    "strategic_importance": 8,
    "house_culture": "Daily culture and routine moments",
    "relationship_updates": "Showmance and relationship developments",
    "day_timeline": "How the day unfolded chronologically"
}}

Focus on creating a cohesive daily story from these summaries."""

            response = await asyncio.to_thread(
                self.llm_client.messages.create,
                model=self.llm_model,
                max_tokens=1200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis = self._parse_llm_response(response.content[0].text)
            return self._create_daily_recap_embed(analysis, day_number, total_updates)
            
        except Exception as e:
            logger.error(f"Final daily recap failed: {e}")
            return self._create_pattern_daily_recap_from_summaries(chunk_summaries, day_number, total_updates)
    
    def _create_daily_recap_embed(self, analysis: dict, day_number: int, update_count: int) -> List[discord.Embed]:
        """Create the daily recap embed"""
        embed = discord.Embed(
            title=f"📅 Day {day_number} Recap",
            description=f"**{update_count} updates** • {analysis.get('headline', 'Daily Recap')}\n\n{analysis.get('summary', 'Daily summary not available')}",
            color=0x9b59b6,  # Purple for daily recaps
            timestamp=datetime.now()
        )
        
        # Add timeline if available
        if analysis.get('day_timeline'):
            embed.add_field(
                name="📖 Day Timeline",
                value=analysis['day_timeline'],
                inline=False
            )
        
        # Add strategic analysis
        if analysis.get('strategic_analysis'):
            embed.add_field(
                name="🎯 Strategic Developments",
                value=analysis['strategic_analysis'],
                inline=False
            )
        
        # Add other sections
        if analysis.get('social_dynamics'):
            embed.add_field(
                name="🤝 Alliance Dynamics",
                value=analysis['social_dynamics'],
                inline=False
            )
        
        if analysis.get('entertainment_highlights'):
            embed.add_field(
                name="🎬 Day Highlights",
                value=analysis['entertainment_highlights'],
                inline=False
            )
        
        if analysis.get('relationship_updates'):
            embed.add_field(
                name="💕 Showmance Updates",
                value=analysis['relationship_updates'],
                inline=False
            )
        
        if analysis.get('house_culture'):
            embed.add_field(
                name="🏠 House Culture",
                value=analysis['house_culture'],
                inline=False
            )
        
        # Add key players
        if analysis.get('key_players'):
            players = analysis['key_players'][:8]
            embed.add_field(
                name="⭐ Key Players of the Day",
                value=" • ".join(players),
                inline=False
            )
        
        # Add importance
        importance = analysis.get('strategic_importance', 5)
        importance_bar = "🔥" * min(importance, 10)
        embed.add_field(
            name="📊 Day Importance",
            value=f"{importance_bar} {importance}/10",
            inline=True
        )
        
        embed.set_footer(text=f"Daily Recap • Day {day_number} • BB Superfan AI")
        
        return [embed]
    
    def _create_pattern_daily_recap(self, updates: List[BBUpdate], day_number: int) -> List[discord.Embed]:
        """Create daily recap using pattern matching"""
        # Analyze updates by importance
        important_updates = sorted(
            updates, 
            key=lambda x: self.analyzer.analyze_strategic_importance(x), 
            reverse=True
        )[:10]  # Top 10 most important
        
        # Group by categories
        categories = defaultdict(list)
        for update in important_updates:
            update_categories = self.analyzer.categorize_update(update)
            for category in update_categories:
                categories[category].append(update)
        
        embed = discord.Embed(
            title=f"📅 Day {day_number} Recap",
            description=f"**{len(updates)} updates** • Pattern-based daily summary",
            color=0x9b59b6,
            timestamp=datetime.now()
        )
        
        # Add top moments
        top_moments = []
        for update in important_updates[:5]:
            time_str = self._extract_correct_time(update)
            title = update.title[:80] + "..." if len(update.title) > 80 else update.title
            top_moments.append(f"**{time_str}**: {title}")
        
        if top_moments:
            embed.add_field(
                name="🎯 Top Moments of the Day",
                value="\n".join(top_moments),
                inline=False
            )
        
        # Add categories
        for category, cat_updates in categories.items():
            if cat_updates:
                summary = f"{len(cat_updates)} updates in this category"
                embed.add_field(
                    name=f"{category}",
                    value=summary,
                    inline=True
                )
        
        embed.set_footer(text=f"Daily Recap • Day {day_number} • Pattern Analysis")
        
        return [embed]
    
    def _create_pattern_daily_recap_from_summaries(self, summaries: List[str], day_number: int, total_updates: int) -> List[discord.Embed]:
        """Create pattern-based daily recap from summaries"""
        embed = discord.Embed(
            title=f"📅 Day {day_number} Recap",
            description=f"**{total_updates} updates** • Chunked summary analysis",
            color=0x9b59b6,
            timestamp=datetime.now()
        )
        
        # Add chunk summaries
        summary_text = "\n\n".join(summaries)
        if len(summary_text) > 1000:
            summary_text = summary_text[:997] + "..."
        
        embed.add_field(
            name="📝 Daily Summary",
            value=summary_text,
            inline=False
        )
        
        embed.set_footer(text=f"Daily Recap • Day {day_number} • Chunked Analysis")
        
        return [embed]

    def _extract_correct_time(self, update: BBUpdate) -> str:
        """Extract the correct time from the update content rather than pub_date"""
        # Look for time patterns in the title like "07:56 PM PST"
        time_pattern = r'(\d{1,2}:\d{2})\s*(PM|AM)\s*PST'
        
        # First try the title
        match = re.search(time_pattern, update.title)
        if match:
            time_str = match.group(1)
            ampm = match.group(2)
            return f"{time_str} {ampm}"
        
        # Then try the description
        match = re.search(time_pattern, update.description)
        if match:
            time_str = match.group(1)
            ampm = match.group(2)
            return f"{time_str} {ampm}"
        
        # Fallback to pub_date if no time found in content
        return update.pub_date.strftime("%I:%M %p")

    def _create_pattern_highlights_embed(self) -> Optional[discord.Embed]:
        """Create highlights embed using pattern matching when LLM unavailable"""
        try:
            # Sort updates by importance score
            updates_with_importance = [
                (update, self.analyzer.analyze_strategic_importance(update))
                for update in self.update_queue
            ]
            updates_with_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Select top 5-8 most important updates
            selected_updates = updates_with_importance[:min(8, len(updates_with_importance))]
            
            if not selected_updates:
                return None
            
            embed = discord.Embed(
                title="🎯 Feed Highlights - What Mattered",
                description=f"Key moments from this period ({len(selected_updates)} of {len(self.update_queue)} updates)",
                color=0x95a5a6,
                timestamp=datetime.now()
            )
            
            # Add selected highlights
            for i, (update, importance) in enumerate(selected_updates, 1):
                # Extract correct time from content rather than pub_date
                time_str = self._extract_correct_time(update)
                
                # Clean the title - remove time if it appears at the beginning
                title = update.title
                # Remove time pattern from the beginning of the title
                title = re.sub(r'^\d{1,2}:\d{2}\s*(AM|PM)\s*PST\s*-\s*', '', title)
                
                # Only truncate if extremely long (Discord field limit is 1024 chars)
                if len(title) > 1000:
                    title = title[:997] + "..."
                
                # Add importance indicators
                importance_emoji = "🔥" if importance >= 7 else "⭐" if importance >= 5 else "📝"
                
                embed.add_field(
                    name=f"{importance_emoji} {time_str}",
                    value=title,
                    inline=False
                )
            
            embed.set_footer(text=f"Chen Bot's Pattern Analysis • {len(selected_updates)} key moments selected")
            
            return embed
            
        except Exception as e:
            logger.error(f"Pattern highlights creation failed: {e}")
            return None

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
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
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
    
    def get_daily_updates(self, start_time: datetime, end_time: datetime) -> List[BBUpdate]:
        """Get all updates from a specific 24-hour period"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT title, description, link, pub_date, content_hash, author, importance_score
                FROM updates 
                WHERE pub_date >= ? AND pub_date < ?
                ORDER BY pub_date ASC
            """, (start_time, end_time))
            
            results = cursor.fetchall()
            conn.close()
            
            updates = []
            for row in results:
                update = BBUpdate(*row[:6])  # First 6 fields for BBUpdate
                updates.append(update)
            
            return updates
            
        except Exception as e:
            logger.error(f"Database daily query error: {e}")
            return []
    
    def get_recent_updates(self, hours: int) -> List[BBUpdate]:
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
        
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        super().__init__(command_prefix='!bb', intents=intents)
        
        self.rss_url = "https://rss.jokersupdates.com/ubbthreads/rss/bbusaupdates/rss.php"
        self.db = BBDatabase(self.config.get('database_path', 'bb_updates.db'))
        self.analyzer = BBAnalyzer()
        self.update_batcher = UpdateBatcher(self.analyzer, self.config)
        self.alliance_tracker = AllianceTracker(self.config.get('database_path', 'bb_updates.db'))
        
        # NEW: Initialize prediction system
        self.prediction_system = PredictionSystem(self.config.get('database_path', 'bb_updates.db'))
        
        self.is_shutting_down = False
        self.last_successful_check = datetime.now()
        self.total_updates_processed = 0
        self.consecutive_errors = 0
        
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.remove_command('help')
        self.setup_commands()
    
    def setup_commands(self):
        """Setup all slash commands"""
        
        @self.tree.command(name="status", description="Show bot status and statistics")
        async def status_slash(interaction: discord.Interaction):
            """Show bot status"""
            try:
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
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
                
                queue_size = len(self.update_batcher.update_queue)
                embed.add_field(name="Updates in Queue", value=str(queue_size), inline=True)
                
                llm_status = "✅ Enabled" if self.update_batcher.llm_client else "❌ Disabled"
                embed.add_field(name="LLM Summaries", value=llm_status, inline=True)
                
                # Add cache statistics
                cache_stats = self.update_batcher.processed_hashes_cache.get_stats()
                embed.add_field(
                    name="Hash Cache",
                    value=f"Size: {cache_stats['size']}/{cache_stats['capacity']}\n"
                          f"Hit Rate: {cache_stats['hit_rate']}\n"
                          f"Evictions: {cache_stats['evictions']}",
                    inline=True
                )
                
                # Add rate limiting stats
                rate_stats = self.update_batcher.get_rate_limit_stats()
                embed.add_field(
                    name="Rate Limits",
                    value=f"Minute: {rate_stats['requests_this_minute']}/{rate_stats['minute_limit']}\n"
                          f"Hour: {rate_stats['requests_this_hour']}/{rate_stats['hour_limit']}\n"
                          f"Total: {rate_stats['total_requests']}",
                    inline=True
                )
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error generating status: {e}")
                await interaction.followup.send("Error generating status.", ephemeral=True)

        @self.tree.command(name="summary", description="Get a summary of recent Big Brother updates")
        async def summary_slash(interaction: discord.Interaction, hours: int = 24):
            """Generate a summary of updates"""
            try:
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                
                if hours < 1 or hours > 168:
                    await interaction.response.send_message("Hours must be between 1 and 168", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                updates = self.db.get_recent_updates(hours)
                
                if not updates:
                    await interaction.followup.send(f"No updates found in the last {hours} hours.", ephemeral=True)
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
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                await interaction.followup.send("Error generating summary. Please try again.", ephemeral=True)

        @self.tree.command(name="setchannel", description="Set the channel for Big Brother updates")
        async def setchannel_slash(interaction: discord.Interaction, channel: discord.TextChannel):
            """Set the channel for RSS updates"""
            try:
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                    
                if not channel.permissions_for(interaction.guild.me).send_messages:
                    await interaction.response.send_message(
                        f"I don't have permission to send messages in {channel.mention}", 
                        ephemeral=True
                    )
                    return
                
                self.config.set('update_channel_id', channel.id)
                
                await interaction.response.send_message(
                    f"Update channel set to {channel.mention}", 
                    ephemeral=True
                )
                logger.info(f"Update channel set to {channel.id}")
                
            except Exception as e:
                logger.error(f"Error setting channel: {e}")
                await interaction.response.send_message("Error setting channel. Please try again.", ephemeral=True)

        @self.tree.command(name="commands", description="Show all available commands")
        async def commands_slash(interaction: discord.Interaction):
            """Show available commands"""
            embed = discord.Embed(
                title="Big Brother Bot Commands",
                description="Monitor Jokers Updates RSS feed with intelligent analysis",
                color=0x3498db
            )
            
            commands_list = [
                ("/summary", "Get a summary of recent updates (Admin only)"),
                ("/status", "Show bot status and statistics (Admin only)"),
                ("/setchannel", "Set update channel (Admin only)"),
                ("/commands", "Show this help message"),
                ("/forcebatch", "Force send any queued updates (Admin only)"),
                ("/testllm", "Test LLM connection (Admin only)"),
                ("/sync", "Sync slash commands (Owner only)"),
                ("/alliances", "Show current Big Brother alliances"),
                ("/loyalty", "Show a houseguest's alliance history"),
                ("/betrayals", "Show recent alliance betrayals"),
                ("/removebadalliance", "Remove incorrectly detected alliance (Admin only)"),
                ("/clearalliances", "Clear all alliance data (Owner only)"),
                ("/zing", "Deliver a BB-style zing! (target someone, random, or self-zing)"),
                ("**PREDICTIONS**", "**Prediction Leaderboard Commands**"),
                ("/create_poll", "Create a new prediction poll (Admin only)"),
                ("/vote", "Vote on a prediction poll"),
                ("/close_poll", "Manually close a poll (Admin only)"),
                ("/resolve_poll", "Resolve poll and award points (Admin only)"),
                ("/active_polls", "Show all active prediction polls"),
                ("/poll_status", "Show details about a specific poll"),
                ("/leaderboard", "Show prediction leaderboard"),
                ("/my_predictions", "Show your prediction history")
            ]
            
            for name, description in commands_list:
                embed.add_field(name=name, value=description, inline=False)
            
            embed.set_footer(text="All commands are ephemeral (only you can see the response)")
            
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @self.tree.command(name="forcebatch", description="Force send any queued updates")
        async def forcebatch_slash(interaction: discord.Interaction):
            """Force send batch update"""
            try:
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                queue_size = len(self.update_batcher.update_queue)
                if queue_size == 0:
                    await interaction.followup.send("No updates in queue to send.", ephemeral=True)
                    return
                
                await self.send_batch_update()
                
                await interaction.followup.send(f"Force sent batch of {queue_size} updates!", ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error forcing batch: {e}")
                await interaction.followup.send("Error sending batch.", ephemeral=True)

        @self.tree.command(name="testllm", description="Test LLM connection and functionality")
        async def test_llm_slash(interaction: discord.Interaction):
            """Test LLM integration"""
            try:
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                if not self.update_batcher.llm_client:
                    await interaction.followup.send("❌ LLM client not initialized - check API key", ephemeral=True)
                    return
                
                # Check rate limits
                if not await self.update_batcher._can_make_llm_request():
                    stats = self.update_batcher.get_rate_limit_stats()
                    await interaction.followup.send(
                        f"❌ Rate limit reached\n"
                        f"Minute: {stats['requests_this_minute']}/{stats['minute_limit']}\n"
                        f"Hour: {stats['requests_this_hour']}/{stats['hour_limit']}", 
                        ephemeral=True
                    )
                    return
                
                # Test API call
                await self.update_batcher.rate_limiter.wait_if_needed()
                
                test_response = await asyncio.to_thread(
                    self.update_batcher.llm_client.messages.create,
                    model=self.update_batcher.llm_model,
                    max_tokens=100,
                    messages=[{
                        "role": "user", 
                        "content": "You are a Big Brother superfan. Respond with 'LLM connection successful!' and briefly explain why you love both strategic gameplay and social dynamics in Big Brother."
                    }]
                )
                
                response_text = test_response.content[0].text
                
                embed = discord.Embed(
                    title="✅ LLM Connection Test",
                    description=f"**Model**: {self.update_batcher.llm_model}\n**Response**: {response_text}",
                    color=0x2ecc71,
                    timestamp=datetime.now()
                )
                
                # Add rate limit info
                stats = self.update_batcher.get_rate_limit_stats()
                embed.add_field(
                    name="Rate Limits After Test",
                    value=f"Minute: {stats['requests_this_minute']}/{stats['minute_limit']}\n"
                          f"Hour: {stats['requests_this_hour']}/{stats['hour_limit']}",
                    inline=False
                )
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error testing LLM: {e}")
                await interaction.followup.send(f"❌ LLM test failed: {str(e)}", ephemeral=True)

        @self.tree.command(name="sync", description="Sync slash commands (Owner only)")
        async def sync_slash(interaction: discord.Interaction):
            """Manually sync slash commands"""
            try:
                owner_id = self.config.get('owner_id')
                if not owner_id or interaction.user.id != owner_id:
                    await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                try:
                    synced = await self.tree.sync()
                    await interaction.followup.send(f"✅ Synced {len(synced)} slash commands!", ephemeral=True)
                    logger.info(f"Manually synced {len(synced)} commands")
                except Exception as e:
                    await interaction.followup.send(f"❌ Failed to sync commands: {e}", ephemeral=True)
                    logger.error(f"Manual sync failed: {e}")
                    
            except Exception as e:
                logger.error(f"Error in sync command: {e}")
                await interaction.followup.send("Error syncing commands.", ephemeral=True)

        @self.tree.command(name="alliances", description="Show current Big Brother alliances")
        async def alliances_slash(interaction: discord.Interaction):
            """Show current alliance map"""
            try:
                await interaction.response.defer()  # Removed ephemeral=True to make it public
                
                logger.info(f"Alliance command called by {interaction.user}")
                
                embed = self.alliance_tracker.create_alliance_map_embed()
                
                logger.info(f"Alliance embed created with {len(embed.fields)} fields")
                
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error showing alliances: {e}")
                logger.error(traceback.format_exc())
                await interaction.followup.send("Error generating alliance map.")

        @self.tree.command(name="loyalty", description="Show a houseguest's alliance history")
        async def loyalty_slash(interaction: discord.Interaction, houseguest: str):
            """Show loyalty information for a houseguest"""
            try:
                await interaction.response.defer()  # Removed ephemeral=True to make it public
                
                # Capitalize the name properly
                houseguest = houseguest.strip().title()
                
                embed = self.alliance_tracker.get_houseguest_loyalty_embed(houseguest)
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error showing loyalty: {e}")
                logger.error(traceback.format_exc())
                await interaction.followup.send("Error generating loyalty information.")

        @self.tree.command(name="betrayals", description="Show recent alliance betrayals")
        async def betrayals_slash(interaction: discord.Interaction, days: int = 7):
            """Show recent betrayals"""
            try:
                await interaction.response.defer()  # Removed ephemeral=True to make it public
                
                if days < 1 or days > 30:
                    await interaction.followup.send("Days must be between 1 and 30")
                    return
                
                betrayals = self.alliance_tracker.get_recent_betrayals(days)
                
                embed = discord.Embed(
                    title="💔 Recent Alliance Betrayals",
                    description=f"Betrayals in the last {days} days",
                    color=0xe74c3c,
                    timestamp=datetime.now()
                )
                
                if not betrayals:
                    embed.add_field(
                        name="No Betrayals",
                        value="No alliance betrayals detected in this time period",
                        inline=False
                    )
                else:
                    for i, betrayal in enumerate(betrayals[:10], 1):
                        time_ago = datetime.now() - datetime.fromisoformat(betrayal['timestamp'])
                        hours_ago = int(time_ago.total_seconds() / 3600)
                        time_str = f"{hours_ago}h ago" if hours_ago < 24 else f"{hours_ago//24}d ago"
                        
                        embed.add_field(
                            name=f"⚡ {time_str}",
                            value=betrayal['description'],
                            inline=False
                        )
                
                embed.set_footer(text="Based on live feed updates")
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error showing betrayals: {e}")
                logger.error(traceback.format_exc())
                await interaction.followup.send("Error generating betrayal list.")

        @self.tree.command(name="removebadalliance", description="Remove incorrectly detected alliance (Admin only)")
        async def remove_bad_alliance(interaction: discord.Interaction, alliance_name: str):
            """Remove a bad alliance"""
            try:
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                    
                await interaction.response.defer(ephemeral=True)
                
                # Mark alliance as dissolved
                conn = self.alliance_tracker.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE alliances 
                    SET status = 'dissolved', confidence_level = 0 
                    WHERE name = ?
                """, (alliance_name,))
                
                affected = cursor.rowcount
                conn.commit()
                conn.close()
                
                if affected > 0:
                    await interaction.followup.send(f"✅ Removed alliance: **{alliance_name}**", ephemeral=True)
                    logger.info(f"Removed bad alliance: {alliance_name}")
                else:
                    await interaction.followup.send(f"❌ Alliance '{alliance_name}' not found", ephemeral=True)
                    
            except Exception as e:
                logger.error(f"Error removing alliance: {e}")
                await interaction.followup.send("Error removing alliance", ephemeral=True)

        @self.tree.command(name="clearalliances", description="Clear all alliance data (Owner only)")
        async def clear_alliances(interaction: discord.Interaction):
            """Clear all alliance data - owner only"""
            try:
                owner_id = self.config.get('owner_id')
                if not owner_id or interaction.user.id != owner_id:
                    await interaction.response.send_message("Only the bot owner can use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                conn = self.alliance_tracker.get_connection()
                cursor = conn.cursor()
                
                # Clear all alliance data
                cursor.execute("DELETE FROM alliance_events")
                cursor.execute("DELETE FROM alliance_members")
                cursor.execute("DELETE FROM alliances")
                
                conn.commit()
                conn.close()
                
                await interaction.followup.send("✅ All alliance data has been cleared", ephemeral=True)
                logger.info("Alliance data cleared by owner")
                
            except Exception as e:
                logger.error(f"Error clearing alliances: {e}")
                await interaction.followup.send("Error clearing alliance data", ephemeral=True)

        @self.tree.command(name="zing", description="Deliver a Big Brother style zing!")
        @discord.app_commands.describe(
            zing_type="Choose your zing type",
            target="Only required for targeted zings (leave blank for random/self)"
        )
        @discord.app_commands.choices(zing_type=[
            discord.app_commands.Choice(name="Targeted Zing", value="targeted"),
            discord.app_commands.Choice(name="Random Zing", value="random"),
            discord.app_commands.Choice(name="Self Zing", value="self")
        ])
        async def zing_slash(interaction: discord.Interaction, 
                            zing_type: discord.app_commands.Choice[str],
                            target: discord.Member = None):
            """Deliver a zing to someone"""
            try:
                zing_choice = zing_type.value
                
                # Determine the actual target based on choice
                if zing_choice == "targeted":
                    if not target:
                        await interaction.response.send_message("Please specify a target for a targeted zing!", ephemeral=True)
                        return
                    final_target = target
                elif zing_choice == "random":
                    if target:
                        await interaction.response.send_message("Random zing doesn't need a target - it will pick someone automatically!", ephemeral=True)
                        return
                    # Get all members in the server (excluding bots) - INCLUDE the command user
                    members = [m for m in interaction.guild.members if not m.bot]
                    if not members:
                        await interaction.response.send_message("No valid members to zing!", ephemeral=True)
                        return
                    import random as rand_module
                    final_target = rand_module.choice(members)
                elif zing_choice == "self":
                    if target:
                        await interaction.response.send_message("Self zing doesn't need a target - you're zinging yourself!", ephemeral=True)
                        return
                    # Self-zing
                    final_target = interaction.user
                else:
                    await interaction.response.send_message("Invalid zing type!", ephemeral=True)
                    return
                
                # Select a random zing
                import random as rand_module
                zing = rand_module.choice(ALL_ZINGS)
                
                # Replace {target} with the actual mention
                zing_text = zing.replace("{target}", final_target.mention)
                
                # If the zing contains [name], replace it with a random other member's name
                if "[name]" in zing_text:
                    other_members = [m for m in interaction.guild.members if not m.bot and m.id != final_target.id]
                    if other_members:
                        other_member = rand_module.choice(other_members)
                        zing_text = zing_text.replace("[name]", other_member.display_name)
                    else:
                        # Fallback if no other members
                        zing_text = zing_text.replace("[name]", "someone")
                
                # Create the embed
                embed = discord.Embed(
                    title="🎯 ZING!",
                    description=zing_text,
                    color=0xff1744 if final_target == interaction.user else 0xff9800
                )
                
                # Add Zingbot style footer
                zingbot_phrases = [
                    "ZING! That's what I'm programmed for!",
                    "Another successful zing delivered!",
                    "Zingbot 3000 strikes again!",
                    "I came, I saw, I ZINGED!",
                    "My circuits are buzzing with that zing!",
                    "Zing executed successfully!",
                    "Target acquired and zinged!",
                    "Maximum zing efficiency achieved!"
                ]
                
                embed.set_footer(text=rand_module.choice(zingbot_phrases))
                
                # Add special effects for self-zings
                if final_target == interaction.user:
                    embed.add_field(
                        name="😅 Self-Zing Award",
                        value="You zinged yourself! That takes guts... or poor decision making!",
                        inline=False
                    )
                
                await interaction.response.send_message(embed=embed)
                
                # Log the zing
                logger.info(f"Zing delivered: {interaction.user} zinged {final_target}")
                
            except Exception as e:
                logger.error(f"Error delivering zing: {e}")
                await interaction.response.send_message("Error delivering zing. My circuits must be malfunctioning!", ephemeral=True)

        # NEW: Prediction System Commands
        @self.tree.command(name="create_poll", description="Create a new prediction poll (Admin only)")
        @discord.app_commands.describe(
            category="Poll category",
            title="Poll title",
            description="Poll description",
            hours_open="Hours the poll stays open",
            option1="First option",
            option2="Second option",
            option3="Third option (optional)",
            option4="Fourth option (optional)",
            option5="Fifth option (optional)"
        )
        @discord.app_commands.choices(category=[
            discord.app_commands.Choice(name="Season Winner", value="season_winner"),
            discord.app_commands.Choice(name="Weekly HOH", value="weekly_hoh"),
            discord.app_commands.Choice(name="Weekly Veto", value="weekly_veto"),
            discord.app_commands.Choice(name="Weekly Eviction", value="weekly_eviction"),
            discord.app_commands.Choice(name="Jury Vote", value="jury_vote"),
            discord.app_commands.Choice(name="Special Event", value="special_event")
        ])
        async def create_poll_slash(interaction: discord.Interaction, 
                                   category: discord.app_commands.Choice[str],
                                   title: str, description: str, hours_open: int,
                                   option1: str, option2: str,
                                   option3: str = None, option4: str = None, option5: str = None):
            """Create a new prediction poll"""
            try:
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                
                if hours_open < 1 or hours_open > 168:  # Max 1 week
                    await interaction.response.send_message("Hours must be between 1 and 168 (1 week)", ephemeral=True)
                    return
                
                # Collect options
                options = [option1, option2]
                if option3:
                    options.append(option3)
                if option4:
                    options.append(option4)
                if option5:
                    options.append(option5)
                
                await interaction.response.defer()
                
                poll_id = self.prediction_system.create_poll(
                    category_id=category.value,
                    title=title,
                    description=description,
                    created_by=interaction.user.id,
                    hours_open=hours_open,
                    options=options
                )
                
                if poll_id:
                    category_info = PREDICTION_CATEGORIES[category.value]
                    closes_at = datetime.now() + timedelta(hours=hours_open)
                    
                    embed = discord.Embed(
                        title=f"🗳️ New Prediction Poll Created!",
                        description=f"**{title}**\n{description}",
                        color=0x3498db,
                        timestamp=datetime.now()
                    )
                    
                    embed.add_field(
                        name="📊 Poll Details",
                        value=f"**Category**: {category_info['name']}\n"
                              f"**Points Value**: {category_info['points']}\n"
                              f"**Poll ID**: {poll_id}\n"
                              f"**Closes**: <t:{int(closes_at.timestamp())}:R>",
                        inline=False
                    )
                    
                    options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
                    embed.add_field(
                        name="🔹 Options",
                        value=options_text,
                        inline=False
                    )
                    
                    embed.add_field(
                        name="🗳️ How to Vote",
                        value=f"Use `/vote {poll_id} <option>` to make your prediction!",
                        inline=False
                    )
                    
                    embed.set_footer(text=f"Created by {interaction.user.display_name}")
                    
                    await interaction.followup.send(embed=embed)
                    logger.info(f"Poll created: {poll_id} - {title}")
                    
                else:
                    await interaction.followup.send("Error creating poll. Please try again.", ephemeral=True)
                    
            except Exception as e:
                logger.error(f"Error creating poll: {e}")
                await interaction.followup.send("Error creating poll.", ephemeral=True)

        @self.tree.command(name="vote", description="Vote on a prediction poll")
        @discord.app_commands.describe(
            poll_id="Poll ID to vote on",
            option="Your prediction choice"
        )
        async def vote_slash(interaction: discord.Interaction, poll_id: int, option: str):
            """Vote on a prediction poll"""
            try:
                await interaction.response.defer(ephemeral=True)
                
                result = self.prediction_system.vote_on_poll(poll_id, interaction.user.id, option)
                
                if result['success']:
                    embed = discord.Embed(
                        title="✅ Vote Recorded!",
                        description=result['message'],
                        color=0x2ecc71,
                        timestamp=datetime.now()
                    )
                    
                    # Get poll details for context
                    poll_details = self.prediction_system.get_poll_details(poll_id)
                    if poll_details:
                        embed.add_field(
                            name="📊 Poll Info",
                            value=f"**{poll_details['title']}**\n"
                                  f"Category: {poll_details['category']}\n"
                                  f"Points: {poll_details['points_value']}",
                            inline=False
                        )
                    
                    await interaction.followup.send(embed=embed, ephemeral=True)
                    logger.info(f"Vote recorded: User {interaction.user.id} voted '{option}' on poll {poll_id}")
                    
                else:
                    embed = discord.Embed(
                        title="❌ Vote Failed",
                        description=result['error'],
                        color=0xe74c3c
                    )
                    await interaction.followup.send(embed=embed, ephemeral=True)
                    
            except Exception as e:
                logger.error(f"Error voting: {e}")
                await interaction.followup.send("Error processing vote.", ephemeral=True)

        @self.tree.command(name="close_poll", description="Manually close a poll (Admin only)")
        async def close_poll_slash(interaction: discord.Interaction, poll_id: int):
            """Manually close a poll"""
            try:
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer(ephemeral=True)
                
                success = self.prediction_system.close_poll(poll_id)
                
                if success:
                    await interaction.followup.send(f"✅ Poll {poll_id} has been closed", ephemeral=True)
                    logger.info(f"Poll {poll_id} manually closed by {interaction.user}")
                else:
                    await interaction.followup.send(f"❌ Failed to close poll {poll_id} (not found or already closed)", ephemeral=True)
                    
            except Exception as e:
                logger.error(f"Error closing poll: {e}")
                await interaction.followup.send("Error closing poll.", ephemeral=True)

        @self.tree.command(name="resolve_poll", description="Resolve poll and award points (Admin only)")
        async def resolve_poll_slash(interaction: discord.Interaction, poll_id: int, winning_option: str):
            """Resolve a poll and award points"""
            try:
                if not interaction.user.guild_permissions.administrator:
                    await interaction.response.send_message("You need administrator permissions to use this command.", ephemeral=True)
                    return
                
                await interaction.response.defer()
                
                result = self.prediction_system.resolve_poll(poll_id, winning_option)
                
                if result['success']:
                    embed = discord.Embed(
                        title="🏆 Poll Resolved!",
                        description=result['message'],
                        color=0xf39c12,
                        timestamp=datetime.now()
                    )
                    
                    embed.add_field(
                        name="📊 Results",
                        value=f"**Winners**: {result['winner_count']} users\n"
                              f"**Points Awarded**: {result['points_awarded']} each",
                        inline=False
                    )
                    
                    embed.set_footer(text=f"Resolved by {interaction.user.display_name}")
                    
                    await interaction.followup.send(embed=embed)
                    logger.info(f"Poll {poll_id} resolved: {winning_option} won, {result['winner_count']} winners")
                    
                else:
                    embed = discord.Embed(
                        title="❌ Resolution Failed",
                        description=result['error'],
                        color=0xe74c3c
                    )
                    await interaction.followup.send(embed=embed, ephemeral=True)
                    
            except Exception as e:
                logger.error(f"Error resolving poll: {e}")
                await interaction.followup.send("Error resolving poll.", ephemeral=True)

        @self.tree.command(name="active_polls", description="Show all active prediction polls")
        async def active_polls_slash(interaction: discord.Interaction):
            """Show all active polls"""
            try:
                await interaction.response.defer()
                
                polls = self.prediction_system.get_active_polls()
                
                if not polls:
                    embed = discord.Embed(
                        title="📊 Active Prediction Polls",
                        description="No active polls at the moment.",
                        color=0x95a5a6
                    )
                    await interaction.followup.send(embed=embed)
                    return
                
                embed = discord.Embed(
                    title="📊 Active Prediction Polls",
                    description=f"**{len(polls)} active polls**",
                    color=0x3498db,
                    timestamp=datetime.now()
                )
                
                for poll in polls[:10]:  # Limit to 10 polls
                    closes_timestamp = int(poll['closes_at'].timestamp()) if isinstance(poll['closes_at'], datetime) else int(datetime.fromisoformat(poll['closes_at']).timestamp())
                    
                    embed.add_field(
                        name=f"🗳️ Poll #{poll['poll_id']} - {poll['category']}",
                        value=f"**{poll['title']}**\n"
                              f"{poll['description'][:100]}{'...' if len(poll['description']) > 100 else ''}\n"
                              f"Points: {poll['points_value']} | Votes: {poll['vote_count']}\n"
                              f"Closes: <t:{closes_timestamp}:R>",
                        inline=False
                    )
                
                embed.set_footer(text="Use /vote <poll_id> <option> to make predictions!")
                
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error showing active polls: {e}")
                await interaction.followup.send("Error retrieving active polls.", ephemeral=True)

        @self.tree.command(name="poll_status", description="Show details about a specific poll")
        async def poll_status_slash(interaction: discord.Interaction, poll_id: int):
            """Show detailed poll information"""
            try:
                await interaction.response.defer()
                
                poll_details = self.prediction_system.get_poll_details(poll_id)
                
                if not poll_details:
                    await interaction.followup.send(f"Poll #{poll_id} not found.", ephemeral=True)
                    return
                
                embed = discord.Embed(
                    title=f"📊 Poll #{poll_id} Status",
                    description=f"**{poll_details['title']}**\n{poll_details['description']}",
                    color=0x3498db if poll_details['status'] == 'open' else 0x95a5a6,
                    timestamp=datetime.now()
                )
                
                # Poll info
                status_emoji = "🟢" if poll_details['status'] == 'open' else "🔴" if poll_details['status'] == 'closed' else "✅"
                embed.add_field(
                    name="📋 Poll Info",
                    value=f"**Category**: {poll_details['category']}\n"
                          f"**Status**: {status_emoji} {poll_details['status'].title()}\n"
                          f"**Points**: {poll_details['points_value']}",
                    inline=True
                )
                
                # Timing info
                created_timestamp = int(poll_details['created_at'].timestamp()) if isinstance(poll_details['created_at'], datetime) else int(datetime.fromisoformat(poll_details['created_at']).timestamp())
                closes_timestamp = int(poll_details['closes_at'].timestamp()) if isinstance(poll_details['closes_at'], datetime) else int(datetime.fromisoformat(poll_details['closes_at']).timestamp())
                
                embed.add_field(
                    name="⏰ Timing",
                    value=f"**Created**: <t:{created_timestamp}:R>\n"
                          f"**Closes**: <t:{closes_timestamp}:R>",
                    inline=True
                )
                
                # Options with vote counts
                options_text = []
                total_votes = sum(opt['votes'] for opt in poll_details['options'])
                
                for opt in poll_details['options']:
                    percentage = (opt['votes'] / total_votes * 100) if total_votes > 0 else 0
                    bar_length = int(percentage / 10)  # 10% per bar segment
                    bar = "█" * bar_length + "░" * (10 - bar_length)
                    options_text.append(f"{opt['text']}: {opt['votes']} votes ({percentage:.1f}%)\n{bar}")
                
                embed.add_field(
                    name=f"🔹 Options ({total_votes} total votes)",
                    value="\n\n".join(options_text) if options_text else "No votes yet",
                    inline=False
                )
                
                if poll_details['status'] == 'open':
                    embed.add_field(
                        name="🗳️ How to Vote",
                        value=f"Use `/vote {poll_id} <option>` to make your prediction!",
                        inline=False
                    )
                
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error showing poll status: {e}")
                await interaction.followup.send("Error retrieving poll status.", ephemeral=True)

        @self.tree.command(name="leaderboard", description="Show prediction leaderboard")
        async def leaderboard_slash(interaction: discord.Interaction):
            """Show the prediction leaderboard"""
            try:
                await interaction.response.defer()
                
                leaderboard = self.prediction_system.get_leaderboard(limit=15)
                
                if not leaderboard:
                    embed = discord.Embed(
                        title="🏆 Prediction Leaderboard",
                        description="No predictions have been made yet!",
                        color=0x95a5a6
                    )
                    await interaction.followup.send(embed=embed)
                    return
                
                embed = discord.Embed(
                    title="🏆 Big Brother Prediction Leaderboard",
                    description="Top predictors in the server",
                    color=0xf39c12,
                    timestamp=datetime.now()
                )
                
                # Top 3 get special treatment
                medals = ["🥇", "🥈", "🥉"]
                
                for i, entry in enumerate(leaderboard):
                    try:
                        user = self.get_user(entry['user_id']) or await self.fetch_user(entry['user_id'])
                        username = user.display_name if hasattr(user, 'display_name') else user.name
                    except:
                        username = f"User {entry['user_id']}"
                    
                    medal = medals[i] if i < 3 else f"{i+1}."
                    
                    embed.add_field(
                        name=f"{medal} {username}",
                        value=f"**{entry['total_points']}** points\n"
                              f"{entry['correct_predictions']}/{entry['total_predictions']} correct ({entry['accuracy']}%)",
                        inline=True if i >= 3 else False
                    )
                
                # Add categories info
                categories_text = []
                for cat_id, cat_data in PREDICTION_CATEGORIES.items():
                    categories_text.append(f"{cat_data['name']}: {cat_data['points']} pts")
                
                embed.add_field(
                    name="📊 Point Values",
                    value="\n".join(categories_text),
                    inline=False
                )
                
                embed.set_footer(text="Compete for the title of best BB predictor!")
                
                await interaction.followup.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Error showing leaderboard: {e}")
                await interaction.followup.send("Error retrieving leaderboard.", ephemeral=True)

        @self.tree.command(name="my_predictions", description="Show your prediction history")
        @discord.app_commands.describe(category="Filter by category (optional)")
        @discord.app_commands.choices(category=[
            discord.app_commands.Choice(name="All Categories", value="all"),
            discord.app_commands.Choice(name="Season Winner", value="season_winner"),
            discord.app_commands.Choice(name="Weekly HOH", value="weekly_hoh"),
            discord.app_commands.Choice(name="Weekly Veto", value="weekly_veto"),
            discord.app_commands.Choice(name="Weekly Eviction", value="weekly_eviction"),
            discord.app_commands.Choice(name="Jury Vote", value="jury_vote"),
            discord.app_commands.Choice(name="Special Event", value="special_event")
        ])
        async def my_predictions_slash(interaction: discord.Interaction, category: discord.app_commands.Choice[str] = None):
            """Show user's prediction history"""
            try:
                await interaction.response.defer(ephemeral=True)
                
                category_filter = None if not category or category.value == "all" else category.value
                predictions = self.prediction_system.get_user_predictions(interaction.user.id, category_filter)
                
                if not predictions:
                    embed = discord.Embed(
                        title="📊 Your Predictions",
                        description="You haven't made any predictions yet!" if not category_filter else f"No predictions in {category.name}",
                        color=0x95a5a6
                    )
                    await interaction.followup.send(embed=embed, ephemeral=True)
                    return
                
                # Calculate stats
                total_points = sum(p['points_earned'] for p in predictions)
                correct_predictions = sum(1 for p in predictions if p['points_earned'] > 0 and p['status'] == 'resolved')
                resolved_predictions = sum(1 for p in predictions if p['status'] == 'resolved')
                accuracy = (correct_predictions / resolved_predictions * 100) if resolved_predictions > 0 else 0
                
                embed = discord.Embed(
                    title=f"📊 Your Predictions{' - ' + category.name if category and category.value != 'all' else ''}",
                    description=f"**{total_points}** total points | **{correct_predictions}/{resolved_predictions}** correct ({accuracy:.1f}%)",
                    color=0x3498db,
                    timestamp=datetime.now()
                )
                
                # Show recent predictions
                for prediction in predictions[:10]:  # Limit to 10 most recent
                    status_emoji = "✅" if prediction['points_earned'] > 0 else "❌" if prediction['status'] == 'resolved' else "⏳"
                    
                    predicted_timestamp = int(prediction['predicted_at'].timestamp()) if isinstance(prediction['predicted_at'], datetime) else int(datetime.fromisoformat(prediction['predicted_at']).timestamp())
                    
                    embed.add_field(
                        name=f"{status_emoji} {prediction['category']}",
                        value=f"**{prediction['title']}**\n"
                              f"Predicted: {prediction['prediction']}\n"
                              f"Points: {prediction['points_earned']}\n"
                              f"<t:{predicted_timestamp}:R>",
                        inline=True
                    )
                
                if len(predictions) > 10:
                    embed.add_field(
                        name="📝 Note",
                        value=f"Showing 10 most recent of {len(predictions)} total predictions",
                        inline=False
                    )
                
                await interaction.followup.send(embed=embed, ephemeral=True)
                
            except Exception as e:
                logger.error(f"Error showing user predictions: {e}")
                await interaction.followup.send("Error retrieving your predictions.", ephemeral=True)
        
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
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash command(s)")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")
        
        try:
            self.check_rss_feed.start()
            self.daily_recap_task.start()  # Start daily recap task
            logger.info("RSS feed monitoring and daily recap tasks started")
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
    
    async def filter_duplicates(self, updates: List[BBUpdate]) -> List[BBUpdate]:
        """Filter out duplicate updates"""
        new_updates = []
        
        for update in updates:
            # Check both database and cache
            if not self.db.is_duplicate(update.content_hash):
                if not await self.update_batcher.processed_hashes_cache.contains(update.content_hash):
                    new_updates.append(update)
        
        return new_updates
    
    @tasks.loop(hours=24)
    async def daily_recap_task(self):
        """Daily recap task that runs at 8:01 AM Pacific Time"""
        if self.is_shutting_down:
            return
        
        try:
            # Get current time in Pacific timezone
            pacific_tz = pytz.timezone('US/Pacific')
            now_pacific = datetime.now(pacific_tz)
            
            # Check if it's the right time (8:01 AM Pacific)
            if now_pacific.hour != 8 or now_pacific.minute != 1:
                return
            
            logger.info("Starting daily recap generation")
            
            # Calculate the day period (previous 8:01 AM to current 8:01 AM)
            end_time = now_pacific.replace(tzinfo=None)  # Current time
            start_time = end_time - timedelta(hours=24)  # 24 hours ago
            
            # Get all updates from the day
            daily_updates = self.db.get_daily_updates(start_time, end_time)
            
            if not daily_updates:
                logger.info("No updates found for daily recap")
                return
            
            # Calculate day number (days since season start)
            # For now, we'll use a simple calculation - you might want to adjust this
            season_start = datetime(2025, 7, 1)  # Adjust this date for actual season start
            day_number = (end_time.date() - season_start.date()).days + 1
            
            # Create daily recap
            recap_embeds = await self.update_batcher.create_daily_recap(daily_updates, day_number)
            
            # Send daily recap
            await self.send_daily_recap(recap_embeds)
            
            logger.info(f"Daily recap sent for Day {day_number} with {len(daily_updates)} updates")
            
        except Exception as e:
            logger.error(f"Error in daily recap task: {e}")
            logger.error(traceback.format_exc())
    
    @daily_recap_task.before_loop
    async def before_daily_recap_task(self):
        """Wait for bot to be ready before starting daily recap task"""
        await self.wait_until_ready()
        
        # Calculate time until next 8:01 AM Pacific
        pacific_tz = pytz.timezone('US/Pacific')
        now_pacific = datetime.now(pacific_tz)
        
        # Find next 8:01 AM Pacific
        next_recap_time = now_pacific.replace(hour=8, minute=1, second=0, microsecond=0)
        if now_pacific >= next_recap_time:
            next_recap_time += timedelta(days=1)
        
        # Wait until that time
        wait_seconds = (next_recap_time - now_pacific).total_seconds()
        logger.info(f"Daily recap task will start in {wait_seconds/3600:.1f} hours at {next_recap_time.strftime('%I:%M %p PT')}")
        
        await asyncio.sleep(wait_seconds)
    
    async def send_daily_recap(self, embeds: List[discord.Embed]):
        """Send daily recap to the configured channel"""
        channel_id = self.config.get('update_channel_id')
        if not channel_id:
            logger.warning("Update channel not configured")
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found")
                return
            
            embeds = await self.update_batcher.create_batch_summary()
            
            for embed in embeds[:10]:  # Discord limit
                await channel.send(embed=embed)
            
            logger.info(f"Sent batch update with {len(embeds)} embeds")
            
        except Exception as e:
            logger.error(f"Error sending batch update: {e}")

# Create bot instance
bot = BBDiscordBot()

def main():
    """Main function to run the bot"""
    try:
        bot_token = bot.config.get('bot_token')
        if not bot_token:
            logger.error("No bot token found!")
            return
        
        logger.info("Starting Big Brother Discord Bot with Prediction Leaderboard...")
        bot.run(bot_token, reconnect=True)
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    main()id:
            logger.warning("Update channel not configured for daily recap")
            return
        
        try:
            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Channel {channel_id} not found for daily recap")
                return
            
            # Send all embeds
            for embed in embeds[:5]:  # Limit to 5 embeds max
                await channel.send(embed=embed)
            
            logger.info(f"Daily recap sent with {len(embeds)} embeds")
            
        except Exception as e:
            logger.error(f"Error sending daily recap: {e}")

    @tasks.loop(minutes=2)
    async def check_rss_feed(self):
        """Check RSS feed for new updates with intelligent batching"""
        if self.is_shutting_down:
            return
        
        try:
            feed = feedparser.parse(self.rss_url)
            
            if not feed.entries:
                logger.warning("No entries returned from RSS feed")
                return
            
            updates = self.process_rss_entries(feed.entries)
            new_updates = await self.filter_duplicates(updates)  # Now async
            
            for update in new_updates:
                try:
                    categories = self.analyzer.categorize_update(update)
                    importance = self.analyzer.analyze_strategic_importance(update)
                    
                    self.db.store_update(update, importance, categories)
                    await self.update_batcher.add_update(update)  # Now async
                    
                    # Check for alliance information
                    alliance_events = self.alliance_tracker.analyze_update_for_alliances(update)
                    for event in alliance_events:
                        alliance_id = self.alliance_tracker.process_alliance_event(event)
                        if alliance_id:
                            logger.info(f"Alliance event processed: {event['type'].value}")
                    
                    self.total_updates_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing update: {e}")
                    self.consecutive_errors += 1
            
            if self.update_batcher.should_send_batch():
                await self.send_batch_update()
            
            self.last_successful_check = datetime.now()
            self.consecutive_errors = 0
            
            if new_updates:
                logger.info(f"Added {len(new_updates)} updates to batch queue")
                
        except Exception as e:
            logger.error(f"Error in RSS check: {e}")
            self.consecutive_errors += 1
    
    async def send_batch_update(self):
        """Send intelligent batch summary to Discord"""
        channel_id = self.config.get('update_channel_id')
        if not channel_
