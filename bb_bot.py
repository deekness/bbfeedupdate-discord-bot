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
from datetime import datetime, timedelta, timezone, time as dt_time
from typing import List
import logging
from dataclasses import dataclass
import json
import traceback
from pathlib import Path

# Configure logging (unchanged)
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
    # (unchanged from your version)
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

# BBAnalyzer unchanged from your original code...

# BBDatabase unchanged from your original code...

class BBDiscordBot(commands.Bot):
    def __init__(self):
        self.config = Config()
        if not self.config.get('bot_token'):
            logger.error("Bot token not configured! Please set BOT_TOKEN environment variable")
            sys.exit(1)
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!bb', intents=intents, help_command=None)
        self.rss_url = "https://rss.jokersupdates.com/ubbthreads/rss/bbusaupdates/rss.php"
        self.db = BBDatabase(self.config.get('database_path', 'bb_updates.db'))
        self.analyzer = BBAnalyzer()
        self.is_shutting_down = False
        self.last_successful_check = datetime.now(timezone.utc)
        self.total_updates_processed = 0
        self.consecutive_errors = 0
        self.add_listener(self.on_ready)
        self.add_listener(self.on_command_error)
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_shutting_down = True
        asyncio.create_task(self.close())

    async def on_ready(self):
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Bot is in {len(self.guilds)} guilds')
        try:
            self.check_rss_feed.start()
            self.daily_summary_task.start()
            logger.info("Background tasks started")
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")

    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.send("You don't have permission to use this command.")
        elif isinstance(error, commands.CommandNotFound):
            await ctx.send("Command not found. Use /bbcommands for available commands.")
        else:
            logger.error(f"Command error: {error}")
            await ctx.send("An error occurred while processing the command.")

    # The rest of your existing methods (create_content_hash, process_rss_entries, filter_duplicates,
    # create_update_embed, send_update_to_channel, check_rss_feed, etc.) remain unchanged.

    # Here is the added scheduled daily summary task at 6:01 AM Pacific Time:

    @tasks.loop(minutes=1)
    async def daily_summary_task(self):
        """Send a daily summary at 6:01 AM Pacific Time"""
        if self.is_shutting_down:
            return
        try:
            now_utc = datetime.now(timezone.utc)
            pacific_offset = timedelta(hours=-7)  # Assuming PDT (adjust for daylight saving)
            now_pacific = now_utc + pacific_offset
            # Check if time is 6:01 AM (hour=6, minute=1)
            if now_pacific.hour == 6 and now_pacific.minute == 1:
                # Send summary to update channel if set
                channel_id = self.config.get('update_channel_id')
                if channel_id:
                    channel = self.get_channel(channel_id)
                    if channel:
                        logger.info("Sending daily summary at 6:01 AM Pacific Time")
                        await self.send_daily_summary(channel, hours=24)
                    else:
                        logger.warning(f"Update channel {channel_id} not found for daily summary")
                else:
                    logger.warning("Update channel not configured, skipping daily summary")
                # Sleep 61 seconds to avoid multiple sends in same minute
                await asyncio.sleep(61)
        except Exception as e:
            logger.error(f"Error in daily_summary_task: {e}")

    async def send_daily_summary(self, channel, hours: int):
        """Helper to generate and send daily summary embed to a channel"""
        updates = self.db.get_recent_updates(hours)
        if not updates:
            await channel.send(f"No updates found in the last {hours} hours for daily summary.")
            return
        categories = {}
        for update in updates:
            update_categories = self.analyzer.categorize_update(update)
            for category in update_categories:
                categories.setdefault(category, []).append(update)
        embed = discord.Embed(
            title=f"Big Brother Updates Summary ({hours}h)",
            description=f"**{len(updates)} total updates**",
            color=0x3498db,
            timestamp=datetime.now(timezone.utc)
        )
        for category, cat_updates in categories.items():
            top_updates = sorted(
                cat_updates,
                key=lambda x: self.analyzer.analyze_strategic_importance(x),
                reverse=True
            )[:3]
            summary_text = "\n".join(
                [f"• {u.title[:100]}..." if len(u.title) > 100 else f"• {u.title}" for u in top_updates]
            )
            embed.add_field(
                name=f"{category} ({len(cat_updates)} updates)",
                value=summary_text or "No updates",
                inline=False
            )
        await channel.send(embed=embed)

    # Slash commands added to reduce chat clutter:

    async def setup_hook(self):
        # Register slash commands
        @self.tree.command(name="summary", description="Generate a summary of updates from past hours (default 24)")
        async def summary(interaction: discord.Interaction, hours: int = 24):
            if hours < 1 or hours > 168:
                await interaction.response.send_message("Hours must be between 1 and 168", ephemeral=True)
                return
            updates = self.db.get_recent_updates(hours)
            if not updates:
                await interaction.response.send_message(f"No updates found in the last {hours} hours.", ephemeral=True)
                return
            categories = {}
            for update in updates:
                update_categories = self.analyzer.categorize_update(update)
                for category in update_categories:
                    categories.setdefault(category, []).append(update)
            embed = discord.Embed(
                title=f"Big Brother Updates Summary ({hours}h)",
                description=f"**{len(updates)} total updates**",
                color=0x3498db,
                timestamp=datetime.now(timezone.utc)
            )
            for category, cat_updates in categories.items():
                top_updates = sorted(
                    cat_updates,
                    key=lambda x: self.analyzer.analyze_strategic_importance(x),
                    reverse=True
                )[:3]
                summary_text = "\n".join(
                    [f"• {u.title[:100]}..." if len(u.title) > 100 else f"• {u.title}" for u in top_updates]
                )
                embed.add_field(
                    name=f"{category} ({len(cat_updates)} updates)",
                    value=summary_text or "No updates",
                    inline=False
                )
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @self.tree.command(name="setchannel", description="Set the channel for RSS updates (Admin only)")
        async def setchannel(interaction: discord.Interaction, channel: discord.TextChannel):
            if not interaction.user.guild_permissions.administrator:
                await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
                return
            if not channel.permissions_for(interaction.guild.me).send_messages:
                await interaction.response.send_message(f"I don't have permission to send messages in {channel.mention}", ephemeral=True)
                return
            self.config.set('update_channel_id', channel.id)
            await interaction.response.send_message(f"Update channel set to {channel.mention}", ephemeral=True)
            logger.info(f"Update channel set to {channel.id}")

        @self.tree.command(name="status", description="Show bot status")
        async def status(interaction: discord.Interaction):
            embed = discord.Embed(
                title="Big Brother Bot Status",
                color=0x2ecc71 if self.consecutive_errors == 0 else 0xe74c3c,
                timestamp=datetime.now(timezone.utc)
            )
            embed.add_field(name="RSS Feed", value=self.rss_url, inline=False)
            embed.add_field(name="Update Channel",
                            value=f"<#{self.config.get('update_channel_id')}>" if self.config.get('update_channel_id') else "Not set",
                            inline=True)
            embed.add_field(name="Updates Processed", value=str(self.total_updates_processed), inline=True)
            embed.add_field(name="Consecutive Errors", value=str(self.consecutive_errors), inline=True)
            time_since_check = datetime.now(timezone.utc) - self.last_successful_check
            embed.add_field(name="Last RSS Check", value=f"{time_since_check.total_seconds():.0f} seconds ago", inline=True)
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @self.tree.command(name="commands", description="Show bot commands help")
        async def commands_help(interaction: discord.Interaction):
            embed = discord.Embed(
                title="Big Brother Bot Commands",
                description="Monitor Jokers Updates RSS feed with intelligent analysis",
                color=0x3498db
            )
            embed.add_field(name="/summary [hours]", value="Generate summary of updates", inline=False)
            embed.add_field(name="/status", value="Show bot status", inline=False)
            embed.add_field(name="/setchannel [channel]", value="Set update channel (Admin only)", inline=False)
            embed.add_field(name="/commands", value="Show this help message", inline=False)
            await interaction.response.send_message(embed=embed, ephemeral=True)

def main():
    try:
        bot = BBDiscordBot()
        logger.info("Starting Big Brother Discord Bot...")
        bot.run(bot.config.get('bot_token'))
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    main()
