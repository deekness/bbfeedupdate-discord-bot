import asyncio
import logging
from datetime import datetime, timedelta, timezone

import discord
from discord import app_commands
from discord.ext import tasks

import feedparser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# --- Helper Data Classes ---

class BBUpdate:
    def __init__(self, title, description, link, pub_date, author=None):
        self.title = title
        self.description = description
        self.link = link
        self.pub_date = pub_date
        self.author = author


# --- Analyzer stub for categories and importance ---
class BBAnalyzer:
    def categorize_update(self, update: BBUpdate):
        # Dummy categorization
        return ["General"]

    def analyze_strategic_importance(self, update: BBUpdate):
        # Dummy importance scoring (1 to 10)
        return 5


# --- Simple in-memory DB stub ---
class BBDatabase:
    def __init__(self):
        self.updates = []

    def store_update(self, update: BBUpdate, importance: int, categories: list[str]):
        # Could store importance and categories if desired
        self.updates.append(update)

    def get_recent_updates(self, hours=24):
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [u for u in self.updates if u.pub_date >= cutoff]


# --- The Bot Class ---

class BBDiscordBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)
        self.config = {
            'bot_token': 'YOUR_BOT_TOKEN_HERE',
            'update_channel_id': 123456789012345678,  # Set your channel ID here
            'rss_url': 'https://jokersupdates.com/bigbrother/feeds/livefeedupdates.xml',
            'max_consecutive_errors': 10,
        }

        self.db = BBDatabase()
        self.analyzer = BBAnalyzer()
        self.rss_url = self.config['rss_url']

        self.is_shutting_down = False
        self.consecutive_errors = 0
        self.total_updates_processed = 0

        self.last_successful_check = datetime.now(timezone.utc)

        self.check_rss_feed.start()

    def process_rss_entries(self, entries):
        updates = []
        for entry in entries:
            title = entry.get('title', 'No Title')
            description = entry.get('description', '')
            link = entry.get('link', '')
            pub_date = None
            try:
                pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                pub_date = datetime.now(timezone.utc)
            author = entry.get('author', None)
            updates.append(BBUpdate(title, description, link, pub_date, author))
        return updates

    def filter_duplicates(self, updates):
        # Simple filter: skip if title already in DB (naive)
        existing_titles = {u.title for u in self.db.updates}
        return [u for u in updates if u.title not in existing_titles]

    def create_update_embed(self, update: BBUpdate) -> discord.Embed:
        categories = self.analyzer.categorize_update(update)
        importance = self.analyzer.analyze_strategic_importance(update)

        colors = {
            1: 0x95a5a6,  # gray
            2: 0x3498db,  # blue
            3: 0x2ecc71,  # green
            4: 0xf39c12,  # orange
            5: 0xe74c3c   # red
        }
        color = colors.get(min(importance // 2 + 1, 5), 0x95a5a6)

        # Truncate title to max 256 chars (Discord embed title limit)
        title = update.title if len(update.title) <= 256 else update.title[:253] + "..."

        embed = discord.Embed(
            title=title,
            url=update.link,
            description=update.description[:2048],  # max embed description length
            color=color,
            timestamp=update.pub_date
        )
        embed.set_footer(text=f"Importance: {importance}/10 | Categories: {', '.join(categories)}")
        if update.author:
            embed.set_author(name=update.author)
        return embed

    @tasks.loop(minutes=2)
    async def check_rss_feed(self):
        if self.is_shutting_down:
            logger.info("Shutting down, skipping RSS feed check")
            return

        try:
            feed = feedparser.parse(self.rss_url)
            entries = feed.entries if hasattr(feed, 'entries') else []
            updates = self.process_rss_entries(entries)
            new_updates = self.filter_duplicates(updates)

            logger.info(f"Found {len(entries)} entries, {len(new_updates)} new updates")

            if not new_updates:
                self.consecutive_errors = 0
                return

            channel_id = self.config.get("update_channel_id")
            if not channel_id:
                logger.error("Update channel ID not configured")
                return

            channel = self.get_channel(channel_id)
            if not channel:
                logger.error(f"Could not find channel with ID {channel_id}")
                return

            for update in new_updates:
                categories = self.analyzer.categorize_update(update)
                importance = self.analyzer.analyze_strategic_importance(update)
                self.db.store_update(update, importance, categories)

                embed = self.create_update_embed(update)
                try:
                    await channel.send(embed=embed)
                except Exception as e:
                    logger.error(f"Error sending update embed: {e}")

                self.total_updates_processed += 1

            self.last_successful_check = datetime.now(timezone.utc)
            self.consecutive_errors = 0

        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Error during RSS feed check: {e}")
            if self.consecutive_errors >= self.config.get("max_consecutive_errors", 10):
                logger.critical("Too many consecutive errors, shutting down...")
                await self.close()

    @check_rss_feed.before_loop
    async def before_check_rss_feed(self):
        await self.wait_until_ready()

    # Slash command to get today's summary with ephemeral reply
    @app_commands.command(name="bbsummary", description="Get today's Big Brother updates summary")
    async def bbsummary(self, interaction: discord.Interaction):
        utc_now = datetime.now(timezone.utc)

        # Pacific Time offset handling
        # Assume daylight savings UTC-7, adjust if needed
        pacific_offset = timedelta(hours=-7)
        pacific_now = utc_now + pacific_offset

        if pacific_now.hour < 6 or (pacific_now.hour == 6 and pacific_now.minute < 1):
            # Before 6:01 AM PT, show previous day from 6:01 AM PT
            day_start_pacific = (pacific_now - timedelta(days=1)).replace(hour=6, minute=1, second=0, microsecond=0)
        else:
            day_start_pacific = pacific_now.replace(hour=6, minute=1, second=0, microsecond=0)

        day_start_utc = day_start_pacific - pacific_offset

        updates = self.db.get_recent_updates(hours=48)  # extra window
        filtered_updates = [u for u in updates if day_start_utc <= u.pub_date <= utc_now]

        if not filtered_updates:
            await interaction.response.send_message("No updates found for the current day so far.", ephemeral=True)
            return

        summary_text = ""
        for update in filtered_updates:
            time_str = update.pub_date.astimezone().strftime("%H:%M")
            summary_text += f"[{time_str}] {update.title}\n"
            if len(summary_text) > 1800:
                summary_text += "...\n(Truncated)"
                break

        embed = discord.Embed(
            title="Big Brother Daily Summary",
            description=summary_text,
            color=0x3498db,
            timestamp=utc_now
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def setup_hook(self):
        # Register commands here
        self.tree.add_command(self.bbsummary)
        await self.tree.sync()


async def main():
    bot = BBDiscordBot()
    await bot.start(bot.config.get('bot_token'))


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot interrupted and shutting down")
