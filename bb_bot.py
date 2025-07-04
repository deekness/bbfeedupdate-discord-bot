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
        """Create enhanced Discord embed with advanced analytics"""
        categories = self.analyzer.categorize_update(update)
        importance = self.analyzer.analyze_strategic_importance(update)
        houseguests = self.analyzer.extract_houseguests(f"{update.title} {update.description}")
        sentiment = self.analyzer.advanced.analyze_sentiment(f"{update.title} {update.description}")
        
        # Alliance detection
        alliances = self.analyzer.advanced.track_alliances(update)
        
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
        
        # Add alliance alerts
        if alliances:
            alliance_text = []
            for alliance in alliances:
                if alliance['type'] == 'new_alliance':
                    alliance_text.append(f"🚨 New Alliance: {', '.join(alliance['members'])}")
            
            if alliance_text:
                embed.add_field(name="Alliance Alert", value="\n".join(alliance_text), inline=False)
        
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
        """Check RSS feed for new updates with advanced analytics"""
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
                    houseguests = self.analyzer.extract_houseguests(f"{update.title} {update.description}")
                    sentiment = self.analyzer.advanced.analyze_sentiment(f"{update.title} {update.description}")
                    
                    # Store with enhanced data
                    self.db.store_update(update, importance, categories, sentiment, houseguests)
                    
                    # Advanced analytics
                    self.analyzer.advanced.track_alliances(update)
                    
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
    
    # ENHANCED COMMANDS
    @commands.command(name='powerrankings')
    async def power_rankings(self, ctx):
        """Show current power rankings"""
        try:
            recent_updates = self.db.get_recent_updates(168)  # Last week
            rankings = self.analyzer.advanced.calculate_power_rankings(recent_updates)
            
            embed = discord.Embed(
                title="🏆 Big Brother Power Rankings",
                description="Weekly power rankings based on strategic positioning",
                color=0xf39c12,
                timestamp=datetime.now()
            )
            
            for i, player in enumerate(rankings[:10], 1):
                rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
                
                stats_text = []
                if player['hoh_wins'] > 0:
                    stats_text.append(f"HOH: {player['hoh_wins']}")
                if player['pov_wins'] > 0:
                    stats_text.append(f"POV: {player['pov_wins']}")
                if player['social_connections'] > 0:
                    stats_text.append(f"Connections: {player['social_connections']}")
                
                stats_str = " | ".join(stats_text) if stats_text else "No major stats"
                
                embed.add_field(
                    name=f"{rank_emoji} {player['name']}",
                    value=f"**Score:** {player['power_ranking']:.1f}\n**Stats:** {stats_str}",
                    inline=True
                )
            
            embed.set_footer(text="Based on strategic mentions, competition wins, and social connections")
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating power rankings: {e}")
            await ctx.send("Error generating power rankings. Please try again.")
    
    @commands.command(name='alliances')
    async def show_alliances(self, ctx):
        """Show detected alliances"""
        try:
            embed = discord.Embed(
                title="🤝 Detected Alliances",
                description="Active alliances based on update analysis",
                color=0x9b59b6,
                timestamp=datetime.now()
            )
            
            alliances = self.analyzer.advanced.alliance_tracker
            
            if not alliances:
                embed.add_field(name="No Alliances", value="No clear alliances detected yet", inline=False)
            else:
                for i, (alliance_key, alliance_data) in enumerate(alliances.items(), 1):
                    members_str = ", ".join(alliance_data.members)
                    strength_bar = "█" * int(alliance_data.strength) + "░" * (10 - int(alliance_data.strength))
                    
                    embed.add_field(
                        name=f"Alliance #{i}",
                        value=f"**Members:** {members_str}\n**Strength:** {strength_bar} ({alliance_data.strength:.1f}/10)\n**Mentions:** {alliance_data.mentions_count}",
                        inline=False
                    )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error showing alliances: {e}")
            await ctx.send("Error retrieving alliance data. Please try again.")
    
    @commands.command(name='evictionpredict')
    async def eviction_prediction(self, ctx):
        """Predict eviction likelihood"""
        try:
            recent_updates = self.db.get_recent_updates(72)  # Last 3 days
            predictions = self.analyzer.advanced.predict_eviction(recent_updates)
            
            embed = discord.Embed(
                title="🎯 Eviction Predictions",
                description="Likelihood of eviction based on recent activity",
                color=0xe74c3c,
                timestamp=datetime.now()
            )
            
            if not predictions:
                embed.add_field(name="No Predictions", value="Not enough data for eviction predictions", inline=False)
            else:
                # Sort by likelihood
                sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                
                for houseguest, likelihood in sorted_predictions[:5]:
                    risk_level = "🔴 High" if likelihood > 0.7 else "🟡 Medium" if likelihood > 0.3 else "🟢 Low"
                    percentage = f"{likelihood * 100:.1f}%"
                    
                    embed.add_field(
                        name=f"{houseguest}",
                        value=f"**Risk:** {risk_level}\n**Likelihood:** {percentage}",
                        inline=True
                    )
            
            embed.set_footer(text="Based on targeting language, campaign activity, and strategic positioning")
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating eviction predictions: {e}")
            await ctx.send("Error generating predictions. Please try again.")
    
    @commands.command(name='sentiment')
    async def house_sentiment(self, ctx, hours: int = 24):
        """Show house sentiment analysis"""
        try:
            recent_updates = self.db.get_recent_updates(hours)
            
            if not recent_updates:
                await ctx.send(f"No updates found in the last {hours} hours.")
                return
            
            # Analyze overall sentiment
            total_positive = 0
            total_negative = 0
            total_neutral = 0
            
            for update in recent_updates:
                sentiment = self.analyzer.advanced.analyze_sentiment(f"{update.title} {update.description}")
                total_positive += sentiment["positive"]
                total_negative += sentiment["negative"]
                total_neutral += sentiment["neutral"]
            
            # Calculate averages
            count = len(recent_updates)
            avg_positive = total_positive / count
            avg_negative = total_negative / count
            avg_neutral = total_neutral / count
            
            # Determine overall mood
            if avg_positive > avg_negative and avg_positive > avg_neutral:
                mood = "😊 Positive"
                color = 0x2ecc71
            elif avg_negative > avg_positive and avg_negative > avg_neutral:
                mood = "😤 Tense"
                color = 0xe74c3c
            else:
                mood = "😐 Neutral"
                color = 0x95a5a6
            
            embed = discord.Embed(
                title="🏠 House Sentiment Analysis",
                description=f"Overall house mood: {mood}",
                color=color,
                timestamp=datetime.now()
            )
            
            embed.add_field(name="😊 Positive", value=f"{avg_positive * 100:.1f}%", inline=True)
            embed.add_field(name="😤 Negative", value=f"{avg_negative * 100:.1f}%", inline=True)
            embed.add_field(name="😐 Neutral", value=f"{avg_neutral * 100:.1f}%", inline=True)
            
            embed.add_field(name="Updates Analyzed", value=str(count), inline=True)
            embed.add_field(name="Time Period", value=f"{hours} hours", inline=True)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating sentiment analysis: {e}")
            await ctx.send("Error generating sentiment analysis. Please try again.")
    
    @commands.command(name='heatmap')
    async def activity_heatmap(self, ctx, hours: int = 24):
        """Show houseguest activity heatmap"""
        try:
            recent_updates = self.db.get_recent_updates(hours)
            activity = self.analyzer.advanced.get_activity_heatmap(recent_updates)
            
            embed = discord.Embed(
                title="🔥 Activity Heatmap",
                description=f"Houseguest mentions in the last {hours} hours",
                color=0xf39c12,
                timestamp=datetime.now()
            )
            
            if not activity:
                embed.add_field(name="No Activity", value="No houseguests mentioned in recent updates", inline=False)
            else:
                # Sort by activity
                sorted_activity = sorted(activity.items(), key=lambda x: x[1], reverse=True)
                
                for houseguest, mentions in sorted_activity[:10]:
                    heat_level = "🔥🔥🔥" if mentions > 10 else "🔥🔥" if mentions > 5 else "🔥" if mentions > 2 else "💤"
                    
                    embed.add_field(
                        name=f"{houseguest}",
                        value=f"{heat_level} {mentions} mentions",
                        inline=True
                    )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating activity heatmap: {e}")
            await ctx.send("Error generating heatmap. Please try again.")
    
    @commands.command(name='relationships')
    async def show_relationships(self, ctx):
        """Show relationship mapping"""
        try:
            relationships = self.analyzer.advanced.get_relationship_map()
            
            embed = discord.Embed(
                title="💫 Relationship Map",
                description="Strongest connections based on co-mentions",
                color=0x3498db,
                timestamp=datetime.now()
            )
            
            if not relationships:
                embed.add_field(name="No Relationships", value="No clear relationships detected yet", inline=False)
            else:
                for houseguest, connections in relationships.items():
                    if connections:  # Only show if they have connections
                        connection_text = []
                        for connected_hg, strength in connections[:3]:  # Top 3 connections
                            connection_text.append(f"{connected_hg} ({strength})")
                        
                        embed.add_field(
                            name=f"{houseguest}",
                            value="\n".join(connection_text),
                            inline=True
                        )
            
            embed.set_footer(text="Numbers indicate how often they're mentioned together")
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error showing relationships: {e}")
            await ctx.send("Error retrieving relationship data. Please try again.")
    
    @commands.command(name='predict')
    async def prediction_game(self, ctx, prediction_type: str = None, target: str = None, *, value: str = None):
        """Make predictions about Big Brother events"""
        if not prediction_type:
            # Show help
            embed = discord.Embed(
                title="🔮 Prediction Game",
                description="Make predictions about Big Brother events",
                color=0x9b59b6
            )
            
            embed.add_field(
                name="How to Use",
                value="!bbpredict [type] [target] [value]",
                inline=False
            )
            
            embed.add_field(
                name="Prediction Types",
                value="• `eviction` - Who will be evicted\n• `hoh` - Who will win HOH\n• `pov` - Who will win POV\n• `finale` - Who will win BB",
                inline=False
            )
            
            embed.add_field(
                name="Examples",
                value="• `!bbpredict eviction Michael`\n• `!bbpredict hoh Sarah`\n• `!bbpredict finale Angela`",
                inline=False
            )
            
            await ctx.send(embed=embed)
            return
        
        if not target or not value:
            await ctx.send("Please provide all parameters. Use `!bbpredict` for help.")
            return
        
        try:
            # Store prediction
            confidence = 0.5  # Default confidence
            self.db.store_prediction(str(ctx.author.id), prediction_type, target, value, confidence)
            
            embed = discord.Embed(
                title="🔮 Prediction Recorded",
                description=f"Your prediction has been saved!",
                color=0x2ecc71,
                timestamp=datetime.now()
            )
            
            embed.add_field(name="Type", value=prediction_type.title(), inline=True)
            embed.add_field(name="Target", value=target, inline=True)
            embed.add_field(name="Prediction", value=value, inline=True)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            await ctx.send("Error storing prediction. Please try again.")
    
    @commands.command(name='mypredictions')
    async def my_predictions(self, ctx):
        """Show your prediction history"""
        try:
            predictions = self.db.get_user_predictions(str(ctx.author.id))
            
            embed = discord.Embed(
                title="🔮 Your Predictions",
                description=f"Prediction history for {ctx.author.display_name}",
                color=0x9b59b6,
                timestamp=datetime.now()
            )
            
            if not predictions:
                embed.add_field(name="No Predictions", value="You haven't made any predictions yet!", inline=False)
            else:
                for prediction in predictions:
                    status = "✅ Correct" if prediction['was_correct'] == 1 else "❌ Wrong" if prediction['was_correct'] == 0 else "⏳ Pending"
                    
                    embed.add_field(
                        name=f"{prediction['type'].title()}: {prediction['target']}",
                        value=f"**Prediction:** {prediction['value']}\n**Status:** {status}",
                        inline=True
                    )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error retrieving predictions: {e}")
            await ctx.send("Error retrieving predictions. Please try again.")
    
    @commands.command(name='blindsides')
    async def detect_blindsides(self, ctx):
        """Detect potential blindsides"""
        try:
            recent_updates = self.db.get_recent_updates(48)  # Last 2 days
            blindsides = self.analyzer.advanced.detect_blindsides(recent_updates)
            
            embed = discord.Embed(
                title="🎭 Blindside Alert",
                description="Potential blindsides detected",
                color=0xe74c3c,
                timestamp=datetime.now()
            )
            
            if not blindsides:
                embed.add_field(name="No Blindsides", value="No potential blindsides detected", inline=False)
            else:
                for blindside in blindsides:
                    confidence_emoji = "🔴" if blindside['confidence'] > 0.8 else "🟡" if blindside['confidence'] > 0.5 else "🟢"
                    
                    embed.add_field(
                        name=f"{confidence_emoji} {blindside['target']}",
                        value=f"**Alert:** {blindside['update']}\n**Confidence:** {blindside['confidence'] * 100:.0f}%",
                        inline=False
                    )
            
            embed.set_footer(text="Based on language suggesting unawareness of targeting")
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error detecting blindsides: {e}")
            await ctx.send("Error detecting blindsides. Please try again.")
    
    # ORIGINAL COMMANDS (Enhanced)
    @commands.command(name='summary')
    async def daily_summary(self, ctx, hours: int = 24):
        """Generate enhanced summary with analytics"""
        try:
            if hours < 1 or hours > 168:
                await ctx.send("Hours must be between 1 and 168")
                return
                
            updates = self.db.get_recent_updates(hours)
            
            if not updates:
                await ctx.send(f"No updates found in the last {hours} hours.")
                return
            
            # Enhanced categorization with analytics
            categories = {}
            sentiment_data = {"positive": 0, "negative": 0, "neutral": 0}
            
            for update in updates:
                # Basic categorization
                update_categories = self.analyzer.categorize_update(update)
                for category in update_categories:
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(update)
                
                # Sentiment analysis
                sentiment = self.analyzer.advanced.analyze_sentiment(f"{update.title} {update.description}")
                sentiment_data["positive"] += sentiment["positive"]
                sentiment_data["negative"] += sentiment["negative"]
                sentiment_data["neutral"] += sentiment["neutral"]
            
            # Calculate overall mood
            avg_sentiment = {k: v / len(updates) for k, v in sentiment_data.items()}
            
            if avg_sentiment["positive"] > avg_sentiment["negative"]:
                mood_emoji = "😊"
                mood_text = "Positive"
            elif avg_sentiment["negative"] > avg_sentiment["positive"]:
                mood_emoji = "😤"
                mood_text = "Tense"
            else:
                mood_emoji = "😐"
                mood_text = "Neutral"
            
            embed = discord.Embed(
                title=f"📊 Big Brother Enhanced Summary ({hours}h)",
                description=f"**{len(updates)} total updates** | **House Mood:** {mood_emoji} {mood_text}",
                color=0x3498db,
                timestamp=datetime.now()
            )
            
            # Add categories with enhanced info
            for category, cat_updates in categories.items():
                top_updates = sorted(cat_updates, 
                                   key=lambda x: self.analyzer.analyze_strategic_importance(x), 
                                   reverse=True)[:3]
                
                summary_text = "\n".join([f"• {update.title[:80]}..." 
                                        if len(update.title) > 80 
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
        """Show enhanced bot status"""
        try:
            embed = discord.Embed(
                title="🤖 Enhanced Big Brother Bot Status",
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
            
            # Enhanced features status
            embed.add_field(name="🧠 AI Features", value="✅ Active", inline=True)
            embed.add_field(name="📊 Analytics", value="✅ Tracking", inline=True)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error generating status: {e}")
            await ctx.send("Error generating status.")
    
    # TEST COMMANDS (Updated)
    @commands.command(name='testanalyzer')
    async def test_analyzer(self, ctx):
        """Test the enhanced AI analyzer"""
        
        sample_updates = [
            {
                "title": "HOH Competition Results - Week 3",
                "description": "Sarah wins Head of Household competition after intense endurance challenge. She's already talking about backdooring Michael who has been getting too close to everyone. The house is buzzing with speculation about nominations.",
                "author": "BBUpdater1"
            },
            {
                "title": "Secret Alliance Formation",
                "description": "Late night meeting between Sarah, Jessica, and David in the backyard. They're forming a tight alliance called 'The Trio' and planning to control the game. Michael and Lisa are completely unaware of this new power structure.",
                "author": "StrategyWatcher"
            },
            {
                "title": "Showmance Drama",
                "description": "Jake and Amanda had a heated argument about their relationship. Amanda feels Jake is being too clingy while Jake thinks Amanda is being distant. Other houseguests are getting annoyed by their constant drama.",
                "author": "DramaAlert"
            }
        ]
        
        embed = discord.Embed(
            title="🧪 Enhanced AI Analyzer Test",
            description="Testing advanced Big Brother analytics",
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
            
            # Basic analysis
            categories = self.analyzer.categorize_update(mock_update)
            importance = self.analyzer.analyze_strategic_importance(mock_update)
            houseguests = self.analyzer.extract_houseguests(f"{mock_update.title} {mock_update.description}")
            
            # Advanced analysis
            sentiment = self.analyzer.advanced.analyze_sentiment(f"{mock_update.title} {mock_update.description}")
            alliances = self.analyzer.advanced.track_alliances(mock_update)
            
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
            
            if alliances:
                for alliance in alliances:
                    if alliance['type'] == 'new_alliance':
                        analysis.append(f"**🚨 Alliance:** {', '.join(alliance['members'])}")
            
            embed.add_field(
                name=f"#{i}: {sample['title'][:40]}...",
                value="\n".join(analysis),
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    @commands.command(name='testembeds')
    async def test_embeds(self, ctx):
        """Test enhanced Discord embeds"""
        
        strategic_update = BBUpdate(
            title="BREAKING: Major Alliance Forms to Target Comp Beast",
            description="Sarah, Jessica, and David secretly meet in storage room to form 'The Trio' alliance. They're planning to backdoor Michael next week because he's won too many competitions. The house is completely unaware of this new power structure. Michael thinks he's safe but he's actually the biggest target.",
            link="https://jokersupdates.com/strategic-update",
            pub_date=datetime.now(),
            content_hash="strategic_test",
            author="StrategyMaster"
        )
        
        strategic_embed = self.create_update_embed(strategic_update)
        await ctx.send("**🎯 Enhanced Strategic Update Example:**", embed=strategic_embed)
        
        await asyncio.sleep(2)
        
        drama_update = BBUpdate(
            title="Kitchen Confrontation Turns Ugly",
            description="Lisa and Michael's argument about dishes escalated into personal attacks. Lisa called Michael lazy and entitled while Michael fired back saying Lisa is controlling and dramatic. The whole house is now taking sides and the tension is unbearable. This could split existing alliances.",
            link="https://jokersupdates.com/drama-update",
            pub_date=datetime.now(),
            content_hash="drama_test",
            author="DramaDetector"
        )
        
        drama_embed = self.create_update_embed(drama_update)
        await ctx.send("**💥 Enhanced Drama Update Example:**", embed=drama_embed)
    
    @commands.command(name='testhelp')
    async def test_help(self, ctx):
        """Show all available test commands"""
        
        embed = discord.Embed(
            title="🧪 Enhanced Big Brother Bot Test Commands",
            description="Test all the advanced AI features",
            color=0x9b59b6
        )
        
        embed.add_field(
            name="**Basic Tests**",
            value="• `!bbtestanalyzer` - Test enhanced AI analysis\n• `!bbtestembeds` - Test enhanced Discord embeds",
            inline=False
        )
        
        embed.add_field(
            name="**Analytics Tests**",
            value="• `!bbpowerrankings` - Show power rankings\n• `!bballiances` - Show detected alliances\n• `!bbsentiment` - Show house mood analysis",
            inline=False
        )
        
        embed.add_field(
            name="**Prediction Tests**",
            value="• `!bbpredict` - Make predictions\n• `!bbmypredictions` - View your predictions\n• `!bbevictionpredict` - Eviction predictions",
            inline=False
        )
        
        embed.add_field(
            name="**Advanced Features**",
            value="• `!bbheatmap` - Activity heatmap\n• `!bbrelationships` - Relationship mapping\n• `!bbblindisides` - Blindside detection",
            inline=False
        )
        
        embed.set_footer(text="All features work with or without live Big Brother updates!")
        
        await ctx.send(embed=embed)

def main():
    """Main function to run the enhanced bot"""
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
            """Show all available commands"""
            embed = discord.Embed(
                title="🏠 Enhanced Big Brother Bot Commands",
                description="AI-powered Big Brother analysis with advanced features",
                color=0x3498db
            )
            
            embed.add_field(
                name="**📊 Main Commands**",
                value="• `!bbsummary [hours]` - Enhanced summary with sentiment\n• `!bbstatus` - Bot status and health\n• `!bbsetchannel [ID]` - Set update channel (Admin)",
                inline=False
            )
            
            embed.add_field(
                name="**🏆 Analytics Commands**",
                value="• `!bbpowerrankings` - Weekly power rankings\n• `!bballiances` - Detected alliances\n• `!bbsentiment [hours]` - House mood analysis\n• `!bbheatmap [hours]` - Activity heatmap",
                inline=False
            )
            
            embed.add_field(
                name="**🔮 Prediction Commands**",
                value="• `!bbpredict [type] [target] [value]` - Make predictions\n• `!bbmypredictions` - Your prediction history\n• `!bbevictionpredict` - Eviction likelihood",
                inline=False
            )
            
            embed.add_field(
                name="**🎯 Advanced Features**",
                value="• `!bbrelationships` - Relationship mapping\n• `!bbblindisides` - Blindside detection\n• `!bbtesthelp` - Test all features",
                inline=False
            )
            
            embed.set_footer(text="This bot uses advanced AI to analyze Big Brother strategy, drama, and relationships!")
            
            await ctx.send(embed=embed)
        
        bot_token = bot.config.get('bot_token')
        if not bot_token:
            logger.error("No bot token found!")
            return
        
        logger.info("Starting Enhanced Big Brother Discord Bot...")
        bot.run(bot_token, reconnect=True)
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    main()import discord
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
from typing import List, Dict, Set, Tuple
import logging
from dataclasses import dataclass
import json
import time
import traceback
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict, Counter
import statistics

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
            "max_consecutive_errors": 10,
            "current_season": 27,
            "bb27_houseguests": [
                "Angela", "Tucker", "Makensy", "Cam", "Chelsie", 
                "Rubina", "Kimo", "Leah", "Quinn", "Joseph",
                "T'Kor", "Cedric", "Brooklyn", "Kenney", "Lisa"
            ]
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
            "max_consecutive_errors": int(os.getenv('MAX_CONSECUTIVE_ERRORS', '10')),
            "current_season": int(os.getenv('CURRENT_SEASON', '27')),
            "bb27_houseguests": [
                "Angela", "Tucker", "Makensy", "Cam", "Chelsie", 
                "Rubina", "Kimo", "Leah", "Quinn", "Joseph",
                "T'Kor", "Cedric", "Brooklyn", "Kenney", "Lisa"
            ]
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

@dataclass
class AllianceData:
    """Represents alliance information"""
    members: List[str]
    strength: float
    first_detected: datetime
    last_mentioned: datetime
    mentions_count: int

@dataclass
class HouseguestStats:
    """Represents houseguest statistics"""
    name: str
    mentions_count: int
    hoh_wins: int
    pov_wins: int
    nominations: int
    strategy_mentions: int
    drama_mentions: int
    romance_mentions: int
    target_level: float
    social_connections: List[str]
    power_ranking: float

class AdvancedAnalyzer:
    """Advanced Big Brother analytics and predictions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.houseguests = config.get('bb27_houseguests', [])
        self.sentiment_positive = [
            'happy', 'excited', 'love', 'amazing', 'great', 'wonderful', 
            'perfect', 'fantastic', 'awesome', 'good', 'fun', 'laugh'
        ]
        self.sentiment_negative = [
            'angry', 'hate', 'mad', 'upset', 'annoyed', 'frustrated',
            'terrible', 'awful', 'bad', 'worst', 'annoying', 'drama'
        ]
        self.blindside_keywords = [
            'doesn\'t know', 'no idea', 'thinks they\'re safe', 'blindside',
            'surprise', 'shock', 'unsuspecting', 'unaware'
        ]
        
        # Initialize tracking dictionaries
        self.alliance_tracker = {}
        self.houseguest_stats = {name: HouseguestStats(
            name=name, mentions_count=0, hoh_wins=0, pov_wins=0,
            nominations=0, strategy_mentions=0, drama_mentions=0,
            romance_mentions=0, target_level=0.0, social_connections=[],
            power_ranking=0.0
        ) for name in self.houseguests}
        
        self.relationship_matrix = defaultdict(lambda: defaultdict(int))
        self.voting_patterns = defaultdict(list)
        self.predictions = {}
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of house updates"""
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
    
    def track_alliances(self, update: BBUpdate) -> List[Dict]:
        """Detect and track alliance formations"""
        content = f"{update.title} {update.description}".lower()
        
        # Look for alliance keywords
        alliance_keywords = ['alliance', 'group', 'team', 'together', 'meet', 'plan']
        if not any(keyword in content for keyword in alliance_keywords):
            return []
        
        # Extract mentioned houseguests
        mentioned_houseguests = []
        for houseguest in self.houseguests:
            if houseguest.lower() in content:
                mentioned_houseguests.append(houseguest)
        
        if len(mentioned_houseguests) < 2:
            return []
        
        # Track relationships
        for i, hg1 in enumerate(mentioned_houseguests):
            for hg2 in mentioned_houseguests[i+1:]:
                self.relationship_matrix[hg1][hg2] += 1
                self.relationship_matrix[hg2][hg1] += 1
        
        # Detect new alliances
        new_alliances = []
        if len(mentioned_houseguests) >= 3:
            alliance_key = tuple(sorted(mentioned_houseguests))
            
            if alliance_key not in self.alliance_tracker:
                self.alliance_tracker[alliance_key] = AllianceData(
                    members=mentioned_houseguests,
                    strength=1.0,
                    first_detected=update.pub_date,
                    last_mentioned=update.pub_date,
                    mentions_count=1
                )
                new_alliances.append({
                    'members': mentioned_houseguests,
                    'type': 'new_alliance',
                    'strength': 1.0
                })
            else:
                # Update existing alliance
                alliance = self.alliance_tracker[alliance_key]
                alliance.last_mentioned = update.pub_date
                alliance.mentions_count += 1
                alliance.strength = min(10.0, alliance.strength + 0.5)
        
        return new_alliances
    
    def predict_eviction(self, recent_updates: List[BBUpdate]) -> Dict[str, float]:
        """Predict eviction likelihood based on recent updates"""
        eviction_indicators = {}
        
        for update in recent_updates:
            content = f"{update.title} {update.description}".lower()
            
            # Look for campaign activity
            campaign_keywords = ['campaign', 'vote', 'evict', 'target', 'backdoor', 'pawn']
            if any(keyword in content for keyword in campaign_keywords):
                for houseguest in self.houseguests:
                    if houseguest.lower() in content:
                        if houseguest not in eviction_indicators:
                            eviction_indicators[houseguest] = 0.0
                        
                        # Increase risk for targets
                        if any(word in content for word in ['target', 'backdoor', 'evict']):
                            eviction_indicators[houseguest] += 2.0
                        elif 'campaign' in content:
                            eviction_indicators[houseguest] += 1.0
        
        # Normalize to probabilities
        if eviction_indicators:
            max_score = max(eviction_indicators.values())
            if max_score > 0:
                for houseguest in eviction_indicators:
                    eviction_indicators[houseguest] = min(1.0, eviction_indicators[houseguest] / max_score)
        
        return eviction_indicators
    
    def track_competition_performance(self, updates: List[BBUpdate]) -> Dict[str, Dict]:
        """Track competition wins and identify comp beasts"""
        comp_stats = defaultdict(lambda: {'hoh': 0, 'pov': 0, 'nominations': 0})
        
        for update in updates:
            content = f"{update.title} {update.description}".lower()
            
            # Track HOH wins
            if 'hoh' in content or 'head of household' in content:
                if 'wins' in content or 'winner' in content:
                    for houseguest in self.houseguests:
                        if houseguest.lower() in content:
                            comp_stats[houseguest]['hoh'] += 1
                            self.houseguest_stats[houseguest].hoh_wins += 1
            
            # Track POV wins
            if 'pov' in content or 'power of veto' in content:
                if 'wins' in content or 'winner' in content:
                    for houseguest in self.houseguests:
                        if houseguest.lower() in content:
                            comp_stats[houseguest]['pov'] += 1
                            self.houseguest_stats[houseguest].pov_wins += 1
            
            # Track nominations
            if 'nomination' in content or 'nominate' in content:
                for houseguest in self.houseguests:
                    if houseguest.lower() in content:
                        comp_stats[houseguest]['nominations'] += 1
                        self.houseguest_stats[houseguest].nominations += 1
        
        # Identify comp beasts (3+ comp wins)
        comp_beasts = []
        for houseguest, stats in comp_stats.items():
            total_wins = stats['hoh'] + stats['pov']
            if total_wins >= 3:
                comp_beasts.append({
                    'name': houseguest,
                    'total_wins': total_wins,
                    'hoh_wins': stats['hoh'],
                    'pov_wins': stats['pov'],
                    'threat_level': min(10.0, total_wins * 1.5)
                })
        
        return {'stats': dict(comp_stats), 'comp_beasts': comp_beasts}
    
    def detect_blindsides(self, updates: List[BBUpdate]) -> List[Dict]:
        """Detect potential blindsides"""
        blindsides = []
        
        for update in updates:
            content = f"{update.title} {update.description}".lower()
            
            if any(keyword in content for keyword in self.blindside_keywords):
                for houseguest in self.houseguests:
                    if houseguest.lower() in content:
                        blindsides.append({
                            'target': houseguest,
                            'update': update.title,
                            'confidence': 0.7,
                            'timestamp': update.pub_date
                        })
        
        return blindsides
    
    def calculate_power_rankings(self, updates: List[BBUpdate]) -> List[Dict]:
        """Calculate weekly power rankings"""
        # Reset power rankings
        for houseguest in self.houseguests:
            self.houseguest_stats[houseguest].power_ranking = 0.0
        
        # Analyze recent updates
        for update in updates:
            content = f"{update.title} {update.description}".lower()
            
            for houseguest in self.houseguests:
                if houseguest.lower() in content:
                    stats = self.houseguest_stats[houseguest]
                    
                    # Strategic positioning
                    if any(word in content for word in ['strategy', 'plan', 'alliance']):
                        stats.power_ranking += 2.0
                        stats.strategy_mentions += 1
                    
                    # Competition wins
                    if 'wins' in content and ('hoh' in content or 'pov' in content):
                        stats.power_ranking += 3.0
                    
                    # Being a target (negative)
                    if any(word in content for word in ['target', 'backdoor', 'evict']):
                        stats.power_ranking -= 2.0
                        stats.target_level += 1.0
                    
                    # Social connections
                    mentioned_with = [hg for hg in self.houseguests if hg.lower() in content and hg != houseguest]
                    for connection in mentioned_with:
                        if connection not in stats.social_connections:
                            stats.social_connections.append(connection)
                            stats.power_ranking += 0.5
        
        # Sort by power ranking
        rankings = []
        for houseguest in self.houseguests:
            stats = self.houseguest_stats[houseguest]
            rankings.append({
                'name': houseguest,
                'power_ranking': stats.power_ranking,
                'hoh_wins': stats.hoh_wins,
                'pov_wins': stats.pov_wins,
                'strategy_mentions': stats.strategy_mentions,
                'social_connections': len(stats.social_connections),
                'target_level': stats.target_level
            })
        
        return sorted(rankings, key=lambda x: x['power_ranking'], reverse=True)
    
    def get_activity_heatmap(self, updates: List[BBUpdate]) -> Dict[str, int]:
        """Generate activity heatmap for houseguests"""
        activity = defaultdict(int)
        
        for update in updates:
            content = f"{update.title} {update.description}".lower()
            for houseguest in self.houseguests:
                if houseguest.lower() in content:
                    activity[houseguest] += 1
        
        return dict(activity)
    
    def get_relationship_map(self) -> Dict[str, List[Tuple[str, int]]]:
        """Get relationship connections between houseguests"""
        relationships = {}
        
        for hg1 in self.houseguests:
            connections = []
            for hg2, count in self.relationship_matrix[hg1].items():
                if count > 0:
                    connections.append((hg2, count))
            
            # Sort by connection strength
            connections.sort(key=lambda x: x[1], reverse=True)
            relationships[hg1] = connections[:5]  # Top 5 connections
        
        return relationships

class BBAnalyzer:
    """Analyzes Big Brother updates for strategic insights"""
    
    def __init__(self, config: Config):
        self.config = config
        self.advanced = AdvancedAnalyzer(config)
        
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
        """Extract houseguest names from text using known houseguest list"""
        mentioned_houseguests = []
        text_lower = text.lower()
        
        for houseguest in self.config.get('bb27_houseguests', []):
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

class BBDatabase:
    """Enhanced database with advanced analytics storage"""
    
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
        """Initialize the database schema with advanced analytics tables"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Main updates table
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
                    sentiment_positive REAL DEFAULT 0.0,
                    sentiment_negative REAL DEFAULT 0.0,
                    sentiment_neutral REAL DEFAULT 1.0,
                    mentioned_houseguests TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Alliance tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alliances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    members TEXT,
                    strength REAL,
                    first_detected TIMESTAMP,
                    last_mentioned TIMESTAMP,
                    mentions_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Houseguest statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS houseguest_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    mentions_count INTEGER DEFAULT 0,
                    hoh_wins INTEGER DEFAULT 0,
                    pov_wins INTEGER DEFAULT 0,
                    nominations INTEGER DEFAULT 0,
                    strategy_mentions INTEGER DEFAULT 0,
                    drama_mentions INTEGER DEFAULT 0,
                    romance_mentions INTEGER DEFAULT 0,
                    target_level REAL DEFAULT 0.0,
                    social_connections TEXT,
                    power_ranking REAL DEFAULT 0.0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    prediction_type TEXT,
                    prediction_target TEXT,
                    prediction_value TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    was_correct BOOLEAN
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON updates(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pub_date ON updates(pub_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mentioned_houseguests ON updates(mentioned_houseguests)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id)")
            
            conn.commit()
            logger.info("Enhanced database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
        finally:
            conn.close()
    
    def store_update(self, update: BBUpdate, importance_score: int = 1, categories: List[str] = None, 
                    sentiment: Dict[str, float] = None, mentioned_houseguests: List[str] = None):
        """Store an update with enhanced analytics data"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            categories_str = ",".join(categories) if categories else ""
            houseguests_str = ",".join(mentioned_houseguests) if mentioned_houseguests else ""
            
            sentiment = sentiment or {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            
            cursor.execute("""
                INSERT INTO updates (content_hash, title, description, link, pub_date, author, 
                                   importance_score, categories, sentiment_positive, sentiment_negative, 
                                   sentiment_neutral, mentioned_houseguests)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (update.content_hash, update.title, update.description, update.link, 
                  update.pub_date, update.author, importance_score, categories_str,
                  sentiment["positive"], sentiment["negative"], sentiment["neutral"],
                  houseguests_str))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database store error: {e}")
            raise
    
    def store_prediction(self, user_id: str, prediction_type: str, target: str, value: str, confidence: float):
        """Store user prediction"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions (user_id, prediction_type, prediction_target, prediction_value, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, prediction_type, target, value, confidence))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
    
    def get_user_predictions(self, user_id: str) -> List[Dict]:
        """Get user's prediction history"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT prediction_type, prediction_target, prediction_value, confidence, 
                       created_at, was_correct
                FROM predictions 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 10
            """, (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            predictions = []
            for row in results:
                predictions.append({
                    'type': row[0],
                    'target': row[1],
                    'value': row[2],
                    'confidence': row[3],
                    'created_at': row[4],
                    'was_correct': row[5]
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []
    
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

class BBDiscordBot(commands.Bot):
    """Enhanced Discord bot with advanced analytics"""
    
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
        self.analyzer = BBAnalyzer(self.config)
        
        self.is_shutting_down = False
        self.last_successful_check = datetime.now()
        self.total_updates_processed = 0
        self.consecutive_errors = 0
        
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal
