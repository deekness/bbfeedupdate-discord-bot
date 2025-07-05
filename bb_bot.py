def main():
    """Main function to run the bot"""
    try:
        bot = BBDiscordBot()
        # Remove the default help command first
        bot.remove_command('help')
        
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
        
        @bot.command(name='commands')
        async def commands_help(ctx):
            """Show available commands"""
            embed = discord.Embed(
                title="Big Brother Bot Commands",
                description="Monitor Big Brother updates with smart analysis",
                color=0x3498db
            )
            
            commands_list = [
                ("!bbsummary [hours]", "Get a summary of recent updates (default: 24h)"),
                ("!bbhouseguests", "Show all tracked houseguests and stats"),
                ("!bbsearch <query>", "Search for updates containing specific text"),
                ("!bbaddname <n> <aliases>", "Add aliases for a houseguest (Mod only)"),
                ("!bbstatus", "Show bot status and statistics"),
                ("!bbsetchannel #channel", "Set update channel (Admin only)"),
                ("!bbcommands", "Show this help message")
            ]
            
            for name, description in commands_list:
                embed.add_field(name=name, value=description, inline=False)
            
            embed.set_footer(text="Big Brother Bot v2.0 | Smart houseguest tracking enabled")
            
            await ctx.send(embed=embed)
        
        bot_token = bot.config.get('bot_token')
        if not bot_token or bot_token == "":
            logger.error("No bot token found!")
            logger.error("Please set the BOT_TOKEN environment variable or add it to config.json")
            logger.error("Get your bot token from: https://discord.com/developers/applications")
            return
        
        logger.info("Starting Big Brother Discord Bot v2.0...")
        try:
            bot.run(bot_token, reconnect=True)
        except discord.errors.LoginFailure:
            logger.error("Failed to login! The bot token is invalid.")
            logger.error("Please check your BOT_TOKEN and make sure it's correct.")
            logger.error("Get a valid token from: https://discord.com/developers/applications")
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())

if __name__ == "__main__":
    main()
