# 🤖 Notion Automation Pro++ 

An enhanced autonomous agent that integrates Notion with multiple platforms (Slack, Gmail, Trello) using OpenAI to automatically process and organize content.

## ✨ Features

- 🤖 **Autonomous Monitoring**: Continuously monitors Slack, Gmail, and Trello for new content
- 📝 **AI-Powered Processing**: Uses OpenAI GPT to generate structured Notion pages
- 📊 **Visual Dashboard**: Real-time progress tracking and statistics
- 👥 **User Assignment**: Automatic user assignment based on content source
- 🛡️ **Robust Error Handling**: Comprehensive retry logic and error recovery
- ⚡ **Rate Limiting**: Built-in API rate limiting to prevent quota exhaustion
- 📈 **Detailed Logging**: Comprehensive logging with colored console output
- 🔧 **Easy Configuration**: Environment-based configuration management

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd notion-automation-pro

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create your environment configuration:

```bash
# Generate environment template
python -m notion_automation_pro.main --create-env

# Copy and edit the template
cp .env.template .env
# Edit .env with your API credentials
```

### 3. Required API Credentials

You'll need to obtain API credentials for:

- **Notion**: [Create integration](https://developers.notion.com/docs/create-a-notion-integration)
- **OpenAI**: [Get API key](https://platform.openai.com/api-keys)
- **Slack**: [Create app](https://api.slack.com/apps)
- **Gmail**: [Enable Gmail API](https://developers.google.com/gmail/api/quickstart/python)
- **Trello**: [Get API credentials](https://trello.com/app-key)

### 4. Environment Variables

```bash
# Notion Configuration
NOTION_TOKEN=your_notion_token_here
NOTION_DATABASE_IDS=database_id_1,database_id_2
NOTION_DASHBOARD_PAGE_ID=your_dashboard_page_id

# OpenAI Configuration  
OPENAI_API_KEY=your_openai_api_key_here

# Slack Configuration
SLACK_TOKEN=your_slack_token_here
SLACK_CHANNELS=C0123456789,C9876543210

# Gmail Configuration
GMAIL_CREDENTIALS=token.json

# Trello Configuration
TRELLO_API_KEY=your_trello_api_key_here
TRELLO_TOKEN=your_trello_token_here
TRELLO_BOARD_ID=your_board_id_here
```

### 5. Run the Application

```bash
python -m notion_automation_pro.main
```

## 📋 How It Works

1. **Monitoring**: The agent continuously monitors configured platforms for new content
2. **Processing**: New content is processed using OpenAI to generate structured information
3. **Creation**: Notion pages are automatically created with the processed content
4. **Assignment**: Users are assigned to pages based on the content source
5. **Dashboard**: Visual dashboard is updated with progress and statistics

## 🏗️ Architecture

```
notion_automation_pro/
├── config.py              # Configuration management
├── main.py                # Application entry point
├── autonomous_agent.py    # Main agent orchestrator
├── services/              # Platform integrations
│   ├── openai_service.py  # OpenAI integration
│   ├── notion_service.py  # Notion API client
│   ├── slack_service.py   # Slack integration
│   ├── gmail_service.py   # Gmail integration
│   └── trello_service.py  # Trello integration
└── utils/                 # Utility modules
    ├── logger.py          # Enhanced logging
    ├── rate_limiter.py    # API rate limiting
    └── retry.py           # Retry mechanisms
```

## 🔧 Configuration Options

### Polling Interval
```bash
POLLING_INTERVAL=1  # Minutes between monitoring cycles
```

### User Mapping
```bash
USER_SLACK=Użytkownik1
USER_GMAIL=Użytkownik2  
USER_TRELLO=Użytkownik3
USER_NOTION=UżytkownikDomyślny
```

### Logging
```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Rate Limiting
```bash
SLACK_RATE_LIMIT=50      # Calls per minute
MAX_RETRIES=3            # Maximum retry attempts
RETRY_DELAY=1.0          # Base retry delay in seconds
```

## 📊 Monitoring & Logging

The application provides comprehensive monitoring:

- **Console Output**: Colored, real-time status updates
- **Log Files**: Detailed logs saved to `logs/notion_automation.log`
- **Dashboard**: Visual progress tracking in Notion
- **Statistics**: Detailed usage statistics for all services

## 🛡️ Error Handling

- **Automatic Retries**: Failed API calls are automatically retried with exponential backoff
- **Rate Limiting**: Built-in protection against API rate limits
- **Graceful Degradation**: Service failures don't crash the entire application
- **Detailed Logging**: All errors are logged with context for debugging

## 🔒 Security Best Practices

- **Environment Variables**: All credentials stored in environment variables
- **No Hardcoded Secrets**: No API keys or tokens in source code
- **Secure Defaults**: Conservative rate limits and timeouts
- **Input Validation**: All inputs validated before processing

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify all API credentials are correct
   - Check token permissions and scopes

2. **Rate Limiting**
   - Reduce polling frequency
   - Check API usage quotas

3. **Service Validation Failures**
   - Run with `LOG_LEVEL=DEBUG` for detailed diagnostics
   - Verify network connectivity to APIs

### Debug Mode

```bash
LOG_LEVEL=DEBUG python -m notion_automation_pro.main
```

## 📈 Performance

- **Efficient Polling**: Only processes new content since last check
- **Batch Processing**: Multiple items processed in single cycles
- **Memory Management**: Bounded memory usage with cleanup
- **Rate Limiting**: Respects API limits to prevent throttling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original concept and implementation
- Enhanced by Codegen AI with improved architecture, error handling, and monitoring
- Built with modern Python best practices and comprehensive testing

---

**Note**: This is an enhanced version of the original Notion automation script with significant improvements in reliability, security, and maintainability.

