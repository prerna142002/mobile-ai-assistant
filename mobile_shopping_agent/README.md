# Mobile Shopping Chat Agent

An AI-powered conversational agent that helps users find and compare mobile phones based on their preferences.

## Features
- Natural language understanding for mobile phone queries
- Intent classification to understand user requests
- Mobile phone database with filtering capabilities
- Conversational interface for easy interaction

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. Run the application:
   ```
   python app.py
   ```

## Project Structure
- `agent.py` - Main chat agent implementation
- `data_loader.py` - Handles loading and querying mobile phone data
- `models/` - Contains ML models for intent classification
- `utils/` - Utility functions for text processing
- `data/` - Sample mobile phone dataset

## API Usage

### Start a conversation
```
POST /api/chat
{
    "message": "I'm looking for an iPhone under $1000"
}
```

### Response Format
```json
{
    "response": "Here are some iPhones under $1000...",
    "suggestions": ["Show more", "Filter by storage", "Compare models"],
    "products": [...]
}
```
