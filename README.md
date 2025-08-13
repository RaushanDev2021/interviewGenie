# InterviewGenie - AI-Powered Document Processor

A Django application that extracts text from PDF and Word documents and processes it through AI to provide structured, summarized content.

## Features

- **Document Upload**: Support for PDF and Word (.docx) files
- **Text Extraction**: Uses pdfminer.six and python-docx for reliable text extraction
- **AI Processing**: OpenAI GPT integration for content analysis and structuring
- **Structured Output**: 
  - Key points identification
  - Organized content sections
  - Executive summary
  - Original text preview

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API

Edit `core/settings.py` and set your OpenAI API key:

```python
OPENAI_API_KEY = "your-actual-openai-api-key"
```

Or set environment variables:

```bash
export OPENAI_API_KEY="your-actual-openai-api-key"
```

### 3. Run Django

```bash
python manage.py runserver
```

Visit `http://localhost:8000` to use the application.

## Usage

1. Upload a PDF or Word document
2. The system extracts the text
3. AI processes the content to identify key points and structure
4. View the organized, summarized results

## Configuration Options

In `core/settings.py`:

- `OPENAI_MODEL`: Choose between "gpt-4" or "gpt-3.5-turbo"
- `OPENAI_MAX_TOKENS`: Control response length (default: 2000)
- `OPENAI_API_KEY`: Your OpenAI API key

## API Usage

The AI processing function:
- Limits input text to 3000 characters to manage token usage
- Requests structured JSON responses
- Includes fallback error handling
- Supports both newer and older OpenAI API formats

## Cost Optimization

- Use `gpt-3.5-turbo` instead of `gpt-4` for lower costs
- Adjust `OPENAI_MAX_TOKENS` based on your needs
- Consider implementing text chunking for very long documents

## Error Handling

The system gracefully handles:
- API failures
- JSON parsing errors
- File format issues
- Network timeouts

## Security Notes

- Never commit API keys to version control
- Use environment variables in production
- Consider rate limiting for production use
- Validate file uploads and sizes 