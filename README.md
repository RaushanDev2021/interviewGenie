# 🤖 InterviewGenie - AI Document Processor

A Django app that extracts text from PDF/Word files and processes it with AI to create structured, summarized content.

## ✨ Features

- **Document Upload**: PDF and Word (.docx) support
- **AI Processing**: OpenAI GPT integration for smart analysis
- **Local Fallback**: Advanced NLP processing when AI unavailable
- **Structured Output**: Key points, organized sections, executive summary
- **PDF Reports**: Downloadable analysis documents
- **Modern UI**: Beautiful dark theme with drag & drop

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI (Optional)
Edit `core/settings.py`:
```python
OPENAI_API_KEY = "your-api-key-here"
OPENAI_MODEL = "gpt-3.5-turbo"
```

### 3. Run the App
```bash
python manage.py runserver
```

Visit `http://localhost:8000` to use the app.

## 📖 How to Use

1. **Upload**: Drag & drop or click to select a PDF/Word file
2. **Process**: AI analyzes content and creates structured output
3. **View**: See summary, key points, and organized sections
4. **Download**: Get a professional PDF report

## 🔧 Configuration

### OpenAI Settings
- `OPENAI_API_KEY`: Your API key
- `OPENAI_MODEL`: Choose "gpt-4" or "gpt-3.5-turbo"
- `OPENAI_MAX_TOKENS`: Response length (default: 2000)

### Processing Modes
- **AI-Enhanced**: Best results with OpenAI
- **Local Fallback**: Smart NLP when AI unavailable
- **Automatic**: Seamless switching between modes

## 🏗️ Architecture

- **Backend**: Django framework
- **Text Extraction**: pdfminer.six + python-docx
- **AI Processing**: OpenAI GPT API
- **Local Analysis**: NLTK for advanced NLP
- **PDF Generation**: ReportLab
- **Frontend**: Modern CSS with JavaScript

## 📱 Features

### Smart Analysis
- Key point extraction
- Content structuring
- Executive summaries
- Document classification

### User Experience
- Drag & drop uploads
- Responsive design
- Dark theme
- Interactive elements

## 🚨 Common Issues

- **API Quota**: App automatically uses local processing
- **File Errors**: Check format (.pdf, .docx) and size
- **Slow Processing**: Large documents take longer

## 🔒 Security

- CSRF protection
- File validation
- Secure uploads
- Environment variables for API keys

## 📄 Dependencies

```
Django>=5.2.5
pdfminer.six
python-docx
openai>=1.0.0
nltk>=3.8
reportlab>=4.0.0
lxml>=6.0.0
```

## 🚀 Production

```bash
# Use production server
gunicorn core.wsgi:application

# Set environment variables
export DEBUG=False
export OPENAI_API_KEY="your-key"
```

---

**InterviewGenie** - Transform documents into insights with AI! 🚀 