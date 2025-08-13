import os
from pdfminer.high_level import extract_text
from docx import Document
import openai
from django.conf import settings
import json
import re
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO

# Download required NLTK data (this will be done once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text(file_path)
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return "Unsupported file format."

def clean_text(text):
    """Clean and normalize text for better analysis."""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text

def extract_key_phrases(text, num_phrases=8):
    """Extract key phrases using TF-IDF-like approach."""
    try:
        # Tokenize and clean
        words = word_tokenize(text.lower())
        # Remove stopwords and short words
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Get most common words
        common_words = word_freq.most_common(20)
        
        # Find sentences containing these key words
        sentences = sent_tokenize(text)
        key_sentences = []
        
        for word, freq in common_words[:10]:
            for sentence in sentences:
                if word.lower() in sentence.lower() and len(sentence) > 30:
                    # Clean and truncate sentence
                    clean_sent = clean_text(sentence)
                    if len(clean_sent) > 50:
                        clean_sent = clean_sent[:150] + "..." if len(clean_sent) > 150 else clean_sent
                    key_sentences.append(clean_sent)
                    if len(key_sentences) >= num_phrases:
                        break
            if len(key_sentences) >= num_phrases:
                break
        
        return key_sentences[:num_phrases]
    except:
        # Fallback to simple sentence extraction
        sentences = re.split(r'[.!?]+', text)
        return [s.strip()[:100] + "..." for s in sentences[:5] if len(s.strip()) > 20]

def identify_document_structure(text):
    """Identify the logical structure of the document."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    structure = {}
    
    if len(paragraphs) == 0:
        return {"Content": ["No structured content found"]}
    
    # Analyze paragraph patterns
    para_lengths = [len(p) for p in paragraphs]
    avg_length = sum(para_lengths) / len(para_lengths) if para_lengths else 0
    
    # Identify potential headers (shorter paragraphs that might be titles)
    potential_headers = []
    for i, para in enumerate(paragraphs):
        if len(para) < avg_length * 0.5 and len(para) < 100:
            potential_headers.append((i, para))
    
    if len(potential_headers) >= 2:
        # Use potential headers to create structure
        for i, (idx, header) in enumerate(potential_headers):
            section_name = header[:50] + "..." if len(header) > 50 else header
            start_idx = idx
            end_idx = potential_headers[i + 1][0] if i + 1 < len(potential_headers) else len(paragraphs)
            
            section_content = paragraphs[start_idx:end_idx]
            if len(section_content) > 1:  # Skip if only header
                structure[section_name] = section_content[1:]  # Exclude header from content
    else:
        # Fallback: organize by content type
        if len(paragraphs) >= 3:
            structure["Introduction"] = paragraphs[:2]
            if len(paragraphs) > 4:
                structure["Main Content"] = paragraphs[2:-2]
                structure["Conclusion"] = paragraphs[-2:]
            else:
                structure["Main Content"] = paragraphs[2:]
        else:
            structure["Content"] = paragraphs
    
    return structure

def create_intelligent_summary(text, max_length=200):
    """Create an intelligent summary using multiple strategies."""
    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Strategy 1: Use first few sentences (often contain main points)
        intro_summary = " ".join(sentences[:2])
        
        # Strategy 2: Find sentences with key terms
        key_words = extract_key_phrases(text, 5)
        key_sentences = []
        for sentence in sentences:
            if any(word.lower() in sentence.lower() for word in key_words[:3]):
                key_sentences.append(sentence)
                if len(" ".join(key_sentences)) > max_length:
                    break
        
        # Strategy 3: Combine strategies
        if len(intro_summary) <= max_length:
            summary = intro_summary
        elif key_sentences:
            summary = " ".join(key_sentences[:2])
        else:
            summary = sentences[0]
        
        # Clean and truncate
        summary = clean_text(summary)
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
        
    except:
        # Fallback to simple truncation
        return text[:max_length] + "..." if len(text) > max_length else text

def analyze_document_metadata(text):
    """Analyze document metadata and characteristics."""
    try:
        # Basic statistics
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(sent_tokenize(text))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # Estimate reading time (average 200 words per minute)
        reading_time = max(1, round(word_count / 200))
        
        # Identify document type based on content patterns
        doc_type = "General Document"
        if any(word in text.lower() for word in ['resume', 'cv', 'curriculum vitae']):
            doc_type = "Resume/CV"
        elif any(word in text.lower() for word in ['report', 'analysis', 'study']):
            doc_type = "Report/Analysis"
        elif any(word in text.lower() for word in ['proposal', 'plan', 'strategy']):
            doc_type = "Proposal/Plan"
        elif any(word in text.lower() for word in ['article', 'paper', 'research']):
            doc_type = "Article/Research"
        
        return {
            "document_type": doc_type,
            "statistics": {
                "characters": char_count,
                "words": word_count,
                "sentences": sentence_count,
                "paragraphs": paragraph_count,
                "estimated_reading_time": f"{reading_time} minute{'s' if reading_time != 1 else ''}"
            }
        }
    except:
        return {
            "document_type": "General Document",
            "statistics": {"error": "Unable to analyze document statistics"}
        }

def process_text_locally(extracted_text):
    """
    Enhanced local text processing with intelligent analysis.
    Provides sophisticated text analysis and structuring without API calls.
    """
    try:
        # Clean the text
        text = clean_text(extracted_text)
        
        # Extract key points using intelligent phrase extraction
        key_points = extract_key_phrases(text, 8)
        
        # Identify document structure
        structured_content = identify_document_structure(text)
        
        # Create intelligent summary
        summary = create_intelligent_summary(text, 250)
        
        # Analyze document metadata
        metadata = analyze_document_metadata(text)
        
        return {
            "key_points": key_points,
            "structured_content": structured_content,
            "summary": summary,
            "processing_method": "local_fallback",
            "metadata": metadata,
            "analysis_quality": "enhanced_local"
        }
        
    except Exception as e:
        # Fallback to basic processing
        return {
            "key_points": ["Enhanced local processing completed"],
            "structured_content": {"Content": ["Text was processed using enhanced local analysis"]},
            "summary": "Text was processed using advanced local analysis methods.",
            "processing_method": "local_fallback",
            "analysis_quality": "basic_fallback",
            "error": str(e)
        }

def process_text_with_ai(extracted_text):
    """
    Process extracted text through OpenAI API to:
    1. Analyze content and identify key points
    2. Restructure into organized format
    3. Create a clear summary
    """
    try:
        # Configure OpenAI
        openai.api_key = settings.OPENAI_API_KEY
        
        # Create the prompt for AI processing
        prompt = f"""
        Please analyze the following text and provide:
        
        1. **Key Points**: Identify the main topics and important information
        2. **Structured Content**: Organize the content into clear sections with bullet points
        3. **Summary**: Create a concise summary (2-3 sentences) highlighting the essential information
        
        Text to analyze:
        {extracted_text[:3000]}  # Limit text length to avoid token limits
        
        Please format your response as JSON with the following structure:
        {{
            "key_points": ["point1", "point2", "point3"],
            "structured_content": {{
                "section1": ["bullet1", "bullet2"],
                "section2": ["bullet1", "bullet2"]
            }},
            "summary": "2-3 sentence summary here"
        }}
        """
        
        # Call OpenAI API - try newer format first, fallback to older
        try:
            # Newer OpenAI API format (v1.0+)
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes and structures text content. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=0.3
            )
            ai_response = response.choices[0].message.content.strip()
        except AttributeError:
            # Fallback to older OpenAI API format
            response = openai.ChatCompletion.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes and structures text content. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=0.3
            )
            ai_response = response.choices[0].message.content.strip()
        
        # Try to parse JSON response
        try:
            processed_data = json.loads(ai_response)
            processed_data["processing_method"] = "ai_processing"
            processed_data["analysis_quality"] = "ai_enhanced"
            return processed_data
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "key_points": ["AI processing completed but response format was unexpected"],
                "structured_content": {"Content": ["AI processed content available"]},
                "summary": "Text was processed by AI but the response format was unexpected.",
                "processing_method": "ai_processing",
                "analysis_quality": "ai_enhanced",
                "raw_ai_response": ai_response
            }
            
    except Exception as e:
        # Check if it's a quota/API error and fall back to local processing
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['quota', 'insufficient', '429', 'billing']):
            # Use enhanced local processing as fallback
            return process_text_locally(extracted_text)
        else:
            # Return error information for other types of failures
            return {
                "key_points": ["Error occurred during AI processing"],
                "structured_content": {"Error": [f"AI processing failed: {str(e)}"]},
                "summary": f"Unable to process text with AI due to error: {str(e)}",
                "processing_method": "error",
                "analysis_quality": "error",
                "error": str(e)
            }

def generate_pdf_report(processed_data, original_filename):
    """
    Generate a professional PDF report from the processed text data.
    """
    try:
        # Create a BytesIO buffer for the PDF
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue,
            spaceBefore=20
        )
        
        normal_style = styles['Normal']
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            leftIndent=20,
            spaceAfter=6
        )
        
        # Build the PDF content
        story = []
        
        # Title
        story.append(Paragraph("Document Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Original filename
        story.append(Paragraph(f"<b>Original File:</b> {original_filename}", normal_style))
        story.append(Spacer(1, 12))
        
        # Processing method
        if processed_data.get('processing_method') == 'ai_processing':
            story.append(Paragraph("<b>Processing Method:</b> AI-Enhanced Analysis", normal_style))
        else:
            story.append(Paragraph("<b>Processing Method:</b> Enhanced Local Analysis", normal_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Paragraph(processed_data.get('summary', 'No summary available.'), normal_style))
        story.append(Spacer(1, 20))
        
        # Key Points
        story.append(Paragraph("Key Points", heading_style))
        key_points = processed_data.get('key_points', [])
        if key_points:
            for point in key_points:
                story.append(Paragraph(f"• {point}", bullet_style))
        else:
            story.append(Paragraph("No key points identified.", normal_style))
        story.append(Spacer(1, 20))
        
        # Structured Content
        story.append(Paragraph("Structured Content", heading_style))
        structured_content = processed_data.get('structured_content', {})
        if structured_content:
            for section_name, content_list in structured_content.items():
                story.append(Paragraph(f"<b>{section_name}</b>", normal_style))
                if isinstance(content_list, list):
                    for item in content_list:
                        story.append(Paragraph(f"• {item}", bullet_style))
                else:
                    story.append(Paragraph(f"• {content_list}", bullet_style))
                story.append(Spacer(1, 12))
        else:
            story.append(Paragraph("No structured content available.", normal_style))
        
        # Build the PDF
        doc.build(story)
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
        
    except Exception as e:
        # Return a simple error PDF if generation fails
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        
        story = [
            Paragraph("Error Generating PDF", styles['Heading1']),
            Spacer(1, 20),
            Paragraph(f"An error occurred while generating the PDF: {str(e)}", styles['Normal'])
        ]
        
        doc.build(story)
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
