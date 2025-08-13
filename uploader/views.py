from django.shortcuts import render
from django.http import HttpResponse
from .utils import extract_text_from_file, process_text_with_ai, generate_pdf_report
import os
import json

def upload_document(request):
    if request.method == 'POST' and request.FILES.get('document'):
        uploaded_file = request.FILES['document']
        file_path = f'/tmp/{uploaded_file.name}'

        # Save file temporarily
        with open(file_path, 'wb+') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Extract text
        extracted_text = extract_text_from_file(file_path)
        
        # Process text with AI
        ai_processed_data = process_text_with_ai(extracted_text)
        
        # Add original text to context for reference
        ai_processed_data['original_text'] = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
        ai_processed_data['original_filename'] = uploaded_file.name

        # Remove temp file
        os.remove(file_path)

        return render(request, 'result.html', ai_processed_data)

    return render(request, 'upload.html')

def download_pdf(request):
    """Download the processed content as a PDF."""
    if request.method == 'POST':
        try:
            # Get the processed data from the form
            processed_data = {
                'summary': request.POST.get('summary', ''),
                'key_points': request.POST.getlist('key_points'),
                'structured_content': {},  # Initialize empty dict
                'processing_method': request.POST.get('processing_method', 'local_fallback'),
                'original_filename': request.POST.get('original_filename', 'document')
            }
            
            # Reconstruct structured content from individual form fields
            structured_content = {}
            for key, value in request.POST.items():
                if key.startswith('structured_content_'):
                    section_name = key.replace('structured_content_', '')
                    if section_name not in structured_content:
                        structured_content[section_name] = []
                    structured_content[section_name].append(value)
            
            processed_data['structured_content'] = structured_content
            
            # Generate PDF
            pdf_content = generate_pdf_report(processed_data, processed_data['original_filename'])
            
            # Create HTTP response with PDF
            response = HttpResponse(pdf_content, content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="analysis_report_{processed_data["original_filename"]}.pdf"'
            
            return response
            
        except Exception as e:
            return HttpResponse(f"Error generating PDF: {str(e)}", status=500)
    
    return HttpResponse("Invalid request method", status=405)

# Create your views here.
