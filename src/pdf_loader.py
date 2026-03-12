import PyPDF2
import io
 
 
def load_pdfs(uploaded_files) -> str:
    """
    Extract and combine text from one or more uploaded PDF files.
 
    How it works:
        1. Reads each uploaded file as bytes
        2. Uses PyPDF2.PdfReader to parse the PDF
        3. Extracts text page by page
        4. Combines all text into one string with document/page markers
 
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
 
    Returns:
        Single string of all extracted text from all PDFs
 
    Raises:
        ValueError: If no text could be extracted (e.g. scanned PDFs)
    """
    all_text = ""
 
    for uploaded_file in uploaded_files:
        try:
            # Read raw bytes from Streamlit upload
            pdf_bytes = uploaded_file.read()
 
            # Wrap in BytesIO so PyPDF2 can parse it
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            num_pages  = len(pdf_reader.pages)
 
            # Document header marker
            file_text = f"\n\n=== Document: {uploaded_file.name} ({num_pages} pages) ===\n\n"
 
            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    file_text += f"[Page {page_num + 1}]\n{page_text.strip()}\n\n"
 
            all_text += file_text
 
        except Exception as e:
            print(f"Warning: Could not read '{uploaded_file.name}': {e}")
            continue
 
    if not all_text.strip():
        raise ValueError(
            "Could not extract any text from the uploaded PDF(s).\n"
            "Make sure the PDF contains selectable text (not scanned images)."
        )
 
    return all_text
