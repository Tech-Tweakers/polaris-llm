from pypdf import PdfReader

reader = PdfReader("pdf-input-01.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"
    print(text)