import pymupdf
import os

from vector_db import VectorDB


def get_pdf_filenames(directory):
    return [f for f in os.listdir(directory) if f.endswith(".pdf")]


def process_pdf(pdf_path):
    pdf_doc = pymupdf.open(filename=pdf_path, filetype="pdf")
    text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        text += page.get_text("text")
    pdf_doc.close()
    return text


if __name__ == "__main__":
    resume_path = os.path.join(os.path.dirname(__file__).split("app")[0], "data", "resumes")
    pdf_filenames = get_pdf_filenames(resume_path)

    raw_items = []
    for fn in pdf_filenames:
        full_path = os.path.join(resume_path, fn)
        raw_items.append({"chunk_heading": fn, "text": process_pdf(full_path)})

    db = VectorDB("linkedin_profiles")
    db.load_data(raw_items, overlap_words=50)

    query = "find me someone who knows machine learning"
    results = db.search(query, k=2)

    for result in results:
        print(result)
        print()
