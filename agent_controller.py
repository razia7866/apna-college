"""
Agent Controller
Plans and executes the full Research Paper AI Assistant workflow
"""

import os
from tqdm import tqdm

from arxiv_tool import search_arxiv, download_pdf
from pdf_parser import extract_text, chunk_text
from vector_store import add_chunks
from rag_pipeline import summarize_paper, generate_literature_review


def run_agent(topic, max_papers=5):
    """
    Main agent function that runs the complete pipeline
    """

    print("🔍 Searching research papers...")
    papers = search_arxiv(topic, max_papers)

    summaries = []

    # Ensure PDF directory exists
    os.makedirs("data/pdfs", exist_ok=True)

    for paper in tqdm(papers):
        print(f"\n📄 Processing: {paper['title']}")

        pdf_path = f"data/pdfs/{paper['id']}.pdf"

        # Download PDF
        if paper["pdf_url"]:
            download_pdf(paper["pdf_url"], pdf_path)
        else:
            continue

        # Extract and chunk text
        text = extract_text(pdf_path)
        chunks = chunk_text(text)

        # Store in vector DB
        add_chunks(paper["id"], chunks)

        # Summarize paper using RAG
        summary = summarize_paper(paper["title"])
        summaries.append(summary)

    if not summaries:
        print("❌ No summaries generated.")
        return

    print("\n🧠 Generating final literature review...")
    final_review = generate_literature_review("\n\n".join(summaries))

    # Save output
    with open("literature_review.txt", "w", encoding="utf-8") as f:
        f.write(final_review)

    print("✅ Literature review saved as literature_review.txt")
