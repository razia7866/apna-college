"""
RAG-based summarization and synthesis
"""

import openai
from vector_store import retrieve

openai.api_key = "AIzaSyBJDIl0223Rcrf0ZUB1T8OMFAfKq99mrgM"


def summarize_paper(title):
    context_chunks = retrieve(title, k=5)
    context = "\n".join(context_chunks)

    prompt = f"""
You are an expert research assistant.

Using the context below, summarize the research paper.

Context:
{context}

Return:
- Main contribution
- Methodology
- Key findings
- Limitations
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


def generate_literature_review(summaries):
    prompt = f"""
You are an academic researcher.

Given the following paper summaries, write a coherent literature review.

Summaries:
{summaries}

Write 600-800 words.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content
