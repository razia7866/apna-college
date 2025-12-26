"""
Tool: Fetch research papers from ArXiv API
"""

import feedparser
import requests
import os

def search_arxiv(query, max_results=5):
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance"
    }

    response = requests.get(url, params=params, timeout=30)
    feed = feedparser.parse(response.text)

    papers = []
    for entry in feed.entries:
        pdf_url = None
        for link in entry.links:
            if link.type == "application/pdf":
                pdf_url = link.href

        papers.append({
            "id": entry.id.split("/")[-1],
            "title": entry.title,
            "summary": entry.summary,
            "pdf_url": pdf_url
        })

    return papers


def download_pdf(pdf_url, save_path):
    response = requests.get(pdf_url, timeout=30)
    with open(save_path, "wb") as f:
        f.write(response.content)
