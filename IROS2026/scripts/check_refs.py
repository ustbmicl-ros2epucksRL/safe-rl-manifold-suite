#!/usr/bin/env python3
"""Check bibliography entries against CrossRef and Google Scholar."""

import re
import json
import urllib.request
import urllib.parse
import time
import sys

TEX_FILE = "../main.tex"


def extract_bibitems(tex_path):
    """Extract bibitem keys and titles from thebibliography."""
    with open(tex_path, "r") as f:
        text = f.read()

    bib_section = re.search(
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
        text, re.DOTALL
    )
    if not bib_section:
        print("No thebibliography found!")
        return []

    entries = []
    blocks = re.split(r"\\bibitem\{", bib_section.group())[1:]
    for block in blocks:
        key = block.split("}")[0]
        # Extract title between ``...''
        title_match = re.search(r"``(.+?)''", block)
        if title_match:
            title = title_match.group(1)
        else:
            # Book-style: \emph{Title}
            emph_match = re.search(r"\\emph\{(.+?)\}", block)
            title = emph_match.group(1) if emph_match else ""
        # Clean LaTeX
        title = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", title)
        title = title.replace("\\", "").replace("{", "").replace("}", "")

        # Extract year
        year_match = re.search(r"(\d{4})", block)
        year = year_match.group(1) if year_match else ""

        # Extract arXiv ID if present
        arxiv_match = re.search(r"arXiv:(\d+\.\d+)", block)
        arxiv_id = arxiv_match.group(1) if arxiv_match else ""

        entries.append({"key": key, "title": title, "year": year, "arxiv": arxiv_id})

    return entries


def search_crossref(title):
    """Search CrossRef for a paper title."""
    query = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?query.bibliographic={query}&rows=3&select=title,DOI,author,published-print,published-online"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "RefChecker/1.0 (mailto:check@example.com)")
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("message", {}).get("items", [])
        return items if items else None
    except Exception as e:
        return {"error": str(e)}


def check_arxiv(arxiv_id):
    """Check if an arXiv paper exists."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode()
        if "<title>" in body and "Error" not in body.split("<title>")[1].split("</title>")[0]:
            title_match = re.search(r"<title>(.*?)</title>", body, re.DOTALL)
            titles = re.findall(r"<title>(.*?)</title>", body, re.DOTALL)
            # First <title> is feed title, second is paper title
            if len(titles) >= 2:
                return titles[1].strip()
        return None
    except Exception as e:
        return None


def normalize(s):
    """Lowercase, remove punctuation for fuzzy match."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_similarity(a, b):
    """Simple word overlap ratio."""
    wa = set(normalize(a).split())
    wb = set(normalize(b).split())
    if not wa or not wb:
        return 0.0
    overlap = wa & wb
    return len(overlap) / max(len(wa), len(wb))


def main():
    entries = extract_bibitems(TEX_FILE)
    print(f"Found {len(entries)} references.\n")
    print(f"{'#':<3} {'Key':<28} {'Status':<12} {'Details'}")
    print("-" * 110)

    issues = []
    for i, entry in enumerate(entries, 1):
        time.sleep(0.5)  # Polite rate

        # Try arXiv first if we have an ID
        if entry["arxiv"]:
            arxiv_title = check_arxiv(entry["arxiv"])
            if arxiv_title:
                sim = title_similarity(entry["title"], arxiv_title)
                if sim >= 0.5:
                    status = "OK"
                    detail = f"sim={sim:.2f} | arXiv:{entry['arxiv']} | {arxiv_title[:55]}"
                    print(f"{i:<3} {entry['key']:<28} {status:<12} {detail}")
                    continue

        # Search CrossRef
        results = search_crossref(entry["title"])

        if isinstance(results, dict) and "error" in results:
            status = "API_ERR"
            detail = results["error"][:80]
            issues.append((entry["key"], status, detail))
        elif results is None:
            status = "NOT FOUND"
            detail = f"Title: {entry['title'][:60]}"
            issues.append((entry["key"], status, detail))
        else:
            best = results[0]
            cr_title = best.get("title", [""])[0] if best.get("title") else ""
            doi = best.get("DOI", "")
            sim = title_similarity(entry["title"], cr_title)

            if sim >= 0.5:
                status = "OK"
                detail = f"sim={sim:.2f} | DOI:{doi} | {cr_title[:55]}"
            else:
                status = "MISMATCH"
                detail = f"sim={sim:.2f} | Ours: {entry['title'][:35]}... | CR: {cr_title[:35]}..."
                issues.append((entry["key"], status, detail))

        print(f"{i:<3} {entry['key']:<28} {status:<12} {detail}")

    print("\n" + "=" * 110)
    if issues:
        print(f"\n!! {len(issues)} issue(s) found:\n")
        for key, status, detail in issues:
            print(f"  [{status}] {key}: {detail}")
    else:
        print("\n  All references verified successfully.")

    return len(issues)


if __name__ == "__main__":
    sys.exit(main())
