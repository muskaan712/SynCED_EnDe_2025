#!/usr/bin/env python3
# data_scrape.py
#
# Purpose:
#   Scrape text data from selected public sources for use in SynCED-EnDe.
#   Currently implemented:
#     - GOV.UK announcements and guidance
#     - Stack Exchange Q&A threads
#
# Input : None (URLs hardcoded or passed in future extensions)
# Output: TSV/CSV with scraped content
#
# Note: This script is for reproducibility. Sources are all public-domain or
#       permissively licensed. FDA scraping code has been removed.

import os
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="data_scrape.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── GOV.UK Scraper ──────────────────────────────────────────────────────
def scrape_govuk(base_url="https://www.gov.uk", limit=5):
    """Scrape GOV.UK announcements (title + body)."""
    collected = []
    try:
        for i in range(1, limit + 1):
            url = f"{base_url}/search/news-and-communications?page={i}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            links = soup.select("a.gem-c-document-list__item-title")
            for a in links:
                title = a.text.strip()
                link = base_url + a["href"]

                try:
                    page = requests.get(link, timeout=10)
                    page.raise_for_status()
                    page_soup = BeautifulSoup(page.text, "html.parser")
                    body = " ".join([p.text.strip() for p in page_soup.select("main p")])
                except Exception as e:
                    logging.warning(f"Failed to fetch GOV.UK page {link}: {e}")
                    body = ""

                collected.append({"source": "govuk", "title": title, "text": body, "url": link})
        logging.info(f"Scraped {len(collected)} GOV.UK entries")
    except Exception as e:
        logging.error(f"GOV.UK scraping failed: {e}")
    return collected

# ─── Stack Exchange Scraper ──────────────────────────────────────────────
def scrape_stackexchange(base_url="https://stackoverflow.com/questions", limit=3):
    """Scrape Stack Exchange (questions + answers)."""
    collected = []
    try:
        for i in range(1, limit + 1):
            url = f"{base_url}?page={i}&sort=votes"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            questions = soup.select("div.s-post-summary")
            for q in questions:
                title_el = q.select_one("h3 a")
                if not title_el:
                    continue
                title = title_el.text.strip()
                link = "https://stackoverflow.com" + title_el["href"]

                try:
                    page = requests.get(link, timeout=10)
                    page.raise_for_status()
                    page_soup = BeautifulSoup(page.text, "html.parser")
                    q_body = " ".join([p.text.strip() for p in page_soup.select("div.s-prose p")])
                except Exception as e:
                    logging.warning(f"Failed to fetch StackExchange page {link}: {e}")
                    q_body = ""

                collected.append({"source": "stackexchange", "title": title, "text": q_body, "url": link})
        logging.info(f"Scraped {len(collected)} StackExchange entries")
    except Exception as e:
        logging.error(f"StackExchange scraping failed: {e}")
    return collected

# ─── Main ───────────────────────────────────────────────────────────────
def main():
    govuk_data = scrape_govuk(limit=3)
    se_data = scrape_stackexchange(limit=2)

    combined = govuk_data + se_data
    if not combined:
        print("⚠️ No data scraped.")
        return

    out_file = "scraped_data.tsv"
    pd.DataFrame(combined).to_csv(out_file, sep="\t", index=False)
    print(f"✅ Wrote {len(combined)} scraped rows to {out_file}")
    logging.info(f"Final scraped dataset size: {len(combined)}")

if __name__ == "__main__":
    main()
