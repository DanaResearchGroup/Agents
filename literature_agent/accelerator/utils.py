import os
import json
import yaml
import requests
import logging
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING
from openai import OpenAI
from dotenv import load_dotenv

if TYPE_CHECKING:
    from .models import RegistryResult, PaperRecord

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- LLM Utilities (from llm_utils) ---
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

def call_deepseek(prompt: str, system_prompt: str = "You are a helpful assistant.") -> dict:
    """Utility to call the DeepSeek API and return a JSON dictionary."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                stream=False
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise e

# --- I/O Utilities (from io_utils) ---
def save_registry(result: 'RegistryResult', outdir: str or Path):
    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)

    # 1. paper_registry.yaml/json
    registry_data = [p.model_dump() for p in result.papers]
    with open(outp / "paper_registry.yaml", "w") as f:
        yaml.dump(registry_data, f, sort_keys=False)

    with open(outp / "paper_registry.json", "w") as f:
        json.dump(registry_data, f, indent=2)

    # 2. manual_retrieval_queue.yaml
    manual_queue = []
    for p in result.papers:
        if p.keep_for_manual_review:
            manual_queue.append({
                "id": p.id,
                "title": p.title,
                "doi": p.doi,
                "year": p.year,
                "journal": p.journal,
                "priority": p.priority,
                "reason": p.match_reason or "Direct relevance.",
                "source_url": p.source_url
            })
    
    with open(outp / "manual_retrieval_queue.yaml", "w") as f:
        yaml.dump({"manual_retrieval_queue": manual_queue}, f, sort_keys=False)

def download_file(url: str, outdir: Path, filename: str) -> Path or None:
    """Download a file from a URL, mimicking a browser."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Referer": "https://www.google.com/",
    }
    try:
        response = requests.get(url, headers=headers, timeout=30, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            return None
            
        if filename.endswith(".file"):
            ext_map = {'application/pdf': '.pdf', 'application/zip': '.zip', 'text/plain': '.txt', 'text/xml': '.xml'}
            for ct, ext in ext_map.items():
                if ct in content_type:
                    filename = filename.rsplit(".", 1)[0] + ext
                    break

        outdir.mkdir(parents=True, exist_ok=True)
        filepath = outdir / filename
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filepath
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None
