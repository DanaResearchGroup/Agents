import yaml
import json
import requests
import logging
from pathlib import Path
from typing import Any
from .models import RegistryResult, PaperRecord

logger = logging.getLogger(__name__)

def save_registry(result: RegistryResult, outdir: str or Path):
    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)

    # 1. paper_registry.yaml
    registry_data = [p.model_dump() for p in result.papers]
    with open(outp / "paper_registry.yaml", "w") as f:
        yaml.dump(registry_data, f, sort_keys=False)

    # 2. paper_registry.json
    with open(outp / "paper_registry.json", "w") as f:
        json.dump(registry_data, f, indent=2)

    # 3. manual_retrieval_queue.yaml
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
                "reason": "; ".join(p.relevance_reason[:2]),
                "source_url": p.source_url
            })
    
    with open(outp / "manual_retrieval_queue.yaml", "w") as f:
        yaml.dump({"manual_retrieval_queue": manual_queue}, f, sort_keys=False)

def download_file(url: str, outdir: Path, filename: str) -> Path or None:
    """
    Download a file from a URL. mimicing a browser to avoid 403 errors.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
    }
    try:
        response = requests.get(url, headers=headers, timeout=30, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        # Check Content-Type to avoid saving HTML landing pages as 'files'
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            logger.info(f"Skipping download for {url}: it appears to be an HTML page, not a direct file.")
            return None
            
        # Refine extension if it's currently generic
        if filename.endswith(".file"):
            ext_map = {
                'application/pdf': '.pdf',
                'application/zip': '.zip',
                'application/x-zip-compressed': '.zip',
                'text/plain': '.txt',
                'application/xml': '.xml',
                'text/xml': '.xml',
                'application/octet-stream': '.bin'
            }
            new_ext = None
            for ct, ext in ext_map.items():
                if ct in content_type:
                    new_ext = ext
                    break
            
            if new_ext:
                filename = filename.rsplit(".", 1)[0] + new_ext

        outdir.mkdir(parents=True, exist_ok=True)
        filepath = outdir / filename
        
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return filepath
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None
