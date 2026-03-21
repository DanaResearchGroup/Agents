import sys
import logging
from pathlib import Path
from accelerator.utils import download_file

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

def test_download(url: str):
    print(f"Testing download for URL: {url}")
    outdir = Path("outputs/test_downloads")
    filename = "test_si_file.file" # Extension will be refined by download_file
    
    filepath = download_file(url, outdir, filename)
    
    if filepath:
        print(f"\n[✓] Success! File saved to: {filepath}")
        print(f"File size: {filepath.stat().st_size} bytes")
    else:
        print(f"\n[✗] Download failed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_download.py <URL>")
        sys.exit(1)
    
    test_download(sys.argv[1])

##### Test 1 ######
# python test_download.py "https://europepmc.org/api/fulltextRepo?pmcId=PMC12878340&type=FILE&fileName=ao5c11182_si_001.pdf&mimeType=application/pdf"

##### Test 2 ######
# python test_download.py "https://pubs.acs.org/doi/suppl/10.1021/acsomega.5c11182/suppl_file/ao5c11182_si_001.pdf"