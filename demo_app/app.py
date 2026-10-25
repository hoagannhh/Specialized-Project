"""CLI demo of the document pipeline; requires trained artifacts and API keys."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline.single_file_pipeline import main
if __name__=='__main__':main()
