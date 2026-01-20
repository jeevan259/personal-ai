#!/usr/bin/env python3
"""
Import documents into knowledge base
"""

import argparse
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import get_settings
from utils.logging_config import setup_logging
from knowledge.ingestors.document_ingestor import DocumentIngestor
import logging

logger = logging.getLogger(__name__)

async def import_knowledge(source_dir: Path, recursive: bool = True):
    """Import documents from directory"""
    settings = get_settings()
    setup_logging(settings.log_level, settings.logs_dir)
    
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return False
        
    logger.info(f"Importing knowledge from: {source_dir}")
    
    # Create ingestor
    ingestor = DocumentIngestor(
        vector_store_dir=settings.memory_db_dir / "chroma_db"
    )
    
    # Ingest documents
    results = await ingestor.ingest_directory(source_dir)
    
    # Print results
    print(f"\nImport Results:")
    print(f"  Total files: {results['total_files']}")
    print(f"  Successful: {results['successful']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Total chunks: {results['total_chunks']}")
    
    if results['failed_files']:
        print(f"\nFailed files ({len(results['failed_files'])}):")
        for file in results['failed_files'][:10]:  # Show first 10
            print(f"  - {file}")
        if len(results['failed_files']) > 10:
            print(f"  ... and {len(results['failed_files']) - 10} more")
            
    return results['successful'] > 0

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Import documents into knowledge base")
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing documents to import"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Import documents recursively (default: True)"
    )
    
    args = parser.parse_args()
    
    success = await import_knowledge(args.source_dir, args.recursive)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))