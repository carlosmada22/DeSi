#!/usr/bin/env python3
"""
Metadata Handler for Multi-Source Documentation

This module handles metadata extraction and enrichment for different
documentation sources, preserving source-specific information.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class MetadataHandler:
    """Handles metadata extraction and enrichment for multiple sources."""
    
    def __init__(self):
        """Initialize the metadata handler."""
        self.source_priorities = {
            'wikijs': 1,      # Wiki.js content gets higher priority
            'datastore': 1,   # Same as wikijs
            'openbis': 2,     # OpenBIS content gets lower priority
        }
    
    def extract_metadata(self, file_path: Path, content: str) -> Dict[str, str]:
        """
        Extract comprehensive metadata from file and content.
        
        Args:
            file_path: Path to the source file
            content: Raw file content
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {}
        
        # Extract basic metadata from content header
        basic_metadata = self._extract_basic_metadata(content)
        metadata.update(basic_metadata)
        
        # Extract file-based metadata
        file_metadata = self._extract_file_metadata(file_path)
        metadata.update(file_metadata)
        
        # Enrich with source-specific metadata
        source = metadata.get('source', '').lower()
        if source:
            source_metadata = self._extract_source_specific_metadata(content, source)
            metadata.update(source_metadata)
        
        # Add processing metadata
        processing_metadata = self._add_processing_metadata(file_path)
        metadata.update(processing_metadata)
        
        return metadata
    
    def _extract_basic_metadata(self, content: str) -> Dict[str, str]:
        """Extract basic metadata from content header."""
        metadata = {}
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('Title: '):
                metadata['title'] = line[len('Title: '):].strip()
            elif line.startswith('URL: '):
                metadata['url'] = line[len('URL: '):].strip()
            elif line.startswith('Source: '):
                metadata['source'] = line[len('Source: '):].strip()
            elif line.strip() == '---':
                break
        
        return metadata
    
    def _extract_file_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from file path and properties."""
        metadata = {}
        
        # File path information
        metadata['file_path'] = str(file_path)
        metadata['file_name'] = file_path.name
        metadata['file_stem'] = file_path.stem
        
        # File modification time
        if file_path.exists():
            mtime = file_path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
            metadata['file_modified'] = dt.isoformat().replace('+00:00', 'Z')
        
        # Extract information from filename patterns
        filename = file_path.stem
        
        # OpenBIS pattern: en_version_section_subsection
        openbis_match = re.match(r'en_([^_]+)_(.+)', filename)
        if openbis_match:
            metadata['version'] = openbis_match.group(1)
            metadata['path_hierarchy'] = openbis_match.group(2).replace('_', '/')
        
        # Wiki.js pattern: en_section_subsection
        wikijs_match = re.match(r'en_(.+)', filename)
        if wikijs_match and not openbis_match:
            metadata['path_hierarchy'] = wikijs_match.group(1).replace('_', '/')
        
        return metadata
    
    def _extract_source_specific_metadata(self, content: str, source: str) -> Dict[str, str]:
        """Extract source-specific metadata."""
        metadata = {}
        
        if source in ['wikijs', 'datastore']:
            metadata.update(self._extract_wikijs_metadata(content))
        elif source == 'openbis':
            metadata.update(self._extract_openbis_metadata(content))
        
        return metadata
    
    def _extract_wikijs_metadata(self, content: str) -> Dict[str, str]:
        """Extract Wiki.js specific metadata."""
        metadata = {}
        lines = content.split('\n')
        
        # Look for breadcrumb navigation
        breadcrumbs = []
        for line in lines[:10]:  # Check first 10 lines
            if line.strip().startswith('/') and not line.strip().startswith('//'):
                breadcrumbs.append(line.strip().lstrip('/'))
        
        if breadcrumbs:
            metadata['breadcrumbs'] = ' > '.join(breadcrumbs)
        
        # Look for "Last edited by" information
        for line in lines:
            if line.startswith('Last edited by'):
                metadata['last_edited_by'] = line[len('Last edited by'):].strip()
            elif re.match(r'\d{2}/\d{2}/\d{4}', line.strip()):
                metadata['last_edited_date'] = line.strip()
        
        # Set source priority
        metadata['source_priority'] = self.source_priorities.get('wikijs', 2)
        
        return metadata
    
    def _extract_openbis_metadata(self, content: str) -> Dict[str, str]:
        """Extract OpenBIS specific metadata."""
        metadata = {}
        
        # Extract version from URL if available
        url = metadata.get('url', '')
        if url:
            version_match = re.search(r'/(\d+\.\d+\.\d+-\d+)/', url)
            if version_match:
                metadata['openbis_version'] = version_match.group(1)
        
        # Determine documentation type from content
        if 'user-documentation' in content.lower():
            metadata['doc_type'] = 'user'
        elif 'software-developer-documentation' in content.lower():
            metadata['doc_type'] = 'developer'
        elif 'system-documentation' in content.lower():
            metadata['doc_type'] = 'system'
        
        # Set source priority
        metadata['source_priority'] = self.source_priorities.get('openbis', 2)
        
        return metadata
    
    def _add_processing_metadata(self, file_path: Path) -> Dict[str, str]:
        """Add processing-related metadata."""
        metadata = {}
        
        # Processing timestamp
        metadata['processed_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        # Determine relative path from data root
        try:
            # Find the data/raw part of the path
            parts = file_path.parts
            if 'raw' in parts:
                raw_index = parts.index('raw')
                if raw_index + 1 < len(parts):
                    metadata['source_directory'] = parts[raw_index + 1]
        except (ValueError, IndexError):
            pass
        
        return metadata
    
    def enrich_chunk_metadata(
        self,
        base_metadata: Dict[str, str],
        chunk_content: str,
        chunk_index: int,
        section_title: str = ""
    ) -> Dict[str, str]:
        """
        Enrich metadata for individual chunks.
        
        Args:
            base_metadata: Base metadata from file
            chunk_content: Content of the chunk
            chunk_index: Index of the chunk in the document
            section_title: Title of the section this chunk belongs to
            
        Returns:
            Enriched metadata for the chunk
        """
        chunk_metadata = base_metadata.copy()
        
        # Add chunk-specific metadata
        chunk_metadata['chunk_index'] = chunk_index
        chunk_metadata['chunk_length'] = len(chunk_content)
        chunk_metadata['chunk_id'] = f"{base_metadata.get('file_stem', 'unknown')}_{chunk_index}"
        
        if section_title:
            chunk_metadata['section_title'] = section_title
        
        # Extract first heading from chunk as chunk title
        lines = chunk_content.split('\n')
        for line in lines:
            if line.strip().startswith('#'):
                chunk_title = re.sub(r'^#+\s*', '', line.strip())
                chunk_metadata['chunk_title'] = chunk_title
                break
        
        # Determine content type
        chunk_metadata['content_type'] = self._determine_content_type(chunk_content)
        
        return chunk_metadata
    
    def _determine_content_type(self, content: str) -> str:
        """Determine the type of content in a chunk."""
        content_lower = content.lower()
        
        # Check for code content
        if any(indicator in content_lower for indicator in ['```', 'code', 'script', 'function', 'class']):
            return 'code'
        
        # Check for procedural content
        if any(indicator in content_lower for indicator in ['step', 'click', 'select', 'navigate', 'enter']):
            return 'procedure'
        
        # Check for conceptual content
        if any(indicator in content_lower for indicator in ['overview', 'introduction', 'concept', 'definition']):
            return 'concept'
        
        # Check for reference content
        if any(indicator in content_lower for indicator in ['api', 'reference', 'parameter', 'method']):
            return 'reference'
        
        return 'general'
    
    def create_source_url(self, metadata: Dict[str, str]) -> str:
        """
        Create a source URL for the content.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Source URL string
        """
        # Use the original URL if available
        if 'url' in metadata and metadata['url']:
            return metadata['url']
        
        # Try to construct URL from other metadata
        source = metadata.get('source', '').lower()
        
        if source == 'openbis':
            version = metadata.get('openbis_version', '20.10.0-11')
            path = metadata.get('path_hierarchy', '')
            return f"https://openbis.readthedocs.io/en/{version}/{path}.html"
        
        elif source in ['wikijs', 'datastore']:
            path = metadata.get('path_hierarchy', '')
            return f"https://datastore.bam.de/en/{path}"
        
        return ""
