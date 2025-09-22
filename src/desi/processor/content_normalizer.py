#!/usr/bin/env python3
"""
Content Normalizer for Multi-Source Documentation

This module normalizes content from different sources (OpenBIS, Wiki.js) into
a consistent markdown format that can be processed by intelligent chunking algorithms.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContentNormalizer:
    """Normalizes content from different documentation sources."""
    
    def __init__(self):
        """Initialize the content normalizer."""
        self.source_handlers = {
            'openbis': self._normalize_openbis_content,
            'datastore': self._normalize_wikijs_content,
            'wikijs': self._normalize_wikijs_content,  # Alternative name
        }
    
    def normalize_file(self, file_path: Path) -> Dict[str, str]:
        """
        Normalize a single file to consistent markdown format.
        
        Args:
            file_path: Path to the file to normalize
            
        Returns:
            Dictionary containing normalized content and metadata
        """
        logger.info(f"Normalizing {file_path}")
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata and content
        metadata, raw_content = self._extract_metadata_and_content(content)
        
        # Determine source type
        source = metadata.get('source', '').lower()
        
        # Apply source-specific normalization
        if source in self.source_handlers:
            normalized_content = self.source_handlers[source](raw_content, metadata)
        else:
            logger.warning(f"Unknown source '{source}' for {file_path}, using generic normalization")
            normalized_content = self._normalize_generic_content(raw_content, metadata)
        
        return {
            'title': metadata.get('title', ''),
            'url': metadata.get('url', ''),
            'source': source,
            'file_path': str(file_path),
            'content': normalized_content,
            'metadata': metadata
        }
    
    def _extract_metadata_and_content(self, content: str) -> Tuple[Dict[str, str], str]:
        """Extract metadata and content from file."""
        metadata = {}
        lines = content.split('\n')
        content_start_idx = 0
        
        # Extract metadata from header
        for i, line in enumerate(lines):
            if line.startswith('Title: '):
                metadata['title'] = line[len('Title: '):].strip()
            elif line.startswith('URL: '):
                metadata['url'] = line[len('URL: '):].strip()
            elif line.startswith('Source: '):
                metadata['source'] = line[len('Source: '):].strip()
            elif line.strip() == '---':
                content_start_idx = i + 1
                break
        
        # Extract content after metadata separator
        raw_content = '\n'.join(lines[content_start_idx:]).strip()
        
        return metadata, raw_content
    
    def _normalize_openbis_content(self, content: str, metadata: Dict[str, str]) -> str:
        """Normalize OpenBIS ReadTheDocs content."""
        lines = content.split('\n')
        normalized_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                normalized_lines.append('')
                continue
            
            # Convert title to H1 if it matches the metadata title
            if line == metadata.get('title', ''):
                normalized_lines.append(f'# {line}')
                continue
            
            # Detect and convert headings (simple heuristic)
            if self._is_likely_heading(line) and not line.startswith('#'):
                # Determine heading level based on context and content
                heading_level = self._determine_heading_level(line, normalized_lines)
                normalized_lines.append(f'{"#" * heading_level} {line}')
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _normalize_wikijs_content(self, content: str, metadata: Dict[str, str]) -> str:
        """Normalize Wiki.js content."""
        lines = content.split('\n')
        normalized_lines = []
        
        # Skip breadcrumb navigation at the start
        skip_until_title = True
        
        for line in lines:
            line = line.strip()
            
            # Skip breadcrumb lines (starting with /)
            if skip_until_title and (line.startswith('/') or line == ''):
                continue
            
            # Skip metadata lines
            if line.startswith('Last edited by') or re.match(r'\d{2}/\d{2}/\d{4}', line):
                continue
            
            if not line:
                normalized_lines.append('')
                continue
            
            # Convert title to H1 if it matches the metadata title
            if line == metadata.get('title', ''):
                normalized_lines.append(f'# {line}')
                skip_until_title = False
                continue
            
            skip_until_title = False
            
            # Detect and convert headings
            if self._is_likely_heading(line) and not line.startswith('#'):
                heading_level = self._determine_heading_level(line, normalized_lines)
                normalized_lines.append(f'{"#" * heading_level} {line}')
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _normalize_generic_content(self, content: str, metadata: Dict[str, str]) -> str:
        """Generic content normalization."""
        lines = content.split('\n')
        normalized_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                normalized_lines.append('')
                continue
            
            # Convert title to H1 if it matches the metadata title
            if line == metadata.get('title', ''):
                normalized_lines.append(f'# {line}')
                continue
            
            # Basic heading detection
            if self._is_likely_heading(line) and not line.startswith('#'):
                normalized_lines.append(f'## {line}')
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _is_likely_heading(self, line: str) -> bool:
        """Determine if a line is likely a heading."""
        # Skip very long lines (likely paragraphs)
        if len(line) > 100:
            return False
        
        # Skip lines with common paragraph indicators
        if any(indicator in line.lower() for indicator in [
            'the ', 'this ', 'that ', 'these ', 'those ', 'a ', 'an ',
            'you can', 'to ', 'for ', 'with ', 'and ', 'or ', 'but '
        ]):
            return False
        
        # Lines that are likely headings
        if (
            # Short lines with title case
            len(line) < 80 and line.istitle() or
            # Lines ending with colons
            line.endswith(':') or
            # Lines that are all caps (but not too long)
            (line.isupper() and len(line) < 50) or
            # Lines with specific patterns
            any(pattern in line for pattern in ['Documentation', 'Overview', 'Installation', 'Usage', 'Configuration'])
        ):
            return True
        
        return False
    
    def _determine_heading_level(self, line: str, previous_lines: List[str]) -> int:
        """Determine the appropriate heading level for a line."""
        # Default to H2 for most headings
        default_level = 2
        
        # H1 is reserved for document titles
        if any(keyword in line for keyword in ['Documentation', 'Overview', 'Guide']):
            return default_level
        
        # H3 for subsections
        if any(keyword in line for keyword in ['Installation', 'Usage', 'Configuration', 'Requirements']):
            return 3
        
        return default_level
