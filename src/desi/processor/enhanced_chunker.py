#!/usr/bin/env python3
"""
Enhanced Chunker for Multi-Source Documentation

This module provides intelligent chunking capabilities adapted from the
all_repos_chunking.py logic but enhanced for multiple documentation sources.
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class EnhancedChunker:
    """Enhanced chunker that maintains context and handles multiple sources."""
    
    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 50
    ):
        """
        Initialize the enhanced chunker.
        
        Args:
            min_chunk_size: Minimum size of a chunk in characters
            max_chunk_size: Maximum size of a chunk in characters  
            chunk_overlap: Overlap between chunks in characters
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_content(self, content: str, source: str = '') -> List[str]:
        """
        Chunk content using intelligent heading-aware strategy.
        
        Args:
            content: The normalized markdown content to chunk
            source: The source type (for source-specific optimizations)
            
        Returns:
            List of content chunks
        """
        if not content.strip():
            return []
        
        # Split content by double newlines to get paragraphs
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return []
        
        # Apply source-specific chunking strategy
        if source.lower() in ['openbis']:
            return self._chunk_technical_content(paragraphs)
        elif source.lower() in ['datastore', 'wikijs']:
            return self._chunk_procedural_content(paragraphs)
        else:
            return self._chunk_generic_content(paragraphs)
    
    def _chunk_technical_content(self, paragraphs: List[str]) -> List[str]:
        """Chunk technical documentation (OpenBIS style)."""
        chunks = []
        current_chunk = ""
        section_heading = None
        subsection_heading = None
        
        for paragraph in paragraphs:
            p = paragraph.strip()
            
            # Identify heading levels
            is_main_heading = p.startswith('# ')
            is_section_heading = p.startswith('## ')
            is_subsection_heading = p.startswith('### ')
            is_any_heading = is_main_heading or is_section_heading or is_subsection_heading
            
            # Handle main and section headings
            if is_main_heading or is_section_heading:
                # Save current chunk if it meets minimum size
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with this heading
                current_chunk = paragraph
                section_heading = paragraph
                subsection_heading = None
                continue
            
            # Handle subsection headings
            if is_subsection_heading:
                subsection_heading = paragraph
                
                # If current chunk is getting large, start a new one
                if len(current_chunk) >= self.max_chunk_size * 0.7:
                    if current_chunk and len(current_chunk) >= self.min_chunk_size:
                        chunks.append(current_chunk.strip())
                    
                    # Start new chunk with context
                    if section_heading and section_heading != subsection_heading:
                        current_chunk = section_heading + '\n\n' + subsection_heading
                    else:
                        current_chunk = subsection_heading
                else:
                    # Add subsection to current chunk
                    current_chunk = (current_chunk + '\n\n' + subsection_heading) if current_chunk else subsection_heading
                continue
            
            # Handle regular content
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with context from headings
                if subsection_heading:
                    if section_heading and section_heading != subsection_heading:
                        current_chunk = section_heading + '\n\n' + subsection_heading + '\n\n'
                    else:
                        current_chunk = (subsection_heading or '') + '\n\n'
                elif section_heading:
                    current_chunk = section_heading + '\n\n'
                else:
                    current_chunk = ""
            
            # Add paragraph to current chunk
            current_chunk = (current_chunk + '\n\n' + paragraph) if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_procedural_content(self, paragraphs: List[str]) -> List[str]:
        """Chunk procedural/how-to content (Wiki.js style)."""
        chunks = []
        current_chunk = ""
        current_heading = None
        
        for paragraph in paragraphs:
            p = paragraph.strip()
            
            # Check if this is a heading
            is_heading = p.startswith('#')
            
            # For procedural content, we want to keep steps together
            if is_heading:
                # Save current chunk if it exists and meets minimum size
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with heading
                current_chunk = paragraph
                current_heading = paragraph
                continue
            
            # Check if this looks like a step or instruction
            is_step = self._is_procedural_step(p)
            
            # If adding this paragraph would exceed max size
            if len(current_chunk) + len(paragraph) > self.max_chunk_size:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with heading context if available
                    if current_heading:
                        current_chunk = current_heading + '\n\n' + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Current chunk is too small, just add the paragraph
                    current_chunk = (current_chunk + '\n\n' + paragraph) if current_chunk else paragraph
            else:
                # Add paragraph to current chunk
                current_chunk = (current_chunk + '\n\n' + paragraph) if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_generic_content(self, paragraphs: List[str]) -> List[str]:
        """Generic chunking strategy."""
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size
            if len(current_chunk) + len(paragraph) > self.max_chunk_size:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk = (current_chunk + '\n\n' + paragraph) if current_chunk else paragraph
            else:
                current_chunk = (current_chunk + '\n\n' + paragraph) if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _is_procedural_step(self, text: str) -> bool:
        """Check if text looks like a procedural step."""
        step_indicators = [
            'click', 'select', 'enter', 'navigate', 'open', 'save',
            'step', 'first', 'next', 'then', 'finally',
            'to register', 'to create', 'to upload', 'to download'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in step_indicators)
    
    def split_by_headings(self, content: str) -> List[Tuple[str, str]]:
        """
        Split content by headings, similar to all_repos_chunking.py.
        
        Args:
            content: The markdown content
            
        Returns:
            List of (heading, content) tuples
        """
        lines = content.splitlines()
        sections = []
        current_title = "Introduction"
        current_content = []
        
        for line in lines:
            # Check for markdown headings
            heading_match = re.match(r'^(#{1,6})\s+(.*)$', line)
            if heading_match:
                # Save previous section
                if current_content:
                    text = '\n'.join(current_content).strip()
                    if text:
                        sections.append((current_title, text))
                    current_content = []
                
                # Start new section
                current_title = heading_match.group(2).strip()
            else:
                current_content.append(line)
        
        # Add final section
        if current_content:
            text = '\n'.join(current_content).strip()
            if text:
                sections.append((current_title, text))
        
        return sections
