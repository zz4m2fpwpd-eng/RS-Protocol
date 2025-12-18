#!/usr/bin/env python3
"""
ROBIN Protocol - System 3 Master Index

Master indexing system for data organization and retrieval.
"""

import logging
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


class MasterIndex:
    """Master indexing system for efficient data organization."""
    
    def __init__(self, config=None):
        """Initialize the Master Index system."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.index_type = self.config.get('index_type', 'hash')
        self.indices = defaultdict(dict)
        self.metadata = {
            'created_at': datetime.now(),
            'index_type': self.index_type,
            'version': '1.0.0'
        }
    
    def create_index(self, index_name, data):
        """
        Create a new index.
        
        Args:
            index_name: Name of the index
            data: Data to index
            
        Returns:
            dict: Index metadata
        """
        self.logger.info(f"Creating index: {index_name}")
        
        index_data = {
            'name': index_name,
            'created_at': datetime.now(),
            'size': len(data) if hasattr(data, '__len__') else 1,
            'entries': {}
        }
        
        # Create hash-based index
        if self.index_type == 'hash':
            for i, item in enumerate(data if isinstance(data, list) else [data]):
                key = f"{index_name}_{i}"
                index_data['entries'][key] = item
        
        self.indices[index_name] = index_data
        self.logger.info(f"Index created: {index_name} with {len(index_data['entries'])} entries")
        
        return index_data
    
    def add_to_index(self, index_name, key, value):
        """
        Add an entry to an existing index.
        
        Args:
            index_name: Name of the index
            key: Entry key
            value: Entry value
        """
        if index_name not in self.indices:
            self.logger.warning(f"Index {index_name} not found, creating new index")
            self.indices[index_name] = {
                'name': index_name,
                'created_at': datetime.now(),
                'entries': {}
            }
        
        self.indices[index_name]['entries'][key] = value
        self.logger.info(f"Added entry to index {index_name}: {key}")
    
    def get_from_index(self, index_name, key):
        """
        Retrieve an entry from an index.
        
        Args:
            index_name: Name of the index
            key: Entry key
            
        Returns:
            Entry value or None
        """
        if index_name not in self.indices:
            self.logger.warning(f"Index {index_name} not found")
            return None
        
        return self.indices[index_name]['entries'].get(key)
    
    def search_index(self, index_name, query):
        """
        Search within an index.
        
        Args:
            index_name: Name of the index
            query: Search query
            
        Returns:
            list: Matching entries
        """
        self.logger.info(f"Searching index {index_name} for: {query}")
        
        if index_name not in self.indices:
            return []
        
        results = []
        for key, value in self.indices[index_name]['entries'].items():
            if query in str(key) or query in str(value):
                results.append({'key': key, 'value': value})
        
        self.logger.info(f"Found {len(results)} matches")
        return results
    
    def get_index_stats(self, index_name=None):
        """
        Get statistics about indices.
        
        Args:
            index_name: Optional specific index name
            
        Returns:
            dict: Index statistics
        """
        if index_name:
            if index_name not in self.indices:
                return {'error': f'Index {index_name} not found'}
            
            idx = self.indices[index_name]
            return {
                'name': idx['name'],
                'created_at': idx['created_at'],
                'size': len(idx['entries'])
            }
        else:
            return {
                'total_indices': len(self.indices),
                'index_names': list(self.indices.keys()),
                'total_entries': sum(len(idx['entries']) for idx in self.indices.values())
            }
    
    def save_index(self, index_name, filepath):
        """
        Save an index to file.
        
        Args:
            index_name: Name of the index to save
            filepath: Path to save file
        """
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")
        
        self.logger.info(f"Saving index {index_name} to {filepath}")
        
        with open(filepath, 'w') as f:
            json.dump(self.indices[index_name], f, indent=2, default=str)
        
        self.logger.info(f"Index saved successfully")
    
    def load_index(self, index_name, filepath):
        """
        Load an index from file.
        
        Args:
            index_name: Name for the loaded index
            filepath: Path to load from
        """
        self.logger.info(f"Loading index from {filepath}")
        
        with open(filepath, 'r') as f:
            index_data = json.load(f)
        
        self.indices[index_name] = index_data
        self.logger.info(f"Index {index_name} loaded successfully")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    master_index = MasterIndex()
    print("Master Index system initialized")
