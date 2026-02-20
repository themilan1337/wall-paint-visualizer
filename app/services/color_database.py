"""Color database service with Redis caching"""
import json
import os
from typing import List, Dict, Optional
from app.core.config import get_settings
from app.core.cache import cache, cached

settings = get_settings()


class ColorDatabase:
    """
    Fast color database with:
    - Search by code (F001, F002, etc.)
    - Partial search support
    - Redis caching for performance
    """

    def __init__(self):
        self.colors: Dict = {}
        self.codes_list: List[str] = []
        self.loaded = False

    def load(self):
        """Load colors from JSON file"""
        if self.loaded:
            return

        colors_file = settings.colors_file
        if not os.path.exists(colors_file):
            raise FileNotFoundError(f"Colors file not found: {colors_file}")

        print(f"📚 Loading colors from {colors_file}...")
        with open(colors_file, 'r', encoding='utf-8') as f:
            self.colors = json.load(f)

        # Create sorted list of codes for fast search
        self.codes_list = sorted(self.colors.keys())
        self.loaded = True
        print(f"✓ Loaded {len(self.colors)} colors")

    @cached(ttl=3600, key_prefix="color_search")
    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search colors by code (cached)

        Args:
            query: Search query (e.g., "F00" finds F001, F002, etc.)
            limit: Maximum results

        Returns:
            List of matching colors with data
        """
        if not self.loaded:
            self.load()

        query = query.upper().strip()
        if not query:
            # Return first N colors if query is empty
            return [
                {**self.colors[code], 'code': code}
                for code in self.codes_list[:limit]
            ]

        results = []
        for code in self.codes_list:
            if code.startswith(query) or query in code:
                results.append({
                    **self.colors[code],
                    'code': code
                })
                if len(results) >= limit:
                    break

        return results

    @cached(ttl=3600, key_prefix="color_code")
    def get_by_code(self, code: str) -> Optional[Dict]:
        """
        Get color by exact code (cached)

        Args:
            code: Color code (e.g., "F001")

        Returns:
            Color data or None if not found
        """
        if not self.loaded:
            self.load()

        code = code.upper().strip()
        if code in self.colors:
            return {**self.colors[code], 'code': code}
        return None

    def get_all_codes(self) -> List[str]:
        """Return all available color codes"""
        if not self.loaded:
            self.load()
        return self.codes_list.copy()

    def get_count(self) -> int:
        """Return total number of colors"""
        if not self.loaded:
            self.load()
        return len(self.colors)


# Global instance
color_db = ColorDatabase()
