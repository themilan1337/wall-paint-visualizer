import json
import os
from typing import List, Dict, Optional

class ColorDatabase:
    """
    Быстрая база данных цветов с поддержкой:
    - Поиска по коду (F001, F002, etc.)
    - Поиска по части кода
    - Кэширования для максимальной производительности
    """

    def __init__(self, colors_file: str = 'extract/colors.json'):
        self.colors_file = colors_file
        self.colors: Dict = {}
        self.codes_list: List[str] = []
        self.loaded = False

    def load(self):
        """Загружает цвета из JSON файла"""
        if self.loaded:
            return

        if not os.path.exists(self.colors_file):
            raise FileNotFoundError(f"Colors file not found: {self.colors_file}")

        print(f"Loading colors from {self.colors_file}...")
        with open(self.colors_file, 'r', encoding='utf-8') as f:
            self.colors = json.load(f)

        # Создаем отсортированный список кодов для быстрого поиска
        self.codes_list = sorted(self.colors.keys())
        self.loaded = True
        print(f"Loaded {len(self.colors)} colors")

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Поиск цветов по коду

        Args:
            query: Поисковый запрос (например "F00" найдет F001, F002, etc.)
            limit: Максимальное количество результатов

        Returns:
            Список найденных цветов с их данными
        """
        if not self.loaded:
            self.load()

        query = query.upper().strip()
        if not query:
            # Возвращаем первые N цветов если запрос пустой
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

    def get_by_code(self, code: str) -> Optional[Dict]:
        """
        Получить цвет по точному коду

        Args:
            code: Код цвета (например "F001")

        Returns:
            Данные цвета или None если не найден
        """
        if not self.loaded:
            self.load()

        code = code.upper().strip()
        if code in self.colors:
            return {**self.colors[code], 'code': code}
        return None

    def get_all_codes(self) -> List[str]:
        """Возвращает все доступные коды цветов"""
        if not self.loaded:
            self.load()
        return self.codes_list.copy()

    def get_count(self) -> int:
        """Возвращает общее количество цветов"""
        if not self.loaded:
            self.load()
        return len(self.colors)


# Глобальный экземпляр для использования в приложении
color_db = ColorDatabase()
