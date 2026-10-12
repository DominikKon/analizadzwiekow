import numpy as np
from typing import List


class ChannelManager:
    """
    Klasa odpowiedzialna za zarządzanie i separację kanałów audio (Mono, Lewy, Prawy, Miks).
    """

    @staticmethod
    def get_channel_options(channels_count: int) -> List[str]:
        """
        Zwraca listę czytelnych dla użytkownika opcji wyboru kanału
        na podstawie liczby kanałów w pliku audio.
        """
        if channels_count <= 0:
            return ["Brak danych"]
        if channels_count == 1:
            return ["Kanał 1 (Mono)"]
        if channels_count == 2:
            return ["Miks (Mono)", "Kanał 1 (Lewy)", "Kanał 2 (Prawy)"]
        
        # Dla plików wielokanałowych (> 2)
        options = ["Miks (Wszystkie)"]
        for i in range(1, channels_count + 1):
            options.append(f"Kanał {i}")
        return options

    @staticmethod
    def extract_channel(raw_data: np.ndarray, selection: str) -> np.ndarray:
        """
        Wyodrębnia wybrany kanał lub oblicza miks mono ze wszystkich kanałów.
        """
        if raw_data is None or raw_data.size == 0:
            raise ValueError("Brak danych audio do wyodrębnienia kanału.")

        # Jeśli sygnał jest jednokanałowy (1D lub (N, 1))
        if raw_data.ndim == 1:
            return raw_data
        if raw_data.shape[1] == 1:
            return raw_data[:, 0]

        # Wybór na podstawie tekstu z menu
        if "Kanał 1" in selection or "Lewy" in selection:
            return raw_data[:, 0]
        elif "Kanał 2" in selection or "Prawy" in selection:
            return raw_data[:, 1]
        elif selection.startswith("Kanał "):
            try:
                ch_idx = int(selection.replace("Kanał ", "").strip()) - 1
                if 0 <= ch_idx < raw_data.shape[1]:
                    return raw_data[:, ch_idx]
            except ValueError:
                pass

        # Domyślnie średnia ze wszystkich kanałów (miks mono)
        return np.mean(raw_data, axis=1)

