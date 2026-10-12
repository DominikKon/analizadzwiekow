import os
from typing import Dict, Any, Tuple
import numpy as np
import soundfile as sf


class AudioLoader:
    """
    Klasa odpowiedzialna za bezpieczne wczytywanie plików audio (.mp3, .wav, .flac itp.)
    oraz odczytywanie ich parametrów.
    """

    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Wczytuje plik audio za pomocą biblioteki soundfile.
        Zwraca: (dane_audio_jako_numpy_array, słownik_metadanych).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Plik nie istnieje: {filepath}")

        # Wczytanie z wymuszeniem tablicy 2D: (próbki, kanały)
        data, sample_rate = sf.read(filepath, dtype="float32", always_2d=True)
        sr = int(sample_rate)
        channels = data.shape[1]
        samples = len(data)
        duration = samples / sr if sr > 0 else 0.0

        metadata = {
            "filename": os.path.basename(filepath),
            "filepath": filepath,
            "sample_rate": sr,
            "channels": channels,
            "samples": samples,
            "duration": duration,
        }

        return data, metadata

