from __future__ import annotations

from typing import Tuple
import numpy as np


class FourierAnalyzer:
    """
    Klasa odpowiedzialna za analizę widmową (Transformata Fouriera / FFT)
    oraz wymuszenie i walidację kryterium Nyquista-Shannona.
    """

    @staticmethod
    def compute_fft(
        audio_signal: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oblicza jednoronną transformatę Fouriera (rFFT) dla sygnału rzeczywistego.
        Wykorzystuje okno Hanna, aby zminimalizować efekt przecieku widma.

        Zwraca:
            (częstotliwości [Hz], magnituda [amplituda])
        """
        n_samples = len(audio_signal)
        if n_samples == 0:
            return np.array([]), np.array([])

        window = np.hanning(n_samples)
        windowed_signal = audio_signal * window

        fft_values = np.fft.rfft(windowed_signal)
        frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

        # Skalowanie magnitugy do wartości amplitudy rzeczywistej
        magnitude = np.abs(fft_values) * (2.0 / n_samples)

        return frequencies, magnitude

    @staticmethod
    def find_highest_significant_frequency(
        frequencies: np.ndarray, magnitude: np.ndarray, threshold_ratio: float = 0.02
    ) -> float:
        """
        Wyszukuje najwyższą istotną częstotliwość w widmie, której magnituda
        przekracza próg istotności (domyślnie 2% maksymalnej amplitudy).
        """
        if len(magnitude) == 0 or np.max(magnitude) == 0:
            return 0.0

        max_mag = np.max(magnitude)
        threshold = max_mag * threshold_ratio

        significant_indices = np.where(magnitude >= threshold)[0]
        if len(significant_indices) == 0:
            return float(frequencies[-1])

        highest_idx = significant_indices[-1]
        return float(frequencies[highest_idx])

    @staticmethod
    def validate_nyquist_frequency(
        target_sr: int, max_present_frequency: float
    ) -> Tuple[bool, str, int]:
        """
        Ścisła walidacja częstotliwości próbkowania:
        Nie pozwala na wartość mniejszą niż dwukrotność najwyższej częstotliwości (f_s >= 2 * f_max).

        Zwraca:
            (czy_dozwolone: bool, komunikat: str, minimalne_wymagane_fs: int)
        """
        min_required_sr = int(np.ceil(2.0 * max_present_frequency))
        if min_required_sr <= 0:
            min_required_sr = 1

        if target_sr < min_required_sr:
            error_msg = (
                f"Zablokowano: Częstotliwość próbkowania {target_sr} Hz łamie kryterium Nyquista!\n\n"
                f"• Najwyższa wykryta składowa w sygnale: {max_present_frequency:.1f} Hz\n"
                f"• Wymagana minimalna częstotliwość (2 × f_max): {min_required_sr} Hz\n\n"
                f"Zwiększ wartość częstotliwości próbkowania do co najmniej {min_required_sr} Hz."
            )
            return False, error_msg, min_required_sr

        success_msg = (
            f"Wartość {target_sr} Hz jest poprawna i spełnia kryterium Nyquista "
            f"(wymagane min. {min_required_sr} Hz)."
        )
        return True, success_msg, min_required_sr
