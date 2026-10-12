import math
from typing import Tuple
import numpy as np
import scipy.signal as signal


class AudioResampler:
    """
    Klasa odpowiedzialna za konwersję częstotliwości próbkowania (resampling)
    oraz optymalizację punktów danych pod kątem szybkiego renderowania wykresów.
    """

    @staticmethod
    def resample(
        audio_signal: np.ndarray, orig_sr: int, target_sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zmienia częstotliwość próbkowania jednokanałowego sygnału z orig_sr na target_sr.
        Zwraca: (próbkowany_sygnał, wektor_czasu_w_sekundach).
        """
        if orig_sr <= 0 or target_sr <= 0:
            raise ValueError("Częstotliwość próbkowania musi być większa od zera.")

        if orig_sr == target_sr:
            time_vector = np.arange(len(audio_signal)) / target_sr
            return audio_signal.copy(), time_vector

        # Wyznaczenie up i down przez NWD (największy wspólny dzielnik)
        gcd = math.gcd(int(target_sr), int(orig_sr))
        up = int(target_sr) // gcd
        down = int(orig_sr) // gcd

        # Resampling wielofazowy z filtrem antyaliasingowym
        if up <= 2000 and down <= 2000:
            resampled = signal.resample_poly(audio_signal, up, down).astype(np.float32)
        else:
            # Fallback w przypadku nietypowych wartości częstotliwości
            num_samples = int(round(len(audio_signal) * (target_sr / orig_sr)))
            resampled = signal.resample(audio_signal, num_samples).astype(np.float32)

        time_vector = np.arange(len(resampled)) / target_sr
        return resampled, time_vector

    @staticmethod
    def downsample_for_plot(
        x: np.ndarray, y: np.ndarray, max_points: int = 15000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pomocnicza redukcja liczby punktów na wykresie dla zachowania płynności UI.
        """
        if len(x) <= max_points:
            return x, y
        step = len(x) // max_points
        return x[::step], y[::step]

