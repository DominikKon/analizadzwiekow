from typing import Tuple, Dict, Any
import numpy as np


class FourierAnalyzer:
    """
    Klasa odpowiedzialna za analizę widmową (Transformata Fouriera / FFT)
    oraz weryfikację kryterium Nyquista-Shannona.
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

        # Aplikacja okna czasowego (Hanning)
        window = np.hanning(n_samples)
        windowed_signal = audio_signal * window

        # Szybka transformata Fouriera
        fft_values = np.fft.rfft(windowed_signal)
        frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

        # Skalowanie magnitugy do wartości amplitudy rzeczywistej
        magnitude = np.abs(fft_values) * (2.0 / n_samples)

        return frequencies, magnitude

    @staticmethod
    def find_highest_significant_frequency(
        frequencies: np.ndarray, magnitude: np.ndarray, threshold_ratio: float = 0.03
    ) -> float:
        """
        Wyszukuje najwyższą istotną częstotliwość w widmie, której magnituda
        przekracza zdefiniowany ułamek maksymalnej amplitudy (domyślnie 3%).
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
    def evaluate_nyquist_criterion(
        target_sr: int, max_present_frequency: float
    ) -> Dict[str, Any]:
        """
        Ocena spełnienia kryterium Nyquista-Shannona dla wybranej częstotliwości próbkowania:
        Kryterium: f_s >= 2 * f_max (częstotliwość Nyquista f_N = f_s / 2 >= f_max).
        """
        nyquist_limit = target_sr / 2.0
        required_sr = 2.0 * max_present_frequency
        is_satisfied = nyquist_limit >= max_present_frequency

        if is_satisfied:
            clause = (
                f"✅ KLAUZULA NYQUISTA SPEŁNIONA:\n"
                f"Częstotliwość próbkowania Fs = {target_sr} Hz zapewnia granicę Nyquista f_N = {nyquist_limit:.1f} Hz. "
                f"Jest ona większa niż maksymalna wykryta składowa sygnału ({max_present_frequency:.1f} Hz). "
                f"Brak zjawiska aliasingu (próbkowanie bezstratne pasmowo)."
            )
            status_level = "success"
        else:
            loss_hz = max_present_frequency - nyquist_limit
            clause = (
                f"⚠️ OSTRZEŻENIE - NARUSZENIE KRYTERIUM NYQUISTA:\n"
                f"Wybrana częstotliwość Fs = {target_sr} Hz ustala granicę Nyquista na poziomie {nyquist_limit:.1f} Hz, "
                f"podczas gdy w sygnale występują składowe do {max_present_frequency:.1f} Hz.\n"
                f"Składowe w przedziale {nyquist_limit:.1f} - {max_present_frequency:.1f} Hz (~{loss_hz:.1f} Hz pasma) "
                f"zostaną odcięte lub zniekształcone przez aliasing. "
                f"Zalecane minimalne Fs: {int(np.ceil(required_sr))} Hz."
            )
            status_level = "warning"

        return {
            "is_satisfied": is_satisfied,
            "nyquist_limit": nyquist_limit,
            "max_present_freq": max_present_frequency,
            "required_sr": required_sr,
            "clause": clause,
            "status_level": status_level,
        }

