from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Dict, Any
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from audio_loader import AudioLoader
from channel_manager import ChannelManager
from fourier_analyzer import FourierAnalyzer
from resampler import AudioResampler


class AudioAnalyzerGUI:
    """
    Główna klasa interfejsu graficznego integrująca moduły:
    - AudioLoader (wczytywanie)
    - ChannelManager (obsługa kanałów)
    - AudioResampler (próbkowanie)
    - FourierAnalyzer (FFT i egzekwowanie kryterium Nyquista)
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Analizator Dźwięku MP3 - Dowolne Próbkowanie & Wymóg Nyquista")
        self.root.geometry("1150x820")
        self.root.minsize(950, 650)

        # Stan danych
        self.raw_data: Optional[np.ndarray] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.current_channel_max_freq: float = 0.0

        # Zmienna do pola wpisywania częstotliwości próbkowania
        self.sample_rate_var = tk.StringVar(value="")

        # Referencje do wykresów (osobne obiekty Axes dla pełnej zgodności typów z IDE)
        self.ax_orig_time: Axes
        self.ax_res_time: Axes
        self.ax_orig_fft: Axes
        self.ax_res_fft: Axes

        self._setup_ui()

    def _setup_ui(self):
        """Inicjalizacja i ułożenie komponentów GUI."""
        control_frame = ttk.LabelFrame(self.root, text="Panel sterowania i parametry", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Wiersz 1: Przyciski plików, info i wyjście
        row1_frame = ttk.Frame(control_frame)
        row1_frame.pack(fill="x", pady=2)

        self.btn_load = ttk.Button(
            row1_frame, text="📁 Wczytaj plik (.mp3, .wav)", command=self._on_load_file
        )
        self.btn_load.pack(side="left", padx=5)

        self.lbl_file_info = ttk.Label(
            row1_frame,
            text="Brak wczytanego pliku. Kliknij 'Wczytaj plik', aby rozpocząć.",
            foreground="#555555",
        )
        self.lbl_file_info.pack(side="left", padx=10, fill="x", expand=True)

        # Przycisk Wyjdź
        self.btn_exit = ttk.Button(
            row1_frame, text="🚪 Wyjdź", command=self._on_exit
        )
        self.btn_exit.pack(side="right", padx=5)

        # Wiersz 2: Wybór kanału, DOWOLNE wpisywanie Fs i przycisk analizy
        row2_frame = ttk.Frame(control_frame)
        row2_frame.pack(fill="x", pady=8)

        ttk.Label(row2_frame, text="Kanał dźwięku:").pack(side="left", padx=5)
        self.combo_channel = ttk.Combobox(
            row2_frame, values=["Brak danych"], state="readonly", width=18
        )
        self.combo_channel.current(0)
        self.combo_channel.bind("<<ComboboxSelected>>", self._on_channel_changed)
        self.combo_channel.pack(side="left", padx=5)

        # Pole na DOWOLNĄ częstotliwość próbkowania (zamiast sztywnych opcji)
        ttk.Label(row2_frame, text="Częstotliwość próbkowania:").pack(side="left", padx=(20, 5))
        self.entry_sample_rate = ttk.Entry(
            row2_frame, textvariable=self.sample_rate_var, width=12, font=("Segoe UI", 9)
        )
        self.entry_sample_rate.pack(side="left", padx=2)
        ttk.Label(row2_frame, text="Hz").pack(side="left", padx=(2, 10))

        # Przycisk analizy
        self.btn_analyze = ttk.Button(
            row2_frame,
            text="⚡ Przetwórz i analizuj (FFT)",
            command=self._on_analyze,
            state="disabled",
        )
        self.btn_analyze.pack(side="left", padx=15)

        # Wiersz 3: Etykieta wymogu Nyquista i statusu
        self.nyquist_info_frame = ttk.Frame(control_frame)
        self.nyquist_info_frame.pack(fill="x", pady=(4, 0))

        self.lbl_nyquist_requirement = ttk.Label(
            self.nyquist_info_frame,
            text="Wymóg Nyquista: Wczytaj plik, aby sprawdzić minimalną dozwoloną częstotliwość próbkowania (2 × f_max).",
            foreground="#555555",
            font=("Segoe UI", 9),
        )
        self.lbl_nyquist_requirement.pack(anchor="w", padx=5)

        # Obszar wykresów Matplotlib (2x2)
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(10, 6), dpi=100)
        
        # Bezpośrednie rozpakowanie osi jako obiektów typu Axes
        axes_grid = self.fig.subplots(2, 2)
        self.ax_orig_time = axes_grid[0, 0]
        self.ax_res_time = axes_grid[0, 1]
        self.ax_orig_fft = axes_grid[1, 0]
        self.ax_res_fft = axes_grid[1, 1]

        self._init_empty_plots()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

    def _init_empty_plots(self):
        """Inicjalizacja pustych wykresów."""
        plots_info = [
            (self.ax_orig_time, "Dźwięk przed próbkowaniem (Dziedzina czasu)", "Czas [s]", "Amplituda"),
            (self.ax_res_time, "Dźwięk po próbkowaniu (Dziedzina czasu)", "Czas [s]", "Amplituda"),
            (self.ax_orig_fft, "Widmo FFT - Przed próbkowaniem", "Częstotliwość [Hz]", "Magnituda"),
            (self.ax_res_fft, "Widmo FFT - Po próbkowaniu", "Częstotliwość [Hz]", "Magnituda"),
        ]

        for ax, title, xlabel, ylabel in plots_info:
            ax.clear()
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)

        self.fig.subplots_adjust(hspace=0.45, wspace=0.25, left=0.07, right=0.98, top=0.93, bottom=0.1)

    def _on_load_file(self):
        """Wczytanie pliku audio."""
        filetypes = [
            ("Pliki audio", "*.mp3 *.wav *.ogg *.flac"),
            ("Pliki MP3", "*.mp3"),
            ("Pliki WAV", "*.wav"),
            ("Wszystkie pliki", "*.*"),
        ]
        filepath = filedialog.askopenfilename(
            title="Wybierz plik audio do analizy", filetypes=filetypes
        )
        if not filepath:
            return

        try:
            self.raw_data, self.metadata = AudioLoader.load(filepath)

            mins = int(self.metadata["duration"] // 60)
            secs = self.metadata["duration"] % 60
            info_text = (
                f"Plik: {self.metadata['filename']} | "
                f"Oryginalne Fs: {self.metadata['sample_rate']} Hz | "
                f"Kanałów: {self.metadata['channels']} | "
                f"Czas: {mins}:{secs:04.1f} | "
                f"Próbek: {self.metadata['samples']:,}"
            )
            self.lbl_file_info.config(text=info_text, foreground="#000000")

            # Ustawienie kanałów
            channel_options = ChannelManager.get_channel_options(self.metadata["channels"])
            self.combo_channel.config(values=channel_options)
            self.combo_channel.current(0)

            # Wpisanie domyślnej częstotliwości (równej oryginalnej)
            self.sample_rate_var.set(str(self.metadata["sample_rate"]))

            # Odblokowanie przycisku analizy i aktualizacja limitu Nyquista
            self.btn_analyze.config(state="normal")
            self._update_nyquist_limit()

        except Exception as e:
            messagebox.showerror("Błąd wczytywania", f"Nie udało się wczytać pliku audio:\n{str(e)}")

    def _on_channel_changed(self, _event: Any = None):
        """Wywoływane po zmianie kanału – aktualizuje wyliczony limit Nyquista."""
        self._update_nyquist_limit()

    def _update_nyquist_limit(self):
        """Oblicza maksymalną częstotliwość f_max dla aktualnego kanału i wyświetla wymagane Fs."""
        if self.raw_data is None or self.metadata is None or self.raw_data.size == 0:
            return

        try:
            selected_channel = self.combo_channel.get()
            orig_signal = ChannelManager.extract_channel(self.raw_data, selected_channel)
            orig_sr = self.metadata["sample_rate"]

            freqs_orig, mag_orig = FourierAnalyzer.compute_fft(orig_signal, orig_sr)
            self.current_channel_max_freq = FourierAnalyzer.find_highest_significant_frequency(
                freqs_orig, mag_orig
            )
            min_required_sr = int(np.ceil(2.0 * self.current_channel_max_freq))

            self.lbl_nyquist_requirement.config(
                text=(
                    f"🔒 Wymóg Nyquista: Maksymalna składowa sygnału = {self.current_channel_max_freq:.1f} Hz. "
                    f"Wprowadzana częstotliwość próbkowania MUSI wynosić co najmniej {min_required_sr} Hz."
                ),
                foreground="#0055aa",
            )
        except Exception as ex:
            self.lbl_nyquist_requirement.config(
                text=f"Nie udało się oszacować pasma: {ex}",
                foreground="#aa5500",
            )

    def _on_analyze(self):
        """Walidacja wymogu Nyquista i wykonanie analizy."""
        if self.raw_data is None or self.metadata is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik audio.")
            return

        # 1. Walidacja formatu wpisanej liczby
        raw_sr_str = self.sample_rate_var.get().strip()
        try:
            target_sr = int(raw_sr_str)
            if target_sr <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror(
                "Nieprawidłowa wartość",
                f"Wpisana częstotliwość próbkowania '{raw_sr_str}' nie jest prawidłową dodatnią liczbą całkowitą.\n"
                f"Wpisz np. 16000, 22050, 44100.",
            )
            return

        selected_channel = self.combo_channel.get()
        orig_sr = self.metadata["sample_rate"]

        try:
            # 2. Pobranie sygnału i FFT oryginału
            orig_signal = ChannelManager.extract_channel(self.raw_data, selected_channel)
            freqs_orig, mag_orig = FourierAnalyzer.compute_fft(orig_signal, orig_sr)
            max_significant_freq = FourierAnalyzer.find_highest_significant_frequency(freqs_orig, mag_orig)

            # 3. BLOKADA: Sprawdzenie spełnienia kryterium Nyquista
            is_valid, msg, min_required_sr = FourierAnalyzer.validate_nyquist_frequency(
                target_sr, max_significant_freq
            )

            if not is_valid:
                # Nie zezwalaj na złamanie wymogu Nyquista
                messagebox.showerror("Zablokowano próbkowanie (Wymóg Nyquista)", msg)
                self.lbl_nyquist_requirement.config(
                    text=f"⛔ BŁĄD NYQUISTA: Wartość {target_sr} Hz jest zbyt niska! Wymagane minimum to {min_required_sr} Hz.",
                    foreground="#cc0000",
                )
                return

            # Jeśli wymóg jest spełniony, informujemy użytkownika
            self.lbl_nyquist_requirement.config(
                text=f"✅ {msg}",
                foreground="#007700",
            )

            # 4. Wykonanie próbkowania (resampling)
            orig_time = np.arange(len(orig_signal)) / orig_sr
            resampled_signal, resampled_time = AudioResampler.resample(
                orig_signal, orig_sr, target_sr
            )

            # 5. FFT sygnału po próbkowaniu
            freqs_resamp, mag_resamp = FourierAnalyzer.compute_fft(resampled_signal, target_sr)

            # 6. Optymalizacja punktów do wykresów
            t_orig_p, s_orig_p = AudioResampler.downsample_for_plot(orig_time, orig_signal)
            t_res_p, s_res_p = AudioResampler.downsample_for_plot(resampled_time, resampled_signal)
            f_orig_p, m_orig_p = AudioResampler.downsample_for_plot(freqs_orig, mag_orig)
            f_res_p, m_res_p = AudioResampler.downsample_for_plot(freqs_resamp, mag_resamp)

            # 7. Rysowanie
            # Wykres 1: Czas przed
            self.ax_orig_time.clear()
            self.ax_orig_time.plot(t_orig_p, s_orig_p, color="#1f77b4", linewidth=0.8)
            self.ax_orig_time.set_title(
                f"Przed próbkowaniem - Czas (Fs = {orig_sr} Hz, {selected_channel})",
                fontsize=9,
                fontweight="bold",
            )
            self.ax_orig_time.set_xlabel("Czas [s]", fontsize=8)
            self.ax_orig_time.set_ylabel("Amplituda", fontsize=8)
            self.ax_orig_time.grid(True, linestyle="--", alpha=0.5)

            # Wykres 2: Czas po próbkowaniu
            self.ax_res_time.clear()
            self.ax_res_time.plot(t_res_p, s_res_p, color="#d62728", linewidth=0.8)
            self.ax_res_time.set_title(
                f"Po próbkowaniu - Czas (Nowe Fs = {target_sr} Hz)",
                fontsize=9,
                fontweight="bold",
            )
            self.ax_res_time.set_xlabel("Czas [s]", fontsize=8)
            self.ax_res_time.set_ylabel("Amplituda", fontsize=8)
            self.ax_res_time.grid(True, linestyle="--", alpha=0.5)

            # Wykres 3: FFT przed
            orig_nyquist = orig_sr / 2
            self.ax_orig_fft.clear()
            self.ax_orig_fft.plot(f_orig_p, m_orig_p, color="#1f77b4", linewidth=0.8, label="Widmo sygnału")
            self.ax_orig_fft.axvline(
                orig_nyquist,
                color="purple",
                linestyle="--",
                linewidth=1.2,
                label=f"Nyquist ({orig_nyquist:.0f} Hz)",
            )
            self.ax_orig_fft.set_title(
                f"Przed próbkowaniem - Widmo FFT (Pasmo do {orig_nyquist:.0f} Hz)",
                fontsize=9,
                fontweight="bold",
            )
            self.ax_orig_fft.set_xlabel("Częstotliwość [Hz]", fontsize=8)
            self.ax_orig_fft.set_ylabel("Magnituda", fontsize=8)
            self.ax_orig_fft.grid(True, linestyle="--", alpha=0.5)
            self.ax_orig_fft.set_xlim(left=0, right=orig_nyquist)
            self.ax_orig_fft.legend(loc="upper right", fontsize=7)

            # Wykres 4: FFT po próbkowaniu
            target_nyquist = target_sr / 2
            self.ax_res_fft.clear()
            self.ax_res_fft.plot(f_res_p, m_res_p, color="#d62728", linewidth=0.8, label="Widmo sygnału")
            self.ax_res_fft.axvline(
                target_nyquist,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"Nyquist Nowe ({target_nyquist:.0f} Hz)",
            )
            self.ax_res_fft.set_title(
                f"Po próbkowaniu - Widmo FFT (Granica Nyquista: {target_nyquist:.0f} Hz)",
                fontsize=9,
                fontweight="bold",
            )
            self.ax_res_fft.set_xlabel("Częstotliwość [Hz]", fontsize=8)
            self.ax_res_fft.set_ylabel("Magnituda", fontsize=8)
            self.ax_res_fft.grid(True, linestyle="--", alpha=0.5)
            self.ax_res_fft.set_xlim(left=0, right=target_nyquist)
            self.ax_res_fft.legend(loc="upper right", fontsize=7)

            self.fig.subplots_adjust(hspace=0.45, wspace=0.25, left=0.07, right=0.98, top=0.93, bottom=0.1)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Błąd analizy", f"Wystąpił błąd podczas analizy:\n{str(e)}")

    def _on_exit(self):
        """Obsługa przycisku Wyjdź z potwierdzeniem."""
        if messagebox.askyesno("Wyjście", "Czy na pewno chcesz zamknąć aplikację?"):
            self.root.destroy()
