import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
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
    - FourierAnalyzer (FFT i kryterium Nyquista)
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Analizator Dźwięku MP3 - Fourier & Kryterium Nyquista")
        self.root.geometry("1150x820")
        self.root.minsize(950, 650)

        # Stan danych
        self.raw_data: np.ndarray | None = None
        self.metadata: dict | None = None

        self._setup_ui()

    def _setup_ui(self):
        """Inicjalizacja i ułożenie komponentów GUI."""
        # 1. Górny panel sterowania
        control_frame = ttk.LabelFrame(self.root, text="Panel sterowania i parametry", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        # Wiersz 1: Przyciski plików i zamknięcia
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

        # Przycisk WYJDŹ
        self.btn_exit = ttk.Button(
            row1_frame, text="🚪 Wyjdź", command=self._on_exit
        )
        self.btn_exit.pack(side="right", padx=5)

        # Wiersz 2: Wybór kanału, docelowego Fs i uruchomienie analizy
        row2_frame = ttk.Frame(control_frame)
        row2_frame.pack(fill="x", pady=6)

        ttk.Label(row2_frame, text="Kanał dźwięku:").pack(side="left", padx=5)
        self.combo_channel = ttk.Combobox(
            row2_frame, values=["Brak danych"], state="readonly", width=18
        )
        self.combo_channel.current(0)
        self.combo_channel.pack(side="left", padx=5)

        ttk.Label(row2_frame, text="Docelowa częstotliwość próbkowania:").pack(
            side="left", padx=(20, 5)
        )
        self.combo_sample_rate = ttk.Combobox(
            row2_frame,
            values=["8000", "11025", "16000", "22050", "32000", "44100", "48000"],
            width=12,
        )
        self.combo_sample_rate.set("16000")
        self.combo_sample_rate.pack(side="left", padx=5)
        ttk.Label(row2_frame, text="Hz").pack(side="left", padx=(2, 10))

        self.btn_analyze = ttk.Button(
            row2_frame,
            text="⚡ Przetwórz i analizuj (FFT)",
            command=self._on_analyze,
            state="disabled",
        )
        self.btn_analyze.pack(side="left", padx=20)

        # Wiersz 3: Sekcja z KLAUZULĄ KRYTERIUM NYQUISTA
        self.nyquist_frame = ttk.LabelFrame(
            control_frame, text="Weryfikacja kryterium Nyquista-Shannona", padding=8
        )
        self.nyquist_frame.pack(fill="x", pady=(5, 2))

        self.lbl_nyquist = ttk.Label(
            self.nyquist_frame,
            text="Wczytaj plik i uruchom analizę, aby sprawdzić spełnienie kryterium Nyquista dla wybranego próbkowania.",
            foreground="#666666",
            wraplength=1050,
            justify="left",
            font=("Segoe UI", 9),
        )
        self.lbl_nyquist.pack(anchor="w", padx=5)

        # 2. Obszar wykresów Matplotlib (2x2)
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.axes = self.fig.subplots(2, 2)
        self._init_empty_plots()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

    def _init_empty_plots(self):
        """Ustawienie pustych wykresów początkowych z opisami osi."""
        titles = [
            ("Dźwięk przed próbkowaniem (Dziedzina czasu)", "Czas [s]", "Amplituda"),
            ("Dźwięk po próbkowaniu (Dziedzina czasu)", "Czas [s]", "Amplituda"),
            ("Widmo FFT - Przed próbkowaniem", "Częstotliwość [Hz]", "Magnituda"),
            ("Widmo FFT - Po próbkowaniu", "Częstotliwość [Hz]", "Magnituda"),
        ]

        plot_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for (row, col), (title, xlabel, ylabel) in zip(plot_coords, titles):
            ax = self.axes[row, col]
            ax.clear()
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)

        self.fig.subplots_adjust(hspace=0.45, wspace=0.25, left=0.07, right=0.98, top=0.93, bottom=0.1)

    def _on_load_file(self):
        """Wczytanie pliku przez AudioLoader i przygotowanie opcji kanałów."""
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

            # Aktualizacja informacji o pliku
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

            # Ustawienie kanałów w ChannelManager
            channel_options = ChannelManager.get_channel_options(self.metadata["channels"])
            self.combo_channel.config(values=channel_options)
            self.combo_channel.current(0)

            # Sugerowane Fs
            self.combo_sample_rate.set(str(self.metadata["sample_rate"]))

            self.btn_analyze.config(state="normal")
            self.lbl_nyquist.config(
                text="Plik załadowany. Wybierz kanał oraz docelowe Fs i kliknij 'Przetwórz i analizuj (FFT)', aby ocenić kryterium Nyquista.",
                foreground="#0055cc",
            )

        except Exception as e:
            messagebox.showerror("Błąd wczytywania", f"Nie udało się wczytać pliku audio:\n{str(e)}")

    def _on_analyze(self):
        """Główna akcja przetwarzania: ekstrakcja, resampling, FFT i sprawdzenie Nyquista."""
        if self.raw_data is None or self.metadata is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik audio.")
            return

        try:
            target_sr = int(self.combo_sample_rate.get().strip())
            if target_sr <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror(
                "Błąd", "Częstotliwość próbkowania musi być dodatnią liczbą całkowitą (np. 16000)."
            )
            return

        selected_channel = self.combo_channel.get()
        orig_sr = self.metadata["sample_rate"]

        try:
            # 1. Separacja kanału przez ChannelManager
            orig_signal = ChannelManager.extract_channel(self.raw_data, selected_channel)
            orig_time = np.arange(len(orig_signal)) / orig_sr

            # 2. Resampling przez AudioResampler
            resampled_signal, resampled_time = AudioResampler.resample(
                orig_signal, orig_sr, target_sr
            )

            # 3. Transformata Fouriera przez FourierAnalyzer
            freqs_orig, mag_orig = FourierAnalyzer.compute_fft(orig_signal, orig_sr)
            freqs_resamp, mag_resamp = FourierAnalyzer.compute_fft(resampled_signal, target_sr)

            # 4. Ocena KRYTERIUM NYQUISTA
            max_significant_freq = FourierAnalyzer.find_highest_significant_frequency(freqs_orig, mag_orig)
            nyquist_eval = FourierAnalyzer.evaluate_nyquist_criterion(target_sr, max_significant_freq)

            # Prezentacja klauzuli Nyquista w GUI
            self.lbl_nyquist.config(
                text=nyquist_eval["clause"],
                foreground="#007700" if nyquist_eval["is_satisfied"] else "#aa2200",
                font=("Segoe UI", 9, "bold" if not nyquist_eval["is_satisfied"] else "normal"),
            )

            # 5. Przygotowanie danych do rysowania (downsampling dla responsywności)
            t_orig_p, s_orig_p = AudioResampler.downsample_for_plot(orig_time, orig_signal)
            t_res_p, s_res_p = AudioResampler.downsample_for_plot(resampled_time, resampled_signal)
            f_orig_p, m_orig_p = AudioResampler.downsample_for_plot(freqs_orig, mag_orig)
            f_res_p, m_res_p = AudioResampler.downsample_for_plot(freqs_resamp, mag_resamp)

            # 6. Rysowanie wykresów
            # [0, 0] Oryginalny czas
            ax1 = self.axes[0, 0]
            ax1.clear()
            ax1.plot(t_orig_p, s_orig_p, color="#1f77b4", linewidth=0.8)
            ax1.set_title(f"Przed próbkowaniem - Czas (Fs = {orig_sr} Hz, {selected_channel})", fontsize=9, fontweight="bold")
            ax1.set_xlabel("Czas [s]", fontsize=8)
            ax1.set_ylabel("Amplituda", fontsize=8)
            ax1.grid(True, linestyle="--", alpha=0.5)

            # [0, 1] Resamplowany czas
            ax2 = self.axes[0, 1]
            ax2.clear()
            ax2.plot(t_res_p, s_res_p, color="#d62728", linewidth=0.8)
            ax2.set_title(f"Po próbkowaniu - Czas (Nowe Fs = {target_sr} Hz)", fontsize=9, fontweight="bold")
            ax2.set_xlabel("Czas [s]", fontsize=8)
            ax2.set_ylabel("Amplituda", fontsize=8)
            ax2.grid(True, linestyle="--", alpha=0.5)

            # [1, 0] Oryginalne FFT
            orig_nyquist = orig_sr / 2
            ax3 = self.axes[1, 0]
            ax3.clear()
            ax3.plot(f_orig_p, m_orig_p, color="#1f77b4", linewidth=0.8, label="Widmo sygnału")
            ax3.axvline(orig_nyquist, color="purple", linestyle="--", linewidth=1.2, label=f"Nyquist ({orig_nyquist:.0f} Hz)")
            ax3.set_title(f"Przed próbkowaniem - Widmo FFT (Pasmo do {orig_nyquist:.0f} Hz)", fontsize=9, fontweight="bold")
            ax3.set_xlabel("Częstotliwość [Hz]", fontsize=8)
            ax3.set_ylabel("Magnituda", fontsize=8)
            ax3.grid(True, linestyle="--", alpha=0.5)
            ax3.set_xlim(left=0, right=orig_nyquist)
            ax3.legend(loc="upper right", fontsize=7)

            # [1, 1] Resamplowane FFT z zaznaczoną nową granicą Nyquista
            target_nyquist = target_sr / 2
            ax4 = self.axes[1, 1]
            ax4.clear()
            ax4.plot(f_res_p, m_res_p, color="#d62728", linewidth=0.8, label="Widmo sygnału")
            ax4.axvline(target_nyquist, color="red", linestyle="--", linewidth=1.2, label=f"Nyquist Nowe ({target_nyquist:.0f} Hz)")
            ax4.set_title(f"Po próbkowaniu - Widmo FFT (Granica Nyquista: {target_nyquist:.0f} Hz)", fontsize=9, fontweight="bold")
            ax4.set_xlabel("Częstotliwość [Hz]", fontsize=8)
            ax4.set_ylabel("Magnituda", fontsize=8)
            ax4.grid(True, linestyle="--", alpha=0.5)
            ax4.set_xlim(left=0, right=target_nyquist)
            ax4.legend(loc="upper right", fontsize=7)

            self.fig.subplots_adjust(hspace=0.45, wspace=0.25, left=0.07, right=0.98, top=0.93, bottom=0.1)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Błąd analizy", f"Wystąpił błąd podczas analizy:\n{str(e)}")

    def _on_exit(self):
        """Obsługa przycisku Wyjdź z potwierdzeniem."""
        if messagebox.askyesno("Wyjście", "Czy na pewno chcesz zamknąć aplikację?"):
            self.root.destroy()
