import tkinter as tk
from tkinter import ttk


def main():
    root = tk.Tk()
    root.title("Hello World")

    # Ustawienie rozmiaru i wyśrodkowanie okna na ekranie
    window_width = 400
    window_height = 200

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    center_x = int((screen_width - window_width) / 2)
    center_y = int((screen_height - window_height) / 2)

    root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
    root.minsize(300, 150)

    # Kontener z marginesami
    frame = ttk.Frame(root, padding=20)
    frame.pack(expand=True, fill="both")

    # Etykieta Hello World
    label = ttk.Label(
        frame,
        text="Hello, World! 👋",
        font=("Segoe UI", 20, "bold"),
        anchor="center"
    )
    label.pack(expand=True, pady=10)

    # Przycisk zamknięcia
    button = ttk.Button(frame, text="Zamknij", command=root.destroy)
    button.pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()

