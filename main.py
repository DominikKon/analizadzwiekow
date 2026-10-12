import tkinter as tk
from gui import AudioAnalyzerGUI


def main():
    root = tk.Tk()
    
    # Utworzenie głównej instancji aplikacji
    app = AudioAnalyzerGUI(root)
    
    root.mainloop()


if __name__ == "__main__":
    main()
