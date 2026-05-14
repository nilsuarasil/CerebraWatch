import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

if __name__ == "__main__":
    os.chdir(ROOT)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data", "reports"), exist_ok=True)
    print("Working directory:", os.getcwd())

    try:
        from dashboard import DashboardApp
        import tkinter as tk

        print("Starting CerebraWatch...")
        root = tk.Tk()
        app = DashboardApp(root)

        root.update()
        app.update_patient_ui()

        print("Ready. Loading UI.")
        root.mainloop()

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
