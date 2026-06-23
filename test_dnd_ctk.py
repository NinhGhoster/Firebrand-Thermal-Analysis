import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES

class Tk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

root = Tk()
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', lambda e: print("Dropped:", e.data))
print("Ready")
