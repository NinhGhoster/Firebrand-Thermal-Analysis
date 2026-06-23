import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES

def drop(event):
    print("Dropped:", event.data)

root = TkinterDnD.Tk()
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', drop)
root.mainloop()
