import re
import os

with open('FirebrandThermalAnalysis.py', 'r') as f:
    text = f.read()

# 1. Add Support Extension
text = text.replace(
    'SUPPORTED_EXTENSIONS = (".seq", ".csq", ".jpg", ".ats", ".sfmov", ".img")',
    'SUPPORTED_EXTENSIONS = (".seq", ".csq", ".jpg", ".ats", ".sfmov", ".img", ".nc")'
)

# 2. Add netCDF4 import safely
import_block = """try:
    import fnv
    import fnv.file
except Exception:
    print("FLIR SDK required"); sys.exit(1)

try:
    import netCDF4 as nc
except ImportError:
    nc = None
    print("netCDF4 not found. .nc compressed files will not be supported.")

# --- Video Reader Abstraction ---
class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.num_frames = 0
        self.width = 0
        self.height = 0
        self.is_temp = True
        self.emissivity = None
        self.unit_label = "C"
        
    def get_frame(self, idx: int) -> np.ndarray:
        raise NotImplementedError
    
    def set_emissivity(self, emiss: float):
        pass

class FNVReader(VideoReader):
    def __init__(self, path: str):
        super().__init__(path)
        self.im = fnv.file.ImagerFile(path)
        self.is_temp = self.im.has_unit(fnv.Unit.TEMPERATURE_FACTORY)
        if self.is_temp:
            self.im.unit = fnv.Unit.TEMPERATURE_FACTORY
            self.im.temp_type = fnv.TempType.CELSIUS
        else:
            self.im.unit = fnv.Unit.COUNTS
            self.unit_label = "counts"
        
        self.num_frames = self.im.num_frames
        self.width = self.im.width
        self.height = self.im.height
        try:
            self.emissivity = float(self.im.object_parameters.emissivity)
        except Exception:
            self.emissivity = None
            
    def get_frame(self, idx: int) -> np.ndarray:
        self.im.get_frame(idx)
        return np.array(self.im.final, copy=False).reshape((self.height, self.width))
        
    def set_emissivity(self, emiss: float):
        try:
            obj_params = self.im.object_parameters
            obj_params.emissivity = emiss
            self.im.object_parameters = obj_params
            self.emissivity = emiss
        except Exception:
            pass

class NetCDFReader(VideoReader):
    def __init__(self, path: str):
        super().__init__(path)
        if nc is None:
            raise ImportError("netCDF4 library is required to read .nc files")
        self.ds = nc.Dataset(path, "r")
        self.temp_var = self.ds.variables["temperature"]
        self.num_frames = self.temp_var.shape[0]
        self.height = self.temp_var.shape[1]
        self.width = self.temp_var.shape[2]
        self.unit_label = "C"
        
        if "emissivity_original" in self.ds.ncattrs():
            self.emissivity = float(self.ds.getncattr("emissivity_original"))
            
    def get_frame(self, idx: int) -> np.ndarray:
        return np.array(self.temp_var[idx, :, :])
        
    def set_emissivity(self, emiss: float):
        # Emissivity is baked in during compression for NetCDF, but we update the meta tracker
        self.emissivity = emiss

def create_video_reader(path: str) -> VideoReader:
    if path.lower().endswith(".nc"):
        return NetCDFReader(path)
    return FNVReader(path)
# --------------------------------
"""
text = text.replace('try:\n    import fnv\n    import fnv.file\nexcept Exception:\n    print("FLIR SDK required"); sys.exit(1)', import_block)

# 3. Modify `_load_seq` to use `self.reader`
old_load_seq = """    def _load_seq(self, path: str, reset_settings: bool):
        try:
            im = fnv.file.ImagerFile(path)
            self.unit_is_temp = im.has_unit(fnv.Unit.TEMPERATURE_FACTORY)
            if self.unit_is_temp:
                im.unit = fnv.Unit.TEMPERATURE_FACTORY
                im.temp_type = DESIRED_TEMP_UNIT
                self.unit_label = "C"
            else:
                im.unit = fnv.Unit.COUNTS
                self.unit_label = "counts"
            self.im = im
            self.seq_path = path
            if path in self.batch_paths:
                self.batch_index = self.batch_paths.index(path)
            self.num_frames = im.num_frames
            self.width = im.width
            self.height = im.height
            self.current_idx = 0
            self.slider.configure(from_=0, to=max(0, self.num_frames-1))
            self._reset_tracking()
            self._applied_emissivity = None
            self.status.configure(text=f"Status: opened {os.path.basename(path)} | {self.width}x{self.height} | {self.num_frames} frames")
            meta_emiss = None
            try:
                meta_emiss = float(im.object_parameters.emissivity)
            except Exception:
                meta_emiss = None"""

new_load_seq = """    def _load_seq(self, path: str, reset_settings: bool):
        try:
            self.reader = create_video_reader(path)
            self.im = self.reader  # Compatibility alias
            self.unit_is_temp = self.reader.is_temp
            self.unit_label = self.reader.unit_label
            
            self.seq_path = path
            if path in self.batch_paths:
                self.batch_index = self.batch_paths.index(path)
            
            self.num_frames = self.reader.num_frames
            self.width = self.reader.width
            self.height = self.reader.height
            self.current_idx = 0
            self.slider.configure(from_=0, to=max(0, self.num_frames-1))
            self._reset_tracking()
            self._applied_emissivity = None
            self.status.configure(text=f"Status: opened {os.path.basename(path)} | {self.width}x{self.height} | {self.num_frames} frames")
            
            meta_emiss = self.reader.emissivity"""

text = text.replace(old_load_seq, new_load_seq)

# Replace remaining self.im usages inside class methods
text = text.replace('self.im.get_frame(idx)', 'self.reader.get_frame(idx)')
text = text.replace('np.array(self.im.final, copy=True).reshape(\n                        (self.height, self.width)\n                    )', 'self.reader.get_frame(idx)')
text = text.replace('np.array(self.im.final, copy=False).reshape((self.height, self.width))', 'self.reader.get_frame(self.current_idx)')
text = text.replace('self.im.get_frame(self.current_idx)', '')

# Fix ensure_emissivity
old_emiss = """    def _ensure_emissivity(self):
        if self.im is None:
            return
        try:
            emiss = float(self.var_emissivity.get())
            emiss = max(0.01, min(1.0, emiss))
        except Exception:
            return
        if self._applied_emissivity is None or abs(self._applied_emissivity - emiss) > 1e-4:
            try:
                obj_params = self.im.object_parameters
                obj_params.emissivity = emiss
                self.im.object_parameters = obj_params
                self._applied_emissivity = emiss
            except Exception:
                pass"""
new_emiss = """    def _ensure_emissivity(self):
        if self.im is None:
            return
        try:
            emiss = float(self.var_emissivity.get())
            emiss = max(0.01, min(1.0, emiss))
        except Exception:
            return
        if self._applied_emissivity is None or abs(self._applied_emissivity - emiss) > 1e-4:
            self.reader.set_emissivity(emiss)
            self._applied_emissivity = emiss"""
text = text.replace(old_emiss, new_emiss)

old_apply = """            obj_params = self.im.object_parameters
            obj_params.emissivity = emiss
            self.im.object_parameters = obj_params"""
new_apply = """            self.reader.set_emissivity(emiss)"""
text = text.replace(old_apply, new_apply)

# Export worker update
old_worker = """        im = fnv.file.ImagerFile(seq_path)
        unit_is_temp = im.has_unit(fnv.Unit.TEMPERATURE_FACTORY)
        if unit_is_temp:
            im.unit = fnv.Unit.TEMPERATURE_FACTORY
            im.temp_type = DESIRED_TEMP_UNIT
        else:
            im.unit = fnv.Unit.COUNTS

        try:
            obj_params = im.object_parameters
            obj_params.emissivity = float(settings.get("emissivity", 0.9))
            im.object_parameters = obj_params
        except Exception:
            pass

        num_frames = im.num_frames
        width = im.width
        height = im.height"""
new_worker = """        reader = create_video_reader(seq_path)
        reader.set_emissivity(float(settings.get("emissivity", 0.9)))
        num_frames = reader.num_frames
        width = reader.width
        height = reader.height"""
text = text.replace(old_worker, new_worker)

# Worker loop
old_worker_loop = """            im.get_frame(idx)
            frame = np.array(im.final, copy=False).reshape((height, width))"""
new_worker_loop = """            frame = reader.get_frame(idx)"""
text = text.replace(old_worker_loop, new_worker_loop)

# Add About Dialog
about_func = """    def show_about_dialog(self):
        about = ctk.CTkToplevel(self)
        about.title("About Firebrand Thermal Analysis")
        about.geometry("400x250")
        about.resizable(False, False)
        
        # Center the window
        about.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - 200
        y = self.winfo_y() + (self.winfo_height() // 2) - 125
        about.geometry(f"+{x}+{y}")
        
        # Ensure it stays on top
        about.attributes("-topmost", True)
        
        ctk.CTkLabel(about, text="Firebrand Thermal Analysis", font=("Fira Sans", 20, "bold"), text_color="#3B82F6").pack(pady=(20, 5))
        ctk.CTkLabel(about, text=f"Version {APP_VERSION}", font=("Fira Sans", 14)).pack(pady=(0, 15))
        
        ctk.CTkLabel(about, text="Developed by:", font=("Fira Sans", 12, "bold"), text_color="gray").pack()
        ctk.CTkLabel(about, text="H. Nguyen, J. Filippi, T. Penman, M. Peace, A. Filkov", font=("Fira Sans", 12)).pack(pady=(0, 15))
        
        def _open_compressor():
            webbrowser.open("https://github.com/NinhGhoster/SEQ-CSQ-compressor")
            
        repo_btn = ctk.CTkButton(about, text="SEQ-CSQ-compressor on GitHub", fg_color="#F59E0B", text_color="#0F172A", hover_color="#D97706", command=_open_compressor)
        repo_btn.pack(pady=(5, 20))
"""
text = text.replace('    def show_export_menu(self):', about_func + '\n    def show_export_menu(self):')

# Add "About" button to footer
old_footer = """        ctk.CTkButton(
            footer_frame,
            text="Check for updates",
            command=self.on_check_updates,
            font=("Fira Sans", 12),
            height=30,
        ).pack(fill="x", padx=10, pady=(2, 8))"""
new_footer = """        btn_row = ctk.CTkFrame(footer_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(2, 8))
        
        ctk.CTkButton(
            btn_row,
            text="Check for updates",
            command=self.on_check_updates,
            font=("Fira Sans", 12),
            height=30,
        ).pack(side="left", expand=True, padx=(0, 2))
        
        ctk.CTkButton(
            btn_row,
            text="About...",
            command=self.show_about_dialog,
            font=("Fira Sans", 12),
            height=30,
        ).pack(side="right", expand=True, padx=(2, 0))"""
text = text.replace(old_footer, new_footer)

with open('FirebrandThermalAnalysis.py', 'w') as f:
    f.write(text)

print("Refactor complete.")
