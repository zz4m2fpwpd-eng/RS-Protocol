# ==============================================================================
# PROJECT: TITAN-RS (ZENITH GUI - V1.1)
# DEVELOPER: Robin Sandhu
# ARCHITECTURE: CustomTkinter | Threaded Core | System-Sandhu Protocol
# STATUS: PATCHED (Fixed missing 're' import)
# ==============================================================================

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import threading
import sys
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import warnings
import re  # <--- CRITICAL FIX: Added missing regex module

# --- EMBEDDED TITAN CORE ENGINE (V61.0 LOGIC) ---
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from urllib.parse import urlparse
import requests
import gc
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import networkx as nx

# PDF Check
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Configuration
warnings.filterwarnings("ignore")
matplotlib.use('Agg')
sns.set_theme(style="white", context="paper", font_scale=1.2)

# Global Signals for GUI Updates
class LogRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, str_val):
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", str_val)
        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")
    
    def flush(self):
        pass

# --- CORE FUNCTIONS (Streamlined for GUI) ---
def smart_load(filepath):
    try:
        if os.path.getsize(filepath) > 200*1024*1024:
            chunks = []
            for chunk in pd.read_csv(filepath, chunksize=50000, encoding='latin1', on_bad_lines='skip', low_memory=False):
                if len(chunks)*50000*0.1 > 200000: break
                chunks.append(chunk.sample(frac=0.1))
            return pd.concat(chunks, ignore_index=True)
        return pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip')
    except: return None

def medical_decoder(df):
    codebook = {r'^_MICHD': 'TARGET_HEART_DISEASE', r'^CVDSTRK': 'TARGET_STROKE', r'^DIABETE': 'TARGET_DIABETES', 
                r'^CVDINFR': 'TARGET_HEART_ATTACK', r'^CVDCRHD': 'TARGET_ANGINA', r'^_RFHLTH': 'TARGET_GOOD_HEALTH',
                r'^HadHeartAttack': 'TARGET_HEART_ATTACK', r'^Stroke': 'TARGET_STROKE', r'^class$': 'TARGET_CLASS'}
    for col in df.columns:
        for p, n in codebook.items():
            if re.search(p, col, re.IGNORECASE) and n not in df.columns: df.rename(columns={col: n}, inplace=True)
    return df

def analyze_structure_and_fix(df):
    df = medical_decoder(df)
    drops = [c for c in df.columns if 'Unnamed:' in c or c.lower() in ['index', 'id', 'patient_id']]
    if drops: df.drop(columns=drops, inplace=True, errors='ignore')
    
    # Target Logic
    best_target = None
    priority = ['DIED', 'DEATH', 'TARGET_HEART_DISEASE', 'TARGET_STROKE', 'TARGET_DIABETES', 'TARGET', 'CLASS']
    for p in priority:
        matches = [c for c in df.columns if c.upper() == p]
        if matches: best_target = matches[0]; break
    if not best_target:
        if 2 <= df[df.columns[-1]].nunique() <= 10: best_target = df.columns[-1]
        elif 2 <= df[df.columns[0]].nunique() <= 10: best_target = df.columns[0]
            
    if not best_target: return None, [], df
    
    # Gov Logic
    vals = df[best_target].dropna().unique()
    if all(x in [1,2,3,4,7,9] for x in vals[:5]):
         df = df[df[best_target].isin([1,2])].copy()
         df[best_target] = df[best_target].apply(lambda x: 1 if x==1 else 0)

    preds = [c for c in df.columns if c != best_target and df[c].nunique() < 100]
    return best_target, preds, df

def render_chart_task(task):
    # Worker function for plots
    type_, data, save_path, meta = task
    out_file = f"{save_path}/{meta['fname']}"
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import networkx as nx
        plt.switch_backend('Agg')
        sns.set_theme(style="white", context="paper")
        
        fig = plt.figure(figsize=(10, 6))
        if type_ == 'hist':
            sns.histplot(x=data[meta['col']].values.tolist(), hue=data[meta['target']].values.tolist(), kde=True, bins=15, palette="viridis")
            plt.title(f"Distribution: {meta['title']}", fontweight='bold')
        elif type_ == 'violin':
            sns.violinplot(x=data[meta['target']].values.tolist(), y=data[meta['col']].values.tolist(), palette="muted", split=True)
            plt.title(f"Density: {meta['title']}", fontweight='bold')
        elif type_ == 'network':
            G = nx.Graph()
            for col, imp in data:
                G.add_node(col, size=imp*5000); G.add_edge(meta['target'], col, weight=imp*10)
            pos = nx.spring_layout(G, k=0.5); 
            nx.draw(G, pos, with_labels=True, node_color='#3498db', font_weight='bold')
            plt.title(f"Feature Network: {meta['title']}", fontweight='bold')
            
        plt.tight_layout(); plt.savefig(out_file, dpi=150); plt.close(fig); gc.collect()
        return out_file
    except: return None

def worker_audit(filepath, result_queue):
    # The heavy lifting logic
    try:
        fname = os.path.basename(filepath)
        result_queue.put(f"\n[START] Auditing {fname}...")
        
        df = smart_load(filepath)
        if df is None: result_queue.put("  ❌ Failed to load."); return
        
        target, preds, df = analyze_structure_and_fix(df)
        if not target: result_queue.put("  ⚠️ No Target Found."); return
        
        result_queue.put(f"  > Target: {target}")
        
        # Cleanup
        df = df.dropna(axis=1, how='all')
        nums = df.select_dtypes(include=[np.number]).columns
        df[nums] = df[nums].fillna(0)
        
        # Modeling
        X = pd.get_dummies(df[preds], drop_first=True)
        y = LabelEncoder().fit_transform(df[target].astype(str))
        X = X.iloc[:, :300] # Limit width
        
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=1)
        model.fit(X_tr, y_tr)
        
        if len(np.unique(y))==2: auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
        else: auc = 0.5
        
        result_queue.put(f"  > Model AUC: {auc:.3f}")
        
        # Charting
        out_dir = f"Titan_GUI_Results/{fname}_Audit"
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        tasks = []
        if hasattr(model, 'feature_importances_'):
            top = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10).index
            # Network Data
            net_data = [(c, model.feature_importances_[list(X.columns).index(c)]) for c in top]
            tasks.append(('network', net_data, out_dir, {'target': target, 'fname': 'Network.png', 'title': fname}))
            
            for col in top:
                orig = col.split('_')[0]
                if orig in df.columns:
                    tasks.append(('hist', df[[target, orig]], out_dir, {'col': orig, 'target': target, 'fname': f'Hist_{orig}.png', 'title': orig}))

        result_queue.put(f"  > Generating {len(tasks)} visuals...")
        
        # Sequential Execution for GUI Stability
        charts = []
        for t in tasks:
            res = render_chart_task(t)
            if res: charts.append(res)
            
        # PDF
        if PDF_AVAILABLE:
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt=f"TITAN-RS REPORT: {fname}", ln=1, align='C')
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Target: {target} | AUC: {auc:.3f}", ln=1, align='C')
            for c in charts:
                try: pdf.add_page(); pdf.image(c, x=10, y=30, w=180)
                except: pass
            pdf.output(f"{out_dir}/Report.pdf")
            result_queue.put("  ✅ PDF Report Generated.")
            
        result_queue.put(f"✅ {fname} COMPLETE.\n")
        
    except Exception as e:
        result_queue.put(f"❌ Error: {str(e)}\n")

# --- GUI CLASS ---
class TitanGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Branding & Layout
        self.title("TITAN-RS | System-Sandhu Protocol")
        self.geometry("900x700")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")

        # Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. Sidebar (Controls)
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="TITAN-RS", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.ver_label = ctk.CTkLabel(self.sidebar, text="V1.1 Zenith", font=ctk.CTkFont(size=12))
        self.ver_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        self.add_file_btn = ctk.CTkButton(self.sidebar, text="Add File(s)", command=self.add_files)
        self.add_file_btn.grid(row=2, column=0, padx=20, pady=10)
        
        self.add_folder_btn = ctk.CTkButton(self.sidebar, text="Add Folder", command=self.add_folder)
        self.add_folder_btn.grid(row=3, column=0, padx=20, pady=10)

        # 2. Main Area (Header)
        self.header = ctk.CTkLabel(self, text="Auditing Console", font=ctk.CTkFont(size=20))
        self.header.grid(row=0, column=1, padx=20, pady=20, sticky="w")

        # 3. Queue Display
        self.queue_label = ctk.CTkLabel(self, text="Processing Queue (0 Items):")
        self.queue_label.grid(row=1, column=1, padx=20, sticky="w")
        
        self.queue_box = ctk.CTkTextbox(self, height=100)
        self.queue_box.grid(row=2, column=1, padx=20, pady=(0, 10), sticky="nsew")
        self.queue_box.configure(state="disabled")

        # 4. Log Console
        self.log_label = ctk.CTkLabel(self, text="Real-Time Telemetry:")
        self.log_label.grid(row=3, column=1, padx=20, sticky="w")

        self.log_console = ctk.CTkTextbox(self, width=600, height=300, font=("Consolas", 12))
        self.log_console.grid(row=4, column=1, padx=20, pady=(0, 20), sticky="nsew")
        self.log_console.configure(state="disabled")

        # 5. Action Button
        self.run_btn = ctk.CTkButton(self.sidebar, text="INITIATE PROTOCOL", fg_color="#c0392b", hover_color="#e74c3c", height=50, font=ctk.CTkFont(size=14, weight="bold"), command=self.start_audit)
        self.run_btn.grid(row=5, column=0, padx=20, pady=30)

        # Variables
        self.file_queue = []
        self.queue = multiprocessing.Queue()
        self.running = False

    def update_queue_display(self):
        self.queue_box.configure(state="normal")
        self.queue_box.delete("0.0", "end")
        for f in self.file_queue:
            self.queue_box.insert("end", f"{os.path.basename(f)}\n")
        self.queue_box.configure(state="disabled")
        self.queue_label.configure(text=f"Processing Queue ({len(self.file_queue)} Items):")

    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")])
        if files:
            self.file_queue.extend(files)
            self.update_queue_display()

    def add_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if f.endswith(".csv"):
                        self.file_queue.append(os.path.join(root, f))
            self.update_queue_display()

    def log(self, message):
        self.log_console.configure(state="normal")
        self.log_console.insert("end", message + "\n")
        self.log_console.see("end")
        self.log_console.configure(state="disabled")

    def check_queue(self):
        while not self.queue.empty():
            msg = self.queue.get()
            self.log(msg)
        
        if self.running:
            self.after(100, self.check_queue)

    def start_audit(self):
        if not self.file_queue:
            self.log("⚠️ Queue is empty!")
            return
        
        self.running = True
        self.run_btn.configure(state="disabled", text="RUNNING...")
        self.log("🚀 INITIALIZING TITAN-RS ENGINE...")
        
        threading.Thread(target=self.process_thread, daemon=True).start()
        self.after(100, self.check_queue)

    def process_thread(self):
        # Create output dir
        if not os.path.exists("Titan_GUI_Results"): os.makedirs("Titan_GUI_Results")
        
        for filepath in self.file_queue:
            worker_audit(filepath, self.queue)
            
        self.queue.put("\n✅ BATCH COMPLETE. PROTOCOL FINISHED.")
        self.running = False
        self.run_btn.configure(state="normal", text="INITIATE PROTOCOL")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Force Spawn for Mac GUI safety
    if sys.platform == 'darwin':
        multiprocessing.set_start_method('spawn', force=True)
        
    app = TitanGUI()
    app.mainloop()