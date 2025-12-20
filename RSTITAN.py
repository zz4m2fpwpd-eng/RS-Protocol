# ==============================================================================
# PROJECT: TITAN RS - SYNERGY (V53.0 - VOGUE VISUALS + ECLIPSE STABILITY)
# AUTHOR: Dr. Sandhu
# ARCHITECTURE: Crash Guard | Vogue Charts | Leak Sentinel | Py3.14 Safe
# ==============================================================================
import networkx as nx
import os
import sys
import time
import warnings
import multiprocessing
import re
import requests
import pandas as pd
import numpy as np
from scipy import stats
from urllib.parse import urlparse

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest

# PDF Check
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("   [NOTE] 'fpdf' missing. Outputting folders only.")

# Configuration (VOGUE EDITION STYLE)
matplotlib.use("Agg")
plt.switch_backend('Agg') 
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

SAFE_CORES = max(1, multiprocessing.cpu_count() - 2)
MAX_SAMPLE_ROWS = 200_000 
CHUNK_SIZE = 50_000 
LOG_FILE = "TITAN_BATCH_LOG.txt"

# ==============================================================================
# MODULE 0: THE LOGGER
# ==============================================================================
def log_event(filename, status, reason):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {filename[:30].ljust(30)} | {status.ljust(10)} | {reason}\n"
    try:
        with open(LOG_FILE, "a") as f:
            f.write(entry)
    except: pass

# ==============================================================================
# MODULE 1: DOWNLOADER & PDF
# ==============================================================================
def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except: return False

def download_dataset(url, dest_folder):
    if not os.path.exists(dest_folder): os.makedirs(dest_folder)
    filename = os.path.join(dest_folder, os.path.basename(urlparse(url).path))
    if not filename.endswith(('.csv', '.txt')): filename += ".csv"
    print(f"   [NET] Downloading: {url}...")
    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"     ✅ Saved to: {filename}")
            return filename
    except Exception as e: print(f"     ❌ Download Error: {e}")
    return None

def compile_pdf_report(folder, filename, report_text, charts):
    if not PDF_AVAILABLE: return
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt=f"TITAN RS AUDIT: {filename}", ln=1, align='C')
        pdf.set_font("Arial", size=10)
        for line in report_text.split('\n'):
            safe_line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(200, 5, txt=safe_line, ln=1)
        
        # [V53 FIX] Intelligent Layout Engine
        for i, chart_path in enumerate(charts):
            if i % 2 == 0: pdf.add_page()
            try:
                # Alternate positioning for 2 charts per page
                y_pos = 20 if i % 2 == 0 else 140
                pdf.image(chart_path, x=15, y=y_pos, w=180)
            except: pass
            
        pdf.output(f"{folder}/{filename}_FULL_REPORT.pdf")
        print(f"     📄 PDF Generated: {filename}_FULL_REPORT.pdf")
    except: pass

# ==============================================================================
# MODULE 2: SAFE AUTO-FUSION
# ==============================================================================
def smart_fusion(file_list):
    print("   [FUSION] Scanning for connectable datasets (Content-Aware)...")
    loaded_dfs = {}
    for f in file_list:
        try:
            df_head = pd.read_csv(f, nrows=1000, encoding='latin1', on_bad_lines='skip')
            loaded_dfs[f] = df_head
        except: pass
    if len(loaded_dfs) < 2: return None, None

    keys = list(loaded_dfs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            f1, f2 = keys[i], keys[j]
            df1, df2 = loaded_dfs[f1], loaded_dfs[f2]
            for c1 in df1.columns:
                for c2 in df2.columns:
                    if df1[c1].nunique() < 5 or df2[c2].nunique() < 5: continue
                    set1 = set(df1[c1].dropna().astype(str))
                    set2 = set(df2[c2].dropna().astype(str))
                    intersection = len(set1.intersection(set2))
                    if intersection == 0: continue
                    min_len = min(len(set1), len(set2))
                    ratio = intersection / min_len
                    if ratio > 0.6:
                        print(f"     > DETECTED SPLIT: '{os.path.basename(f1)}' ({c1}) <-> '{os.path.basename(f2)}' ({c2})")
                        if not df1[c1].is_unique and not df2[c2].is_unique:
                            print(f"     ⚠️  CARTESIAN EXPLOSION RISK: Keys not unique. Skipping Merge.")
                            return None, None
                        print(f"       Match Ratio: {ratio:.1%}. Safe to Merge.")
                        data1, data2 = smart_load(f1), smart_load(f2)
                        try:
                            fused = pd.merge(data1, data2, left_on=c1, right_on=c2, how='inner')
                            if len(fused) == 0: return None, None
                            print(f"     ✅ MERGE SUCCESS: {len(fused)} rows.")
                            return fused, f"FUSED_{os.path.basename(f1)[:5]}_{os.path.basename(f2)[:5]}.csv"
                        except: pass
    return None, None

# ==============================================================================
# MODULE 3: TARGET INTELLIGENCE & CLEANING
# ==============================================================================
def medical_decoder(df):
    codebook = {
        r'^_MICHD': 'TARGET_HEART_DISEASE', r'^CVDSTRK': 'TARGET_STROKE', r'^DIABETE': 'TARGET_DIABETES',
        r'^CVDINFR': 'TARGET_HEART_ATTACK', r'^CVDCRHD': 'TARGET_ANGINA', r'^_RFHLTH': 'TARGET_GOOD_HEALTH',
        r'^HadHeartAttack': 'TARGET_HEART_ATTACK', r'^Stroke': 'TARGET_STROKE', r'^class$': 'TARGET_CLASS'
    }
    for col in df.columns:
        for p, n in codebook.items():
            if re.search(p, col, re.IGNORECASE) and n not in df.columns: 
                df.rename(columns={col: n}, inplace=True)
    return df

def analyze_structure_and_fix(df):
    # [V51/V52 FIX] Aggressive Index Cleaning
    drops = [c for c in df.columns if 'Unnamed:' in c or c.lower() in ['index', 'id', 'row_id', 'patient_id']]
    if drops:
        df.drop(columns=drops, inplace=True, errors='ignore')
        print(f"   [CLEANER] Removed artifact columns: {drops}")

    df = medical_decoder(df)
    outcomes = ['DIED', 'L_THREAT', 'HOSPITAL', 'ER_VISIT', 'DISABLE', 'RECOVD', 'DEATH']
    for col in df.columns:
        if any(o in col.upper() for o in outcomes):
            if len(df[col].dropna().unique()) <= 1:
                print(f"   [PHOENIX] Fixing sparse target '{col}': Filling NaNs with 'N'.")
                df[col] = df[col].fillna('N')

    best_target = None
    priority = ['DIED', 'DEATH', 'DEATH_YN', 'TARGET_HEART_DISEASE', 'TARGET_STROKE', 'TARGET_DIABETES', 
                'STROKE', 'DIABETES', 'DIAGNOSIS', 'RESULT', 'TARGET', 'CLASS', 'OUTCOME', 'STATUS', 'DNA', 'GENE']
    for p in priority:
        matches = [c for c in df.columns if c.upper() == p or c.upper() == p.replace('_', '')]
        if matches: best_target = matches[0]; break
    if not best_target:
        for c in df.columns:
            if 'TARGET' in c.upper() and 2 <= df[c].nunique() <= 10: best_target = c; break
    if not best_target:
        last_col = df.columns[-1]
        if 2 <= df[last_col].nunique() <= 10:
            print(f"   [SCAVENGER] No named target found. Assuming '{last_col}' is target.")
            best_target = last_col
        else:
            first_col = df.columns[0]
            if 2 <= df[first_col].nunique() <= 10:
                 print(f"   [SCAVENGER] No named target found. Assuming '{first_col}' is target.")
                 best_target = first_col

    if not best_target: return None, [], df
    
    vals = df[best_target].dropna().unique()
    if all(x in [1,2,3,4,7,9] for x in vals[:5]):
         print(f"   [GOV LOGIC] Detected 1=Yes/2=No scheme. Normalizing...")
         df = df[df[best_target].isin([1,2])].copy()
         df[best_target] = df[best_target].apply(lambda x: 1 if x==1 else 0)
    
    ban = []
    if 'DIED' in best_target.upper() or 'DEATH' in best_target.upper(): 
        ban = ['DATEDIED', 'VAX_DATE', 'ONSET_DATE', 'NUMDAYS', 'DEATH']
    if 'HEART' in best_target.upper(): 
        ban = ['HEART_ATTACK', 'STROKE', 'CVDINFR', 'CVDSTRK', 'ANGINA', 'CORONARY']
    
    preds = []
    leak = ['ID', 'DATE', 'TEXT', 'DESC', 'HISTORY', 'SYMPTOM', 'LAB_DATA', 'FORM_VERS']
    for c in df.columns:
        if c == best_target: continue
        if any(l in c.upper() for l in leak): continue
        if any(h in c.upper() for h in ban): continue
        if df[c].dtype == 'object' and df[c].nunique() > 50: continue
        preds.append(c)
    return best_target, preds, df

# ==============================================================================
# MODULE 4: CORE PIPELINE & HEALER
# ==============================================================================
def smart_load(filepath):
    try:
        with open(filepath, 'r', errors='ignore') as f:
            head = f.readline()
            sep = ';' if ';' in head and ',' not in head else '\t' if '\t' in head else ','
        if os.path.getsize(filepath) > 200*1024*1024:
            chunks = []
            for chunk in pd.read_csv(filepath, sep=sep, chunksize=CHUNK_SIZE, encoding='latin1', on_bad_lines='skip', low_memory=False):
                if len(chunks)*CHUNK_SIZE*0.1 > MAX_SAMPLE_ROWS: break
                chunks.append(chunk.sample(frac=0.1))
            return pd.concat(chunks, ignore_index=True)
        else: return pd.read_csv(filepath, sep=sep, encoding='latin1', on_bad_lines='skip')
    except: return None

def run_statistical_protocols(df, target, predictors):
    print("   [PROTOCOLS] Running Wilkinson/Kelly Statistical Audit...")
    stats_log = []
    num_preds = df[predictors].select_dtypes(include=[np.number]).columns
    if len(num_preds) == 0: return ["No numeric predictors."]
    sample_col = num_preds[0]
    valid_data = df[sample_col].dropna()
    if len(valid_data) < 3:
        stats_log.append("Not enough data for statistical tests.")
        return stats_log
    sample_data = valid_data.sample(min(500, len(valid_data)), random_state=42)
    try:
        stat, p_norm = stats.shapiro(sample_data)
        test = "ANOVA" if p_norm > 0.05 else "Kruskal-Wallis"
        msg = f"     > Normality Check ({sample_col}): p={p_norm:.4f} -> Protocol: {test}"
        print(msg); stats_log.append(msg)
    except: stats_log.append("Normality Check Failed -> Mode: Non-Parametric")
    return stats_log

def leakage_hunter(df, target, predictors):
    print(f"   [HUNTER] Scanning {len(predictors)} features for Non-Linear Leakage...")
    
    # [V52 FIX] Correlation Sentinel
    try:
        y_num = LabelEncoder().fit_transform(df[target].astype(str))
        nums = df[predictors].select_dtypes(include=[np.number]).columns
        if len(nums) > 0:
            corrs = df[nums].apply(lambda x: x.corr(pd.Series(y_num)))
            leaks = corrs[abs(corrs) > 0.95].index.tolist()
            if leaks:
                print(f"     ✂️  ARRESTED SCALAR LEAKS (>95% Correlation): {leaks}")
                predictors = [p for p in predictors if p not in leaks]
    except: pass
    
    sample_size = min(10000, len(df))
    if sample_size < 10: return predictors
    df_s = df.sample(sample_size, random_state=42)
    y = LabelEncoder().fit_transform(df_s[target].astype(str))
    safe = []
    try:
        X = pd.get_dummies(df_s[predictors], drop_first=True)
        if X.shape[1] > 500: X = X.iloc[:, :500]
        model = RandomForestClassifier(n_estimators=20, max_depth=5, n_jobs=SAFE_CORES)
        model.fit(X, y)
        imps = pd.Series(model.feature_importances_, index=X.columns)
        for feat, imp in imps.items():
            if imp > 0.40: print(f"     ✂️  LEAK: '{feat}' ({imp:.2%}) -> ARRESTED.")
            else:
                orig = feat.split('_')[0]
                if orig not in safe and orig in predictors: safe.append(orig)
                elif feat in predictors and feat not in safe: safe.append(feat)
    except: return predictors
    return safe

def safe_encode(df, target, predictors):
    valid = [p for p in predictors if p in df.columns]
    df_sub = df[valid].copy()
    for c in df_sub.select_dtypes(include='object').columns:
        if df_sub[c].nunique() > 10: df_sub[c] = LabelEncoder().fit_transform(df_sub[c].astype(str))
    X = pd.get_dummies(df_sub, drop_first=True)
    if X.shape[1] > 300: X = X.iloc[:, :300]
    y = LabelEncoder().fit_transform(df[target].astype(str))
    return X, y

def heal_data(df, target):
    start_len = len(df)
    df = df.dropna(axis=1, how='all')
    nums = df.select_dtypes(include=[np.number]).columns
    df[nums] = df[nums].fillna(df[nums].median()).fillna(0)
    cats = df.select_dtypes(include=['object']).columns
    for c in cats: df[c] = df[c].fillna("Unknown")
    
    if len(df) > 500:
        try:
            fit_df = df.sample(min(20000, len(df)), random_state=42)
            X_if = fit_df.select_dtypes(include=[np.number]).dropna(axis=1)
            if X_if.shape[1] > 0:
                iso = IsolationForest(contamination=0.05, n_jobs=SAFE_CORES, random_state=42)
                iso.fit(X_if)
                X_full = df[X_if.columns].fillna(0)
                mask = iso.predict(X_full)
                df = df[mask == 1]
                count = start_len - len(df)
                print(f"   [HEALER] Threats Eliminated (Outliers): {count}")
                return df, count
        except: pass
    return df, 0

# V55.5: RENDER CHART FACTORY - TRUE PROCESS ISOLATION
def render_chart_factory(task):
    type_, data, save_path, meta = task
    out_file = f"{save_path}/{meta['fname']}"
    
    # CRITICAL FIX (V55.5): Localize heavy plotting imports only to the worker process
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import networkx as nx 
        import matplotlib
        matplotlib.use('Agg')
        from sklearn.calibration import calibration_curve # Local import for calibration
        import gc
    except ImportError:
        return None

    fig = None
    
    try:
        fig = plt.figure(figsize=(10, 6))
        
        if type_ == 'hist':
            sns.histplot(data=data, x=meta['col'], hue=meta['target'], kde=True, bins=15, palette="viridis")
            plt.title(f'Distribution: {meta["title"]}', fontsize=14, fontweight='bold')
            plt.xlabel(meta["title"], fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
        elif type_ == 'violin':
            sns.violinplot(data=data, x=meta['target'], y=meta['col'], palette="muted", split=True, inner="quartile")
            plt.title(f'Distribution by Outcome: {meta["title"]}', fontsize=14, fontweight='bold')
            plt.xlabel("Target Outcome", fontsize=12)
            plt.ylabel(meta["title"], fontsize=12)
            
        elif type_ == 'calibration':
            y_t, y_p = data
            n_bins = min(20, max(10, len(y_t) // 50))
            p_t, p_p = calibration_curve(y_t, y_p, n_bins=n_bins, strategy='quantile')
            plt.plot(p_p, p_t, marker='o', linewidth=2, label='Titan Model', markersize=8)
            plt.plot([0,1],[0,1],'--', color='gray', label='Perfectly Calibrated')
            plt.xlim([0, 1]); plt.ylim([0, 1]); plt.grid(True, alpha=0.3)
            plt.title('Model Reliability Curve', fontsize=14, fontweight='bold')
            plt.xlabel("Predicted Probability", fontsize=12)
            plt.ylabel("True Probability", fontsize=12)
            plt.legend()
        
        elif type_ == 'network':
            G = nx.Graph()
            for col, imp in data:
                G.add_node(col, size=imp * 5000) 
                G.add_edge(meta['target'], col, weight=imp * 10) 
            
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
            edge_weights = [G[u][v].get('weight', 1) for u,v in G.edges()]
            node_sizes = [G.nodes[n].get('size', 2000) for n in G.nodes()]
            
            nx.draw(G, pos, 
                    with_labels=True, 
                    node_color='lightblue', 
                    node_size=node_sizes, 
                    font_size=10, 
                    font_weight='bold', 
                    edge_color='gray', 
                    width=edge_weights)

        plt.tight_layout()
        plt.savefig(out_file)
        
        # CRITICAL CLEANUP: Force plot closure and garbage collection
        plt.close(fig)
        gc.collect() 

        return out_file
    except Exception as e:
        # Cleanup on crash
        if fig: plt.close(fig)
        return None

def run_pipeline(filepath_or_df, filename=None):
    try:
        if isinstance(filepath_or_df, pd.DataFrame):
            df = filepath_or_df; fname = filename
            print(f"   [START] Auditing Fused Dataset: {fname}...")
        else:
            fname = os.path.basename(filepath_or_df)
            print(f"   [START] Auditing {fname}...")
            df = smart_load(filepath_or_df)
            
        if df is None: 
            log_event(fname, "FAILED", "Could not load file (Format Error)")
            return "Load Fail"
        if len(df) == 0: 
            log_event(fname, "FAILED", "Dataset is empty")
            return "Empty Dataset"
        
        target, preds, df = analyze_structure_and_fix(df)
        if not target: 
            log_event(fname, "SKIPPED", "No identifiable target variable found")
            return "No Target"
        
        df = df.reset_index(drop=True)
        stats_report = run_statistical_protocols(df, target, preds)
        preds = leakage_hunter(df, target, preds)
        if not preds: 
            log_event(fname, "FAILED", "All predictors removed by Leakage Hunter")
            return f"{fname}: ⚠️ FAILED (All predictors identified as Leakage)"
        
        df_clean, threats_removed = heal_data(df, target)
        if len(df_clean) == 0: 
            log_event(fname, "FAILED", "Outlier removal deleted all rows")
            return "Data Cleaned to Empty"
        
        X, y = safe_encode(df_clean, target, preds)
        
        # Split
        X_tr, X_cal_test, y_tr, y_cal_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.4, random_state=42)
        X_cal, X_te, y_cal, y_te = train_test_split(X_cal_test, y_cal_test, test_size=0.75, random_state=42)
        
        base_model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=SAFE_CORES)
        base_model.fit(X_tr, y_tr)
        
        # [V53 FIX] CRASH GUARD + CHART RESTORATION
        model = base_model
        y_prob = None
        auc = 0
        
        # [V54 UPGRADE] ROBUST CALIBRATION LOGIC
        if len(np.unique(y))==2: 
            try:
                # 1. Try 'prefit' (Best for big data)
                calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
                calibrated.fit(X_cal, y_cal)
                model = calibrated
            except Exception:
                try:
                    # 2. Fallback: Internal CV (Fixes Python 3.14 Alpha Crash)
                    print(f"     ⚠️  Calibration Mode Switch: 'prefit' failed -> Retrying with CV=3...")
                    # Must re-init base model to avoid 'already fitted' warnings in CV mode
                    base_model_cv = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=SAFE_CORES)
                    calibrated = CalibratedClassifierCV(base_model_cv, method='sigmoid', cv=3)
                    # Use combined Train+Cal set for CV fit
                    X_comb = np.vstack((X_tr, X_cal))
                    y_comb = np.hstack((y_tr, y_cal))
                    calibrated.fit(X_comb, y_comb)
                    model = calibrated
                except Exception as e:
                    # 3. Final Safety Net: Use Uncalibrated Base Model
                    print(f"     ⚠️  Calibration Skipped (Env Error): {str(e)[:40]}... Using Base Model.")
                    model = base_model

            # Calculate Probabilities & AUC using whatever model survived
            y_prob = model.predict_proba(X_te)[:,1]
            auc = roc_auc_score(y_te, y_prob)
        else: 
            # Multi-class fallback
            y_prob = base_model.predict_proba(X_te)
            auc = roc_auc_score(y_te, y_prob, multi_class='ovr', average='macro')
            model = base_model
            
        out_dir = "Titan_Synergy_Results"
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        sub_dir = f"{out_dir}/{fname}_Audit"
        if not os.path.exists(sub_dir): os.makedirs(sub_dir)
        
        with open(f"{sub_dir}/REPORT.md", "w") as f:
            f.write(f"# TITAN AUDIT: {fname}\nTarget: {target}\nAUC: {auc:.3f}\n")
            f.write(f"Threats Removed: {threats_removed}\n")
            for l in stats_report: f.write(f"- {l}\n")
            
        # [V53 FIX] Universal Charting Logic (Works even if calibration crashes)
        tasks = []
        if len(np.unique(y))==2 and y_prob is not None: 
            tasks.append(('calibration', (y_te, y_prob), sub_dir, {'fname': 'Calibration.png', 'title': 'Calibration'}))
        
      # [V55.4 FIX] Extract Feature Importances Safely & Generate All Charts (FINAL)
        importances = None
        feature_names = X.columns
        
        # 1. Try direct access (Base Model)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        
        # 2. Try nested access (Calibrated Model - Prefit)
        elif hasattr(model, 'estimator'):
            if hasattr(model.estimator, 'feature_importances_'):
                importances = model.estimator.feature_importances_
        
        # 3. Try list access (Calibrated Model - CV Fallback)
        elif hasattr(model, 'calibrated_classifiers_'):
            # Average importances across all CV folds (The critical fix)
            imps = [clf.estimator.feature_importances_ for clf in model.calibrated_classifiers_ if hasattr(clf.estimator, 'feature_importances_')]
            if imps:
                # Calculate the mean importance across all successful CV models
                importances = np.mean(imps, axis=0)
        
        # Generate Charts (If Importances Found)
        if importances is not None:
            # We must use list conversion for compatibility with Numpy 2.x and Python 3.14 indexing
            feature_list = list(feature_names)
            num_features = min(15, len(importances))
            top_indices = np.argsort(importances)[::-1][:num_features]
            
            # Network Graph Data Prep (CRITICAL V55.4 FIX: Use index lookup for reliability)
            top_features_data = []
            
            # Iterate through the indices of the top features
            for i in top_indices:
                col = feature_list[i]
                # Check if the column exists in the clean DF before charting (extra safety)
                if col.split('_')[0] in df_clean.columns: 
                    top_features_data.append((col, importances[i]))
            
            # 1. Network Graph (1 Chart)
            tasks.append(('network', top_features_data, sub_dir, {'target': target, 'fname': 'Feature_Network.png', 'title': 'Top Predictors Network'}))
            
            # 2. Distribution Charts (15 Histograms + 15 Violins = 30 Charts)
            for encoded_col in [feature_list[i] for i in top_indices]:
                orig_col = encoded_col.split('_')[0]
                
                # We need to chart the original column, which is in df_clean
                if orig_col in df_clean.columns:
                    tasks.append(('violin', df_clean[[target, orig_col]], sub_dir, {'col': orig_col, 'target': target, 'fname': f'Violin_{orig_col}.png', 'title': f'Distribution by Outcome: {orig_col}'}))
                    tasks.append(('hist', df_clean[[target, orig_col]], sub_dir, {'col': orig_col, 'target': target, 'fname': f'Hist_{orig_col}.png', 'title': f'Distribution: {orig_col}'}))
        
        generated_charts = []
        with ProcessPoolExecutor(max_workers=SAFE_CORES) as ex:
            for res in ex.map(render_chart_factory, tasks):
                if res: generated_charts.append(res)
        
        compile_pdf_report(sub_dir, fname, f"Target: {target}\nAUC: {auc:.3f}\nThreats Removed: {threats_removed}", generated_charts)
        
        log_event(fname, "SUCCESS", f"AUC {auc:.3f} | {len(generated_charts)} Charts")
        return f"{fname}: ✅ SUCCESS (AUC {auc:.3f} | {len(generated_charts)} Charts)"
        
    except Exception as e:
        log_event(fname, "ERROR", str(e)[:100])
        return f"Error: {e}\n{traceback.format_exc()}"

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("\n" + "="*70)
    print("🛡️  TITAN RS: SYNERGY (V53.0)")
    print("   Features: Vogue Visuals | Crash Guard | Universal Charts")
    print("="*70)
    
    # Initialize Log
    with open(LOG_FILE, "w") as f:
        f.write(f"TITAN BATCH LOG - STARTED {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    queue = []
    print("\n[INSTRUCTIONS] Drag & Drop folders/files or Paste URLs.")
    print("Type 'START' to execute the batch.")
    
    while True:
        entry = input("\n[QUEUE] Add Input (or START): ").strip().replace("'", "").replace('"', "").replace("\\ ", " ").replace("\\(", "(").replace("\\)", ")")
        
        if entry.upper() == 'START':
            if not queue: print("   ⚠️  Queue is empty!"); continue
            break
        if entry.upper() == 'EXIT': return
        
        if is_url(entry):
            print(f"   > Added URL: {entry}"); queue.append(entry)
        elif os.path.isdir(entry):
            print(f"   > Added Folder: {os.path.basename(entry)}"); queue.append(entry)
        elif os.path.isfile(entry):
            print(f"   > Added File: {os.path.basename(entry)}"); queue.append(entry)
        else: print("   ❌ Invalid Path or URL.")
            
    print(f"\n🚀 STARTING BATCH ({len(queue)} Items)...")
    dl_folder = "Titan_Downloads"
    
    for item in queue:
        if is_url(item):
            f = download_dataset(item, dl_folder)
            if f: print(run_pipeline(f))
        elif os.path.isdir(item):
            files = [os.path.join(item, f) for f in os.listdir(item) if f.endswith('.csv')]
            if len(files) == 0:
                log_event(os.path.basename(item), "EMPTY FOLDER", "No CSVs found inside folder")
                print(f"   ⚠️  Folder empty: {os.path.basename(item)}")
            elif len(files) > 1:
                fused, name = smart_fusion(files)
                if fused is not None: print(run_pipeline(fused, name))
                else: 
                    for f in files: print(run_pipeline(f))
            else:
                for f in files: print(run_pipeline(f))
        elif os.path.isfile(item):
            print(run_pipeline(item))
            
    print(f"\n✅ BATCH COMPLETE. See '{LOG_FILE}' for full autopsy.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
