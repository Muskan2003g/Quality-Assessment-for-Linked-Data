# app.py — Streamlit UI for ZHEClean with Quality Score & PDF report
# Run:  streamlit run app.py   (inside your Python 3.7 venv)
# Deps: streamlit==1.23.1 pandas==1.3.5 (reportlab optional for PDF)

import os
import sys
import re
import json
import time
import glob
import queue
import shutil
import random
import threading
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

# ---------- Config ----------
REPO_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = REPO_ROOT / "dataset"
RUNS_ROOT = REPO_ROOT / "ui_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="ZHEClean UI", page_icon="🧹", layout="wide")

# ---------- Utils ----------
def nice_time(ts=None):
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def safe_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def read_lines(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.rstrip("\n") for ln in f]

def count_nonempty_lines(path: Path):
    return sum(1 for ln in read_lines(path) if ln.strip())

def stream_subprocess(cmd, workdir):
    """
    Stream subprocess stdout -> Streamlit live, return (rc, full_text).
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
        bufsize=1,
        env=os.environ.copy(),
    )
    q = queue.Queue()

    def reader():
        for line in proc.stdout:
            q.put(line)
        try:
            proc.stdout.close()
        except Exception:
            pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    log_area = st.empty()
    acc = []
    while True:
        try:
            line = q.get(timeout=0.1)
            acc.append(line)
            log_area.code("".join(acc[-400:]))  # show tail to keep UI snappy
        except queue.Empty:
            if proc.poll() is not None:
                break

    rc = proc.wait()
    # drain
    while not q.empty():
        acc.append(q.get_nowait())

    return rc, "".join(acc)

def ensure_toy_dataset(name: str):
    d = DATASET_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "train.txt").exists():
        (d / "train.txt").write_text("\n".join(f"E{i} R0 E{i+1}" for i in range(0,180,2)), encoding="utf-8")
        (d / "valid.txt").write_text("E0 R0 E2\nE2 R0 E4\n", encoding="utf-8")
        (d / "test.txt").write_text("E4 R0 E6\nE6 R0 E8\n", encoding="utf-8")
        (d / "entities.dict").write_text("\n".join(f"E{i}\t{i}" for i in range(200))+"\n", encoding="utf-8")
        (d / "relations.dict").write_text("\n".join(f"R{i}\t{i}" for i in range(10))+"\n", encoding="utf-8")
    return d

# ---------- ID-based noise generator (matches run.py expectations) ----------
def _read_dict_any_order(path: Path):
    m = {}
    for ln in read_lines(path):
        if not ln.strip():
            continue
        parts = ln.split("\t") if "\t" in ln else ln.split()
        if len(parts) != 2:
            raise ValueError(f"Bad dict line (need 2 cols): {ln}")
        a, b = parts
        if a.isdigit() and not b.isdigit():
            label, idx = b, int(a)
        elif b.isdigit() and not a.isdigit():
            label, idx = a, int(b)
        else:
            try:
                idx = int(b); label = a
            except Exception:
                raise ValueError(f"Cannot parse dict line: {ln}")
        m[label] = idx
    return m

def triples_to_ids(triple_path: Path, ent2id, rel2id):
    T=[]
    for ln in read_lines(triple_path):
        if not ln.strip(): continue
        parts = ln.split()
        if len(parts)!=3: continue
        h,r,t = parts
        if h in ent2id and r in rel2id and t in ent2id:
            T.append((ent2id[h], rel2id[r], ent2id[t]))
    return T

def write_triples(path: Path, triples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h,r,t in triples:
            f.write(f"{h} {r} {t}\n")

def make_noise_ids(data_name: str, noise_rate: int):
    d = DATASET_ROOT / data_name
    ent2id = _read_dict_any_order(d/"entities.dict")
    rel2id = _read_dict_any_order(d/"relations.dict")
    train_ids = triples_to_ids(d/"train.txt", ent2id, rel2id)
    test_ids  = triples_to_ids(d/"test.txt",  ent2id, rel2id)
    rnd = random.Random(0)
    ent_ids = list(ent2id.values())
    # noisy train
    noisy=[]
    for h,r,t in train_ids:
        if rnd.random() < (noise_rate/100.0):
            t = rnd.choice(ent_ids)
        noisy.append((h,r,t))
    # negative test
    neg=[]
    for h,r,t in test_ids:
        t2 = rnd.choice(ent_ids)
        neg.append((h,r,t2))
    write_triples(d/f"noise_{noise_rate}.txt", noisy)
    write_triples(d/f"test_negative_{noise_rate}.txt", neg)

def dataset_stats(dpath: Path):
    ents = count_nonempty_lines(dpath/"entities.dict")
    rels = count_nonempty_lines(dpath/"relations.dict")
    train = count_nonempty_lines(dpath/"train.txt")
    valid = count_nonempty_lines(dpath/"valid.txt")
    test = count_nonempty_lines(dpath/"test.txt")
    return dict(entities=ents, relations=rels, train=train, valid=valid, test=test)

# ---- NEW: resolve checkpoint dir for selected settings ----
def _ckpt_dir(dataset: str, model: str, mode: str, noise_rate: int) -> Path:
    # training/repair saved here: checkpoint/<dataset>-<model>-<noise>-<mode>
    return REPO_ROOT / "checkpoint" / f"{dataset}-{model}-{noise_rate}-{mode}"

def locate_repair_outputs(dpath: Path, cpath: Path):
    patterns = [
        "repair*.txt", "repair*.csv",
        "*repaired*.txt", "*repaired*.csv",
        "*suggest*.txt", "*suggest*.csv",
        "*pror*.txt", "*pror*.csv",
    ]
    out = []
    # dataset/ (legacy)
    for pat in patterns:
        out += [Path(p) for p in glob.glob(str(dpath/pat))]
    # checkpoint/ (ZHEClean repair output)
    out += [p for p in cpath.glob("clean_triples_*.txt")]

    uniq = []
    seen = set()
    for p in out:
        if p.resolve() not in seen:
            uniq.append(p)
            seen.add(p.resolve())
    return uniq

def parse_repair_file(path: Path, sample_n=10):
    lines = [ln for ln in read_lines(path) if ln.strip()]
    count = len(lines)
    samples = lines[:sample_n]
    return count, samples

def try_parse_detector_scores(dpath: Path, cpath: Path, noise_rate: int):
    """
    Determine detected clean/dirty counts.

    Priority:
      1) CSV/TSV score files in dataset/ (backwards compatibility)
      2) Text files with 'clean'/'dirty' in name in dataset/
      3) ZHEClean predicted errors in checkpoint/: error_triples_{rate}.txt
         -> dirty = its line count
         -> clean = (len(test.txt) + len(test_negative_{rate}.txt)) - dirty
    """
    candidates = []
    for pat in ["*score*.csv", "*prob*.csv", "*pred*.csv", "*label*.csv",
                "*score*.tsv", "*prob*.tsv", "*pred*.tsv", "*label*.tsv",
                "*clean*.txt", "*dirty*.txt"]:
        candidates += [Path(p) for p in glob.glob(str(dpath/pat))]

    clean_ct = 0
    dirty_ct = 0
    got_any = False

    # Text clean/dirty files
    for p in candidates:
        if p.suffix.lower() == ".txt":
            name = p.name.lower()
            if "clean" in name:
                clean_ct += count_nonempty_lines(p); got_any = True
            if "dirty" in name:
                dirty_ct += count_nonempty_lines(p); got_any = True

    # CSV/TSV score files
    if not got_any:
        import csv
        for p in candidates:
            if p.suffix.lower() in [".csv", ".tsv"]:
                delim = "," if p.suffix.lower()==".csv" else "\t"
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        rdr = csv.DictReader(f, delimiter=delim)
                        c = d = 0
                        used = False
                        for row in rdr:
                            for key in row.keys():
                                lk = key.lower()
                                if any(k in lk for k in ["score","prob","confidence"]):
                                    try:
                                        v = float(row[key])
                                        if v >= 0.5: c += 1
                                        else: d += 1
                                        used = True
                                        break
                                    except:
                                        pass
                        if used:
                            clean_ct += c; dirty_ct += d; got_any = True
                except Exception:
                    pass

    if got_any:
        return {"clean": clean_ct, "dirty": dirty_ct}

    # Fallback: ZHEClean checkpoint predicted errors
    err_path = cpath / f"error_triples_{noise_rate}.txt"
    if err_path.exists():
        dirty = count_nonempty_lines(err_path)
        test_len = count_nonempty_lines(dpath / "test.txt")
        neg_path = dpath / f"test_negative_{noise_rate}.txt"
        neg_len = count_nonempty_lines(neg_path) if neg_path.exists() else 0
        total = test_len + neg_len
        clean = (total - dirty) if total > 0 else None
        return {"clean": clean, "dirty": dirty}

    return None

def make_report_pdf(txt: str, out_path: Path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    margin = 1.5*cm
    y = height - margin

    from textwrap import wrap
    for paragraph in txt.splitlines():
        if not paragraph.strip():
            y -= 0.4*cm
            continue
        wrapped = wrap(paragraph, width=110)
        for line in wrapped:
            c.setFont("Courier", 9.5)
            c.drawString(margin, y, line)
            y -= 0.45*cm
            if y < margin:
                c.showPage()
                y = height - margin
    c.save()

def build_quality_summary(data_name, model, mode, noise_rate, threads):
    d = DATASET_ROOT / data_name
    c = _ckpt_dir(data_name, model, mode, noise_rate)

    stats = dataset_stats(d)
    noise_file = d / f"noise_{noise_rate}.txt"
    train_used = noise_file if noise_file.exists() else (d/"train.txt")
    train_used_name = train_used.name
    train_count = count_nonempty_lines(train_used)

    det = try_parse_detector_scores(d, c, noise_rate)
    clean = det["clean"] if det else None
    dirty = det["dirty"] if det else None

    repair_files = locate_repair_outputs(d, c)
    repairs_total = 0
    repair_samples = []
    sample_file = None
    for rf in repair_files:
        cnt, smp = parse_repair_file(rf)
        repairs_total += cnt
        if sample_file is None and smp:
            sample_file = rf.name
            repair_samples = smp[:10]

    pct_clean = None
    if clean is not None and dirty is not None and (clean + dirty) and (clean >= 0) and (dirty >= 0):
        pct_clean = 100.0 * clean / (clean + dirty)

    return {
        "dataset": data_name,
        "timestamp": nice_time(),
        "model": model,
        "mode": mode,
        "noise_rate": noise_rate,
        "threads": threads,
        "entities": stats["entities"],
        "relations": stats["relations"],
        "train_lines": stats["train"],
        "train_used_file": train_used_name,
        "train_used_lines": train_count,
        "valid_lines": stats["valid"],
        "test_lines": stats["test"],
        "detected_clean": clean,
        "detected_dirty": dirty,
        "pct_clean": pct_clean,
        "repairs_total": repairs_total,
        "repair_sample_file": sample_file,
        "repair_samples": repair_samples,
    }

def format_report_text(info: dict, logs_tail: str = ""):
    lines = []
    lines.append("ZHEClean — ML-driven KG Quality Assessment")
    lines.append("="*60)
    lines.append(f"Date/Time: {info['timestamp']}")
    lines.append("")
    lines.append("Run Configuration")
    lines.append("-"*60)
    lines.append(f"Dataset          : {info['dataset']}")
    lines.append(f"Model / Mode     : {info['model']} / {info['mode']}")
    lines.append(f"Noise rate       : {info['noise_rate']}%")
    lines.append(f"CPU threads      : {info['threads']}")
    lines.append("")
    lines.append("Dataset Stats")
    lines.append("-"*60)
    lines.append(f"Entities         : {info['entities']}")
    lines.append(f"Relations        : {info['relations']}")
    lines.append(f"Train (raw)      : {info['train_lines']}")
    lines.append(f"Valid / Test     : {info['valid_lines']} / {info['test_lines']}")
    lines.append(f"Train used file  : {info['train_used_file']} ({info['train_used_lines']} lines)")
    lines.append("")
    lines.append("Detector Results")
    lines.append("-"*60)
    if info["detected_clean"] is None:
        lines.append("Detected clean/dirty: N/A (no detector score files found)")
    else:
        lines.append(f"Detected clean   : {info['detected_clean']}")
        lines.append(f"Detected dirty   : {info['detected_dirty']}")
        lines.append(f"Clean percentage : {info['pct_clean']:.2f}%")
    lines.append("")
    lines.append("Repair Suggestions")
    lines.append("-"*60)
    lines.append(f"Total suggestions: {info['repairs_total']}")
    if info["repair_samples"]:
        lines.append(f"Sample file      : {info['repair_sample_file']}")
        lines.append("Top suggestions:")
        for s in info["repair_samples"]:
            lines.append(f"  • {s}")
    else:
        lines.append("No repair samples found to preview.")
    if logs_tail:
        lines.append("")
        lines.append("Logs (tail)")
        lines.append("-"*60)
        lines += [ln for ln in logs_tail.splitlines()[-50:]]
    return "\n".join(lines)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Control Panel")
st.sidebar.write(f"**Repo:** `{REPO_ROOT}`")
st.sidebar.write(f"**Datasets:** `{DATASET_ROOT}`")

data_name = st.sidebar.text_input("Dataset name (folder under dataset/)", value="FB15K-237")
dpath = DATASET_ROOT / data_name
dpath.mkdir(parents=True, exist_ok=True)

with st.sidebar.expander("Upload dataset files", expanded=False):
    for fname, label in [
        ("train.txt","Train triples"),
        ("valid.txt","Valid triples"),
        ("test.txt","Test triples"),
        ("entities.dict","Entity dictionary"),
        ("relations.dict","Relation dictionary"),
    ]:
        up = st.file_uploader(f"{label} — {fname}", type=["txt"], key=fname)
        if up is not None:
            safe_write_bytes(dpath/fname, up.read())
            st.success(f"Saved {fname}")

with st.sidebar.expander("Quick start / Toy data", expanded=False):
    if st.button("Create tiny toy dataset"):
        ensure_toy_dataset(data_name)
        st.success(f"Toy dataset created under dataset/{data_name}/")

with st.sidebar.expander("Training Settings", expanded=True):
    model = st.selectbox("Backbone", ["TransE","RotatE"], index=0)
    # valid options for run.py are ['none','soft']
    mode  = st.selectbox("Mode", ["soft","none"], index=0)
    noise_rate = st.slider("Noise rate (%)", 0, 50, 0, 5)  # start at 0 to avoid missing noise files
    threads = st.slider("CPU threads", 1, 8, 4, 1)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

st.title("🧹 ZHEClean — Quality Assessment UI")
st.caption("Generate noise → Train detector → Repair errors. Live logs stream below. Quality metrics & Report are at the bottom.")

# ---------- Actions ----------
colA, colB, colC = st.columns(3)

with colA:
    if st.button("1) Generate Noise"):
        try:
            # Always generate ID-based noise that run.py expects
            make_noise_ids(data_name, noise_rate)
            st.success(f"Noise files created: dataset/{data_name}/noise_{noise_rate}.txt and test_negative_{noise_rate}.txt")
        except Exception as e:
            st.error("Noise generation failed. Check that entities.dict/relations.dict and train/test exist.")
            st.exception(e)

with colB:
    if st.button("2) Train Detector"):
        run_py = REPO_ROOT / "run.py"
        if not run_py.exists():
            st.error("run.py not found in repo root.")
        else:
            st.info(f"Training {model}/{mode} at noise={noise_rate}% …")
            rc, logs = stream_subprocess(
                [sys.executable, str(run_py),
                 "--noise_rate", str(noise_rate),
                 "--mode", mode,
                 "--data_name", data_name,
                 "--model", model],
                REPO_ROOT,
            )
            st.session_state["last_train_logs"] = logs
            if rc == 0:
                st.success("Training completed.")
            else:
                st.error("Training failed. Check logs below.")

with colC:
    if st.button("3) Repair Errors"):
        # Common script names (the original README had minor typos in some forks)
        candidates = ["repair_error.py", "erpair_error.py", "erpai_error.py"]
        rep_script = None
        for c in candidates:
            p = REPO_ROOT / c
            if p.exists():
                rep_script = p
                break
        if not rep_script:
            st.error("Repair script not found (tried repair_error.py / erpair_error.py / erpai_error.py).")
        else:
            st.info(f"Repairing with {model} …")
            # IMPORTANT: pass train_error_rate so it matches the trained checkpoint folder
            rc, logs = stream_subprocess(
                [sys.executable, str(rep_script),
                 "--data_name", data_name,
                 "--model", model,
                 "--train_error_rate", str(noise_rate)],
                REPO_ROOT,
            )
            st.session_state["last_repair_logs"] = logs
            if rc == 0:
                st.success("Repair completed.")
            else:
                st.error("Repair failed. Check logs below.")

st.divider()

# ---------- Quality Score Panel ----------
st.subheader("📊 Quality Score & Outputs")

# Gather info (now uses checkpoint dir under the hood)
info = build_quality_summary(data_name, model, mode, noise_rate, threads)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Entities", info["entities"])
m2.metric("Relations", info["relations"])
m3.metric("Train used", f"{info['train_used_lines']:,}", help=info["train_used_file"])
if info["pct_clean"] is not None:
    m4.metric("Clean % (detected)", f"{info['pct_clean']:.2f}%")
else:
    m4.metric("Clean % (detected)", "N/A", help="Could not find detector outputs in dataset/ or checkpoint/.")

n1, n2 = st.columns(2)
n1.metric("Detected Clean", info["detected_clean"] if info["detected_clean"] is not None else "N/A")
n2.metric("Detected Dirty", info["detected_dirty"] if info["detected_dirty"] is not None else "N/A")

r1, r2 = st.columns([1, 2])
r1.metric("Repair Suggestions", info["repairs_total"])
if info["repair_samples"]:
    r2.write("**Sample Repair Suggestions**" + (f" (from {info['repair_sample_file']})" if info["repair_sample_file"] else ""))
    st.code("\n".join(info["repair_samples"]))

# Tiny charts (no custom colors)
if info["detected_clean"] is not None and info["detected_dirty"] is not None:
    chart_df = pd.DataFrame({
        "label": ["clean","dirty"],
        "count": [info["detected_clean"], info["detected_dirty"]],
    })
    st.bar_chart(chart_df.set_index("label"))

# ---------- Recent files viewer ----------
st.subheader("📂 Recent Files (dataset folder)")
recent_files = []
for p in sorted(dpath.rglob("*"), key=lambda x: x.stat().st_mtime if x.is_file() else 0, reverse=True):
    if p.is_file():
        recent_files.append({
            "file": str(p.relative_to(REPO_ROOT)),
            "KB": round(p.stat().st_size/1024,1),
            "modified": nice_time(p.stat().st_mtime),
        })
df = pd.DataFrame(recent_files[:200])
if df.empty:
    st.info("No files yet. After training/repair, refresh to see outputs.")
else:
    st.dataframe(df, use_container_width=True)
    sel = st.selectbox("Preview file", [""] + df["file"].tolist())
    if sel:
        fpath = REPO_ROOT / sel
        try:
            size = fpath.stat().st_size
            st.write(f"**Size:** {round(size/1024,1)} KB")
            if size <= 500_000:
                st.code(fpath.read_text(errors="ignore")[-8000:])
            with open(fpath, "rb") as fh:
                st.download_button("Download", fh, file_name=fpath.name)
        except Exception as e:
            st.error(f"Cannot open file: {e}")

st.divider()

# ---------- Report generation ----------
st.subheader("🧾 One-Click Report")
logs_tail = ""
for k in ["last_train_logs", "last_repair_logs"]:
    if k in st.session_state and st.session_state[k]:
        logs_tail += st.session_state[k]

report_text = format_report_text(info, logs_tail=logs_tail)
st.text_area("Report preview (editable before export)", report_text, height=260, key="report_preview")

colR1, colR2 = st.columns(2)
with colR1:
    if st.button("Download TXT Report"):
        txt_path = RUNS_ROOT / f"zheclean_report_{data_name}_{int(time.time())}.txt"
        txt_path.write_text(st.session_state["report_preview"], encoding="utf-8")
        with open(txt_path, "rb") as fh:
            st.download_button("Save TXT", fh, file_name=f"{txt_path.name}", key="dl_txt")

with colR2:
    if st.button("Download PDF Report"):
        pdf_path = RUNS_ROOT / f"zheclean_report_{data_name}_{int(time.time())}.pdf"
        try:
            make_report_pdf(st.session_state["report_preview"], pdf_path)
            with open(pdf_path, "rb") as fh:
                st.download_button("Save PDF", fh, file_name=f"{pdf_path.name}", key="dl_pdf")
        except Exception as e:
            st.error("PDF export failed. Install: pip install reportlab==3.6.12")
            st.exception(e)

with st.expander("ℹ️ How the Quality Score works"):
    st.markdown(
        """
- The app writes **ID-based** noise files so `run.py` can load them directly.
- It looks for outputs both in **dataset/** *and* the matching **checkpoint/** folder.
- CSV/TSV with a **score/prob/confidence** column are thresholded at **0.5** to estimate clean/dirty counts.
- If no CSVs are found, it uses `checkpoint/.../error_triples_RATE.txt` to set **Detected Dirty**, and
  computes **Detected Clean** as `(len(test.txt) + len(test_negative_RATE.txt)) - dirty`.
- Repair suggestions are counted from files that match **repair\***, **\*repaired\***, **\*suggest\***, **\*pror\***,
  plus ZHEClean’s `clean_triples_*.txt` in the checkpoint folder.
        """
    )
