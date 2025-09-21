# app.py — Streamlit UI for ZHEClean with Quality Score & PDF report
# Run:  streamlit run app.py   (inside your Python 3.7 venv)
# Deps: streamlit==1.25.0 pandas==1.3.5 reportlab==3.6.12 (reportlab optional)

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

def try_int(x, default=0):
    try:
        return int(x)
    except:
        return default

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

def fallback_make_noise(data_name: str, noise_rate: int):
    d = DATASET_ROOT / data_name
    ents = [ln.split("\t")[0] for ln in read_lines(d/"entities.dict") if "\t" in ln]
    train = [ln.split() for ln in read_lines(d/"train.txt") if ln.strip()]
    # noisy train
    out = []
    rnd = random.Random(0)
    for h,r,t in train:
        if rnd.random() < (noise_rate/100.0):
            t = rnd.choice(ents) if ents else t
        out.append(f"{h} {r} {t}")
    (d/f"noise_{noise_rate}.txt").write_text("\n".join(out)+"\n", encoding="utf-8")
    # negative test
    test = [ln.split() for ln in read_lines(d/"test.txt") if ln.strip()]
    neg = []
    for h,r,t in test:
        t2 = rnd.choice(ents) if ents else t
        neg.append(f"{h} {r} {t2}")
    (d/f"test_negative_{noise_rate}.txt").write_text("\n".join(neg)+"\n", encoding="utf-8")

def dataset_stats(dpath: Path):
    ents = count_nonempty_lines(dpath/"entities.dict")
    rels = count_nonempty_lines(dpath/"relations.dict")
    train = count_nonempty_lines(dpath/"train.txt")
    valid = count_nonempty_lines(dpath/"valid.txt")
    test = count_nonempty_lines(dpath/"test.txt")
    return dict(entities=ents, relations=rels, train=train, valid=valid, test=test)

def locate_repair_outputs(dpath: Path):
    """
    Find likely repair suggestion files.
    Tries patterns like: repair*.txt/csv, *repaired*.txt/csv, *suggest*.txt/csv.
    Returns list[Path]
    """
    patterns = [
        "repair*.txt", "repair*.csv",
        "*repaired*.txt", "*repaired*.csv",
        "*suggest*.txt", "*suggest*.csv",
        "*pror*.txt", "*pror*.csv",
    ]
    out = []
    for pat in patterns:
        out += [Path(p) for p in glob.glob(str(dpath/pat))]
    # de-dup
    uniq = []
    seen = set()
    for p in out:
        if p.resolve() not in seen:
            uniq.append(p)
            seen.add(p.resolve())
    return uniq

def parse_repair_file(path: Path, sample_n=10):
    """
    Very generic parser: each non-empty line is one suggestion.
    If CSV/TSV with columns, try to infer cols (h,r,t,h',r',t',score).
    Returns (count, samples[str]).
    """
    lines = [ln for ln in read_lines(path) if ln.strip()]
    count = len(lines)
    samples = lines[:sample_n]
    return count, samples

def try_parse_detector_scores(dpath: Path):
    """
    Try to find detector outputs to estimate #clean/#dirty.
    Looks for files with names containing 'score', 'prob', 'pred', 'clean', 'dirty'.
    Supports CSV/TSV where a 'score' or 'prob' column exists; threshold at 0.5.
    Returns dict(clean=int, dirty=int) or None if not found.
    """
    candidates = []
    for pat in ["*score*.csv", "*prob*.csv", "*pred*.csv", "*label*.csv",
                "*score*.tsv", "*prob*.tsv", "*pred*.tsv", "*label*.tsv",
                "*clean*.txt", "*dirty*.txt"]:
        candidates += [Path(p) for p in glob.glob(str(dpath/pat))]
    # Text lists
    clean_ct = 0
    dirty_ct = 0
    got_any = False
    for p in candidates:
        if p.suffix.lower() in [".txt"]:
            if "clean" in p.name.lower():
                clean_ct += count_nonempty_lines(p); got_any = True
            if "dirty" in p.name.lower():
                dirty_ct += count_nonempty_lines(p); got_any = True
    # CSV/TSV with 'score/prob' try
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
    return None

def make_report_pdf(txt: str, out_path: Path):
    """
    Create a simple PDF using reportlab. If reportlab missing, raise ImportError.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    margin = 1.5*cm
    max_width = width - 2*margin
    y = height - margin

    # Simple monospaced wrapping
    from textwrap import wrap
    for paragraph in txt.splitlines():
        if not paragraph.strip():
            y -= 0.4*cm
            continue
        wrapped = wrap(paragraph, width=110)  # adjust chars/line
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
    stats = dataset_stats(d)

    # Which train file was used?
    noise_file = d / f"noise_{noise_rate}.txt"
    train_used = noise_file if noise_file.exists() else (d/"train.txt")
    train_used_name = train_used.name
    train_count = count_nonempty_lines(train_used)

    # Detector-derived clean/dirty if we can find them
    det = try_parse_detector_scores(d)
    if det:
        clean = det["clean"]
        dirty = det["dirty"]
    else:
        # Fallback: unknown split — show N/A
        clean = None
        dirty = None

    # Repair suggestions
    repair_files = locate_repair_outputs(d)
    repairs_total = 0
    repair_samples = []
    sample_file = None
    for rf in repair_files:
        cnt, smp = parse_repair_file(rf)
        repairs_total += cnt
        if sample_file is None and smp:
            sample_file = rf.name
            repair_samples = smp[:10]

    # Compute some simple percentages
    pct_clean = None
    if clean is not None and dirty is not None and (clean + dirty) > 0:
        pct_clean = 100.0 * clean / (clean + dirty)

    info = {
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
    return info

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
    mode  = st.selectbox("Mode", ["soft","hard"], index=0)
    noise_rate = st.slider("Noise rate (%)", 0, 50, 20, 5)
    threads = st.slider("CPU threads", 1, 8, 4, 1)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

st.title("🧹 ZHEClean — Quality Assessment UI")
st.caption("Generate noise → Train detector → Repair errors. Live logs stream below. Quality metrics & Report are at the bottom.")

# ---------- Actions ----------
colA, colB, colC = st.columns(3)

with colA:
    if st.button("1) Generate Noise"):
        gen_py = REPO_ROOT / "generate_noise.py"
        if gen_py.exists():
            st.info("Running repository noise generator…")
            rc, logs = stream_subprocess(
                [sys.executable, str(gen_py), "--data_name", data_name, "--noise_rate", str(noise_rate)],
                REPO_ROOT,
            )
            st.session_state["last_noise_logs"] = logs
            if rc == 0:
                st.success("Noise files created.")
            else:
                st.error("Noise generator failed. Using fallback…")
                try:
                    fallback_make_noise(data_name, noise_rate)
                    st.success("Fallback noise files created.")
                except Exception as e:
                    st.exception(e)
        else:
            st.warning("generate_noise.py not found. Using fallback noise creator.")
            try:
                fallback_make_noise(data_name, noise_rate)
                st.success("Fallback noise files created.")
            except Exception as e:
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
                st.error("Training failed. Check logs.")

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
            rc, logs = stream_subprocess(
                [sys.executable, str(rep_script), "--data_name", data_name, "--model", model],
                REPO_ROOT,
            )
            st.session_state["last_repair_logs"] = logs
            if rc == 0:
                st.success("Repair completed.")
            else:
                st.error("Repair failed. Check logs.")

st.divider()

# ---------- Quality Score Panel ----------
st.subheader("📊 Quality Score & Outputs")

# Gather info
info = build_quality_summary(data_name, model, mode, noise_rate, threads)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Entities", info["entities"])
m2.metric("Relations", info["relations"])
m3.metric("Train used", f"{info['train_used_lines']:,}", help=info["train_used_file"])
if info["pct_clean"] is not None:
    m4.metric("Clean % (detected)", f"{info['pct_clean']:.2f}%")
else:
    m4.metric("Clean % (detected)", "N/A", help="Could not find detector score files.")

n1, n2 = st.columns(2)
n1.metric("Detected Clean", info["detected_clean"] if info["detected_clean"] is not None else "N/A")
n2.metric("Detected Dirty", info["detected_dirty"] if info["detected_dirty"] is not None else "N/A")

r1, r2 = st.columns([1, 2])
r1.metric("Repair Suggestions", info["repairs_total"])
if info["repair_samples"]:
    r2.write("**Sample Repair Suggestions**" + (f" (from {info['repair_sample_file']})" if info["repair_sample_file"] else ""))
    st.code("\n".join(info["repair_samples"]))

# Tiny charts (no custom colors)
if info["detected_clean"] is not None:
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
            st.download_button("Save TXT", fh, file_name=txt_path.name, key="dl_txt")

with colR2:
    if st.button("Download PDF Report"):
        pdf_path = RUNS_ROOT / f"zheclean_report_{data_name}_{int(time.time())}.pdf"
        try:
            make_report_pdf(st.session_state["report_preview"], pdf_path)
            with open(pdf_path, "rb") as fh:
                st.download_button("Save PDF", fh, file_name=pdf_path.name, key="dl_pdf")
        except Exception as e:
            st.error("PDF export failed. Is 'reportlab' installed? (pip install reportlab)")
            st.exception(e)

with st.expander("ℹ️ How the Quality Score works"):
    st.markdown(
        """
- The app scans your *dataset folder* for typical detector output files: names containing
  **score/prob/pred/clean/dirty** in `.csv/.tsv/.txt`.
- If it finds CSV/TSV with a column like **score/prob/confidence**, it thresholds at **0.5** to estimate
  **clean vs dirty counts**.
- If it finds text files named like **clean*.txt** or **dirty*.txt**, it counts their lines.
- Repair suggestions are counted from files that match **repair\***, **\*repaired\***, **\*suggest\*** or **\*pror\***.
- If your repo uses different filenames, you can still preview/download outputs above, and the report will include
  your run config + sample lines from the found repair file.
        """
    )

