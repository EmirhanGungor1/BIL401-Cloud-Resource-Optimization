#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIL401 - Atıl kaynak tespiti ve FinOps odaklı kapasite optimizasyonu: analiz hattı (ilk sürüm)

Alt komutlar:
  azure     Azure Public Dataset V2 (vmtable) üzerinde israf sınıflandırma + right-sizing
  google    Google ClusterData2011 (task_events + task_usage) üzerinde israf + zaman bölmeli doğrulama
  selftest  Sentetik veriyle birim ve uçtan uca sınamalar (gerçek veri gerekmez)

ÖNEMLİ:
  * Fiyat (--price-per-vcpu-hour) kullanıcı tarafından verilir; koda gömülü değildir.
    AWS EC2 on-demand sayfasından seçtiğiniz örnek tipi, bölge ve erişim tarihini raporda belirtin.
  * Sütun sırası resmî şema belgelerinden (Azure: AzurePublicDataset; Google: clusterdata format+schema) doğrulanmalıdır.
  * selftest çıktısındaki sayılar SENTETİKTİR; rapora bulgu olarak yazılamaz.
"""
import argparse, glob, json, os, sys, tempfile
import numpy as np
import pandas as pd

AZURE_COLS = ["vmid", "subscriptionid", "deploymentid", "vmcreated", "vmdeleted", "maxcpu", "avgcpu",
              "p95maxcpu", "vmcategory", "vmcorebucket", "vmmemorybucket"]
GOOGLE_EVENT_COLS = ["time", "missing", "job_id", "task_index", "machine_id", "event_type", "user",
                     "sched_class", "priority", "cpu_request", "mem_request", "disk_request", "diff_machine"]
GOOGLE_USAGE_COLS = ["start", "end", "job_id", "task_index", "machine_id", "cpu_mean", "mem_canonical",
                     "mem_assigned", "unmapped_cache", "page_cache", "mem_max", "disk_io_mean",
                     "disk_space", "cpu_max", "disk_io_max", "cpi", "mai", "sample_portion",
                     "agg_type", "cpu_sampled"]
CLASSES = ["Atıl (<%10)", "Düşük (%10-30)", "Uygun (%30-60)", "Yüksek (>=%60)"]
EDGES = [0.10, 0.30, 0.60]
WINDOW_SEC = 300  # Google kullanım penceresi: 300 sn


# ----------------------------------------------------------------- çekirdek fonksiyonlar
def check_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Eksik sütun(lar): %s" % ", ".join(missing))


def clean(df, cols_nonneg):
    """NaN ve negatif değerleri eler; (temiz_df, atılan_satır_sayısı) döndürür."""
    n0 = len(df)
    df = df.dropna(subset=cols_nonneg)
    for c in cols_nonneg:
        df = df[df[c] >= 0]
    return df, n0 - len(df)


def classify(p95_frac):
    """p95 kullanım oranı (0-1) -> sınıf indeksi 0..3 (sınırlar: 0.10, 0.30, 0.60; alt sınır dahil)."""
    return np.searchsorted(np.asarray(EDGES), np.asarray(p95_frac, dtype=float), side="right")


def rightsize(R, p95_frac, u_star=0.6, headroom=0.0, buckets=None):
    """R' = min(R, R*p95*(1+h)/u*); buckets verilirse en yakın büyük boyuta yuvarlanır. Asla büyütmez."""
    R = np.asarray(R, dtype=float)
    need = R * np.asarray(p95_frac, dtype=float) * (1.0 + headroom) / u_star
    if buckets is not None:
        b = np.sort(np.asarray(buckets, dtype=float))
        idx = np.searchsorted(b, need - 1e-12, side="left")
        need = np.where(idx < len(b), b[np.minimum(idx, len(b) - 1)], R)
    return np.minimum(R, need)


def ffd(sizes, capacity):
    """First Fit Decreasing; kullanılan kutu (sunucu) sayısı ve kutu doluluklarını döndürür."""
    sizes = sorted([float(s) for s in sizes], reverse=True)
    if sizes and sizes[0] > capacity + 1e-12:
        raise ValueError("Bir iş yükü sunucu kapasitesinden büyük")
    # Klasik FFD ile aynı yerleştirme (en soldaki uygun kutu); kalan kapasiteler segment ağacında
    # tutulduğundan her arama O(log n): büyük izleme kayıtlarında O(n·kutu) taramadan kaçınılır.
    n = len(sizes)
    if n == 0:
        return 0, []
    size = 1
    while size < n:
        size *= 2
    tree = [float(capacity)] * (2 * size)   # boş kutular tam kapasiteli
    loads = [0.0] * n
    used = 0
    eps = 1e-12
    for s in sizes:
        i = 1
        while i < size:                      # kalan kapasitesi s'ye yeten en soldaki yaprağa in
            i = 2 * i if tree[2 * i] + eps >= s else 2 * i + 1
        b = i - size
        loads[b] += s
        used = max(used, b + 1)
        tree[i] = capacity - loads[b]
        i //= 2
        while i:
            tree[i] = max(tree[2 * i], tree[2 * i + 1]); i //= 2
    return used, loads[:used]


def consolidate(start, end, R, Rp, capacity, max_util=0.8, n_snapshots=24, times=None):
    """Consolidation simülasyonu: seçilen zaman anlarında (snapshot) aynı anda çalışan iş yüklerini
    right-sizing öncesi (R) ve sonrası (R') boyutlarıyla FFD'ye verir; gereken sunucu sayılarını döndürür.
    Etkin sunucu kapasitesi = capacity * max_util. Etkin kapasiteden büyük iş yükleri FFD'ye girmez,
    her biri ayrı bir sunucuya konur ve 'kapasite_ustu' sütununda raporlanır."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    R, Rp = np.asarray(R, dtype=float), np.asarray(Rp, dtype=float)
    cap = float(capacity) * float(max_util)
    if times is None:
        times = np.linspace(start.min(), end.max(), n_snapshots + 2)[1:-1]
    rows = []
    for t in times:
        alive = (start <= t) & (end > t)
        if not alive.any():
            continue
        rec = {"zaman": float(t), "is_yuku": int(alive.sum())}
        for tag, sizes in (("once", R[alive]), ("sonra", Rp[alive])):
            over = sizes > cap + 1e-12
            n, _ = ffd(sizes[~over], cap)
            rec["sunucu_" + tag] = int(n + over.sum())
            rec["kapasite_ustu_" + tag] = int(over.sum())
        rows.append(rec)
    return pd.DataFrame(rows, columns=["zaman", "is_yuku", "sunucu_once", "kapasite_ustu_once",
                                       "sunucu_sonra", "kapasite_ustu_sonra"])


def consolidation_summary(cdf):
    if cdf.empty or cdf["sunucu_once"].sum() == 0:
        return {"konsolidasyon_anlik_sayisi": 0}
    return {"konsolidasyon_anlik_sayisi": int(len(cdf)),
            "sunucu_once_ort": float(cdf["sunucu_once"].mean()),
            "sunucu_sonra_ort": float(cdf["sunucu_sonra"].mean()),
            "sunucu_azalma_orani": 1.0 - float(cdf["sunucu_sonra"].sum()) / float(cdf["sunucu_once"].sum())}


def cost_saving(R, Rp, hours, price):
    """(R-R')*saat*fiyat toplamı. Fiyat kullanıcı girdisidir."""
    return float(np.sum((np.asarray(R) - np.asarray(Rp)) * np.asarray(hours)) * price)


def weighted_waste(R, hours, U):
    R, hours, U = map(lambda a: np.asarray(a, dtype=float), (R, hours, U))
    d = np.sum(R * hours)
    return float(np.sum(R * hours * (1 - U)) / d) if d > 0 else float("nan")


def split_eval(series, split_time, u_star, headroom, R_by_key):
    """Zaman bölmeli doğrulama. series: {key: (starts, usage)}. Öneri yalnızca eğitim penceresinden üretilir."""
    tot_R = tot_Rp = viol = n_test = 0.0
    for k, (st, us) in series.items():
        st = np.asarray(st); us = np.asarray(us, dtype=float)
        tr = us[st < split_time]; te = us[st >= split_time]
        if len(tr) == 0 or len(te) == 0:
            continue
        R = R_by_key[k]
        Rp = min(R, np.percentile(tr, 95) * (1 + headroom) / u_star)
        tot_R += R; tot_Rp += Rp
        viol += np.sum(te > Rp + 1e-12); n_test += len(te)
    saving = 1 - tot_Rp / tot_R if tot_R > 0 else float("nan")
    return saving, (viol / n_test if n_test else float("nan"))


# ----------------------------------------------------------------- Azure
def parse_bucket(x):
    s = str(x).strip()
    if s.startswith(">"):
        return float(s[1:]) * 1.33
    return float(s)


def azure_run(vmtable, out, sample=None, u_star=0.6, price=None, seed=0,
              server_cores=64.0, max_util=0.8, n_snapshots=24):
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(vmtable, header=None, names=AZURE_COLS, nrows=sample)
    check_columns(df, AZURE_COLS)
    df, dropped = clean(df, ["avgcpu", "p95maxcpu", "maxcpu", "vmcreated", "vmdeleted"])
    df = df[df["vmdeleted"] > df["vmcreated"]].copy()
    df["cores"] = df["vmcorebucket"].map(parse_bucket)
    df["hours"] = (df["vmdeleted"] - df["vmcreated"]) / 3600.0
    df["p95"] = (df["p95maxcpu"] / 100.0).clip(0, 1)
    df["U"] = (df["avgcpu"] / 100.0).clip(0, 1)
    df["sinif"] = classify(df["p95"].values)
    df["core_hours"] = df["cores"] * df["hours"]
    buckets = sorted(df["cores"].unique())
    df["cores_new"] = rightsize(df["cores"], df["p95"], u_star, 0.0, buckets)
    shares = df.groupby("sinif")["core_hours"].sum() / df["core_hours"].sum()
    shares = shares.reindex(range(4), fill_value=0.0)
    pd.DataFrame({"sinif": CLASSES, "cekirdek_saat_payi": shares.values}).to_csv(os.path.join(out, "azure_siniflar.csv"), index=False)
    saved_ch = float(((df["cores"] - df["cores_new"]) * df["hours"]).sum())
    summ = {"vm_sayisi": int(len(df)), "atilan_satir": int(dropped), "u_star": u_star,
            "agirlikli_israf_W": weighted_waste(df["cores"], df["hours"], df["U"]),
            "kapasite_azalma_orani": saved_ch / float(df["core_hours"].sum()),
            "tasarruf_cekirdek_saat": saved_ch}
    if price is not None:
        summ["tasarruf_para"] = saved_ch * price
    # Consolidation (B planı): makine bilgisi yok -> varsayımsal homojen sunucu (server_cores çekirdek)
    cdf = consolidate(df["vmcreated"], df["vmdeleted"], df["cores"], df["cores_new"],
                      server_cores, max_util, n_snapshots)
    cdf.to_csv(os.path.join(out, "azure_konsolidasyon.csv"), index=False)
    _plot_consolidation(cdf, out, "azure")
    summ.update({"sunucu_cekirdek": float(server_cores), "azami_sunucu_kullanim": float(max_util)})
    summ.update(consolidation_summary(cdf))
    pd.Series(summ).to_csv(os.path.join(out, "azure_ozet.csv"), header=["deger"])
    sens = []
    for u in [0.4, 0.5, 0.6, 0.7, 0.8]:
        c2 = rightsize(df["cores"], df["p95"], u, 0.0, buckets)
        sch = float(((df["cores"] - c2) * df["hours"]).sum())
        row = {"u_star": u, "kapasite_azalma_orani": sch / float(df["core_hours"].sum()), "tasarruf_cekirdek_saat": sch}
        if price is not None:
            row["tasarruf_para"] = sch * price
        sens.append(row)
    pd.DataFrame(sens).to_csv(os.path.join(out, "azure_duyarlilik.csv"), index=False)
    df.groupby("vmcategory")["core_hours"].sum().to_csv(os.path.join(out, "azure_kategori.csv"))
    _plots(df["p95"].values, shares.values, out, "azure", share_label="Çekirdek-saat payı (%)")
    return summ


# ----------------------------------------------------------------- Google
def google_run(events_glob, usage_glob, out, job_mod=10, min_windows=10, split=0.7, headrooms=(0, .1, .2, .3, .5), price=None,
               u_star=0.6, server_capacity=1.0, max_util=0.8, n_snapshots=24):
    os.makedirs(out, exist_ok=True)
    req = {}
    for f in sorted(glob.glob(events_glob)):
        for ch in pd.read_csv(f, header=None, names=GOOGLE_EVENT_COLS, usecols=["job_id", "task_index", "event_type", "cpu_request"], chunksize=500000):
            ch = ch[(ch["event_type"] == 0) & (ch["job_id"] % job_mod == 0) & (ch["cpu_request"] > 0)]
            for j, t, r in zip(ch["job_id"], ch["task_index"], ch["cpu_request"]):
                req.setdefault((j, t), float(r))
    rows = []
    for f in sorted(glob.glob(usage_glob)):
        for ch in pd.read_csv(f, header=None, names=GOOGLE_USAGE_COLS, usecols=["start", "job_id", "task_index", "cpu_mean"], chunksize=500000):
            ch = ch[ch["job_id"] % job_mod == 0]
            rows.append(ch)
    if not rows or not req:
        raise SystemExit("Eşleşen kayıt bulunamadı: dosya desenlerini ve --job-mod değerini kontrol edin.")
    us = pd.concat(rows, ignore_index=True)
    us, dropped = clean(us, ["cpu_mean", "start"])
    series = {}
    for (j, t), g in us.groupby(["job_id", "task_index"]):
        if (j, t) in req and len(g) >= min_windows:
            g = g.sort_values("start")
            series[(j, t)] = (g["start"].values, g["cpu_mean"].values)
    if not series:
        raise SystemExit("Yeterli pencereli görev bulunamadı (--min-windows).")
    t0 = min(s[0][0] for s in series.values()); t1 = max(s[0][-1] for s in series.values())
    split_time = t0 + split * (t1 - t0)
    keys = list(series); R = np.array([req[k] for k in keys])
    n_w = np.array([len(series[k][0]) for k in keys]); hours = n_w * WINDOW_SEC / 3600.0
    mean_u = np.array([series[k][1].mean() for k in keys]); p95 = np.array([np.percentile(series[k][1], 95) for k in keys])
    U = np.clip(mean_u / R, 0, 1); p95f = np.clip(p95 / R, 0, 1)
    sinif = classify(p95f)
    sh = pd.Series(R * hours).groupby(sinif).sum().reindex(range(4), fill_value=0.0); sh = sh / sh.sum()
    pd.DataFrame({"sinif": CLASSES, "talep_saat_payi": sh.values}).to_csv(os.path.join(out, "google_siniflar.csv"), index=False)
    summ = {"gorev_sayisi": len(keys), "atilan_satir": int(dropped), "agirlikli_israf_W": weighted_waste(R, hours, U)}
    R_by = {k: r for k, r in zip(keys, R)}
    sens = []
    for u in [0.4, 0.5, 0.6, 0.7, 0.8]:
        s, v = split_eval(series, split_time, u, 0.0, R_by); sens.append({"u_star": u, "kapasite_azalma": s, "ihlal_orani": v})
    pd.DataFrame(sens).to_csv(os.path.join(out, "google_duyarlilik.csv"), index=False)
    trade = []
    for h in headrooms:
        s, v = split_eval(series, split_time, 1.0, h, R_by); trade.append({"headroom": h, "kapasite_azalma": s, "ihlal_orani": v})
    td = pd.DataFrame(trade); td.to_csv(os.path.join(out, "google_tasarruf_ihlal.csv"), index=False)
    # Consolidation: R' yalnızca eğitim penceresinden (veri sızıntısı yok); yerleştirme test penceresindeki anlarda.
    # Talepler en büyük makineye göre normalize olduğundan varsayılan sunucu kapasitesi 1.0'dır.
    Rp = np.empty(len(keys))
    for i, k in enumerate(keys):
        st, u = series[k]; tr = u[st < split_time]
        Rp[i] = min(R[i], np.percentile(tr, 95) / u_star) if len(tr) else R[i]
    t_start = np.array([series[k][0][0] for k in keys], dtype=float)
    t_end = np.array([series[k][0][-1] for k in keys], dtype=float) + WINDOW_SEC
    snap = np.linspace(split_time, t_end.max(), n_snapshots + 2)[1:-1]
    cdf = consolidate(t_start, t_end, R, Rp, server_capacity, max_util, times=snap)
    cdf.to_csv(os.path.join(out, "google_konsolidasyon.csv"), index=False)
    _plot_consolidation(cdf, out, "google")
    summ.update({"u_star": u_star, "sunucu_kapasitesi": float(server_capacity), "azami_sunucu_kullanim": float(max_util)})
    summ.update(consolidation_summary(cdf))
    pd.Series(summ).to_csv(os.path.join(out, "google_ozet.csv"), header=["deger"])
    _plots(p95f, sh.values, out, "google", share_label="Talep-saat payı (%)")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3.4)); plt.plot(td["ihlal_orani"] * 100, td["kapasite_azalma"] * 100, "o-")
        for _, r in td.iterrows():
            plt.annotate("h=%.1f" % r["headroom"], (r["ihlal_orani"] * 100, r["kapasite_azalma"] * 100), fontsize=7)
        plt.xlabel("Test penceresi ihlal oranı (%)"); plt.ylabel("Kapasite azalması (%)"); plt.tight_layout()
        plt.savefig(os.path.join(out, "google_tasarruf_ihlal.png"), dpi=200); plt.close()
    except Exception as e:  # grafik zorunlu değil
        print("grafik atlandı:", e)
    return summ


def _tr_thousands(v, _pos=None):
    return "{:,.0f}".format(v).replace(",", ".")


def _plots(p95, shares, out, tag, share_label="Pay (%)", edges=EDGES):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(np.asarray(p95) * 100, bins=40, color="#3b6ea5")
        for e in edges:  # israf sınıfı sınırları
            ax.axvline(e * 100, color="#555555", ls="--", lw=0.8)
        ax.set_xlabel("p95 CPU kullanım oranı (%)"); ax.set_ylabel("İş yükü sayısı")
        ax.yaxis.set_major_formatter(FuncFormatter(_tr_thousands))
        fig.tight_layout(); fig.savefig(os.path.join(out, tag + "_cpu_dagilim.png"), dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        sh = np.asarray(shares) * 100
        bars = ax.bar(range(4), sh, color="#3b6ea5")
        for b, v in zip(bars, sh):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, ("%%%.1f" % v).replace(".", ","), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(4)); ax.set_xticklabels([c.replace(" (", "\n(") for c in CLASSES], fontsize=9)
        ax.set_ylabel(share_label); ax.set_ylim(0, max(sh) * 1.15 if len(sh) else 1)
        fig.tight_layout(); fig.savefig(os.path.join(out, tag + "_israf_siniflari.png"), dpi=200); plt.close(fig)
    except Exception as e:
        print("grafik atlandı:", e)


def _plot_consolidation(cdf, out, tag):
    """Zaman anlarına göre gereken sunucu sayısı: right-sizing öncesi (R) ve sonrası (R')."""
    if cdf.empty:
        return
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        fig, ax = plt.subplots(figsize=(6, 3.4))
        days = cdf["zaman"] / 86400.0
        ax.plot(days, cdf["sunucu_once"], "o-", ms=3, color="#888888", label="Right-sizing öncesi (R)")
        ax.plot(days, cdf["sunucu_sonra"], "s-", ms=3, color="#3b6ea5", label="Right-sizing sonrası (R′)")
        ax.set_xlabel("İz başlangıcından itibaren gün"); ax.set_ylabel("Gereken sunucu sayısı (FFD)")
        ax.yaxis.set_major_formatter(FuncFormatter(_tr_thousands)); ax.set_ylim(bottom=0); ax.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out, tag + "_konsolidasyon.png"), dpi=200); plt.close(fig)
    except Exception as e:
        print("grafik atlandı:", e)


# ----------------------------------------------------------------- sentetik veri + selftest
def make_synth(d, seed=0):
    rng = np.random.default_rng(seed)
    n = 400
    created = rng.integers(0, 86400 * 5, n); life = rng.integers(3600, 86400 * 5, n)
    cores = rng.choice([2, 4, 8, 16, 24], n)
    avg = rng.uniform(1, 50, n); p95 = np.clip(avg + rng.uniform(1, 40, n), 0, 100)
    vm = pd.DataFrame({"vmid": range(n), "sub": 1, "dep": 1, "vmcreated": created, "vmdeleted": created + life,
                       "maxcpu": np.clip(p95 + 5, 0, 100), "avgcpu": avg, "p95maxcpu": p95,
                       "cat": rng.choice(["Interactive", "Delay-insensitive", "Unknown"], n),
                       "cb": cores, "mb": cores * 4})
    vm.to_csv(os.path.join(d, "vmtable.csv"), header=False, index=False)
    ev, us = [], []
    for j in range(0, 60, 10):
        for t in range(3):
            r = rng.uniform(0.05, 0.4); ev.append([0, "", j, t, 1, 0, "u", 1, 1, r, 0.1, 0.01, 0])
            lvl = r * rng.uniform(0.05, 0.9)
            for w in range(40):
                us.append([w * 300, w * 300 + 300, j, t, 1, max(0, lvl * rng.uniform(0.5, 1.2)), 0.1, 0.1, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    pd.DataFrame(ev).to_csv(os.path.join(d, "te_part-00000.csv"), header=False, index=False)
    pd.DataFrame(us).to_csv(os.path.join(d, "tu_part-00000.csv"), header=False, index=False)


def selftest(json_out=None):
    res = []
    def run(no, name, expect, fn):
        try:
            ok = bool(fn()); msg = "Geçti" if ok else "KALDI"
        except Exception as e:
            ok, msg = False, "HATA: %r" % e
        res.append({"no": no, "senaryo": name, "beklenen": expect, "sonuc": msg, "ok": ok})
        print("%-4s %-55s %s" % (no, name, msg))
    rng = np.random.default_rng(1)

    def t1():
        try:
            check_columns(pd.DataFrame({"a": [1]}), ["a", "b"]); return False
        except ValueError:
            check_columns(pd.DataFrame({"a": [1], "b": [2]}), ["a", "b"]); return True
    def t2():
        df = pd.DataFrame({"x": [1, -1, np.nan, 3]}); c, d = clean(df, ["x"]); return len(c) == 2 and d == 2
    def t3():
        return list(classify([0.099, 0.10, 0.299, 0.30, 0.599, 0.60, 1.0])) == [0, 1, 1, 2, 2, 3, 3]
    def t4():
        R = rng.uniform(1, 32, 500); p = rng.uniform(0, 1, 500); b = [1, 2, 4, 8, 16, 32]
        Rp = rightsize(R, p, 0.6, 0.1, b); need = np.minimum(R, R * p * 1.1 / 0.6)
        return bool(np.all(Rp <= R + 1e-9) and np.all(Rp >= need - 1e-9))
    def t5():
        st = np.arange(100) * 300; base = rng.uniform(0, 1, 100); sp = st[70]
        a = {"k": (st, base)}; b = base.copy(); b[70:] += 5; bb = {"k": (st, b)}
        # eğitim penceresi (ilk 70) aynı kalırken test verisi değişirse, öneri aynı kalmalı: ihlal oranı değişir, kapasite azalması değişmez
        s1, v1 = split_eval(a, sp, 0.6, 0.0, {"k": 10.0}); s2, v2 = split_eval(bb, sp, 0.6, 0.0, {"k": 10.0})
        return abs(s1 - s2) < 1e-12 and v2 >= v1
    def t6():
        s = rng.uniform(0.05, 0.9, 200); n, bins = ffd(s, 1.0)
        return all(x <= 1.0 + 1e-9 for x in bins) and n >= int(np.ceil(s.sum() - 1e-9)) and abs(sum(bins) - s.sum()) < 1e-9
    def t7():
        R = np.array([4., 8.]); Rp = np.array([2., 4.]); h = np.array([10., 10.])
        return cost_saving(R, Rp, h, 0.0) == 0.0 and abs(cost_saving(R, Rp, h, 0.05) - 0.05 * 60) < 1e-9
    def e2e():
        with tempfile.TemporaryDirectory() as d:
            make_synth(d)
            a = azure_run(os.path.join(d, "vmtable.csv"), os.path.join(d, "oa"), price=0.04)
            g = google_run(os.path.join(d, "te_part-*.csv"), os.path.join(d, "tu_part-*.csv"), os.path.join(d, "og"), job_mod=10, min_windows=10)
            need = (["azure_siniflar.csv", "azure_ozet.csv", "azure_duyarlilik.csv", "azure_konsolidasyon.csv"],
                    ["google_siniflar.csv", "google_ozet.csv", "google_tasarruf_ihlal.csv", "google_konsolidasyon.csv"])
            files_ok = all(os.path.exists(os.path.join(d, "oa", f)) for f in need[0]) and all(os.path.exists(os.path.join(d, "og", f)) for f in need[1])
            cons_ok = a["konsolidasyon_anlik_sayisi"] > 0 and g["konsolidasyon_anlik_sayisi"] > 0
            return files_ok and cons_ok and a["vm_sayisi"] > 0 and g["gorev_sayisi"] > 0
    def det():
        with tempfile.TemporaryDirectory() as d:
            make_synth(d)
            a = azure_run(os.path.join(d, "vmtable.csv"), os.path.join(d, "o1")); b = azure_run(os.path.join(d, "vmtable.csv"), os.path.join(d, "o2"))
            return a == b
    run("T1", "Şema doğrulama: eksik sütun reddedilir", "Eksik sütunda hata verir, tam şemada kabul eder", t1)
    run("T2", "Eksik ve negatif değerler elenir", "NaN ve negatif satırlar atılır, sayı raporlanır", t2)
    run("T3", "İsraf sınıfı sınır değerleri (%10, %30, %60)", "Alt sınır dahil doğru sınıfa atanır", t3)
    run("T4", "Right-sizing yalnızca küçültür", "R′ ≤ R ve R′ ≥ gereken boyut (yuvarlama dahil)", t4)
    run("T5", "Zaman bölmesinde veri sızıntısı yok", "Test verisi değişince öneri (R′) değişmez", t5)
    run("T6", "Consolidation (FFD) kapasiteyi aşmaz", "Hiçbir sunucu kapasiteyi aşmaz; toplam korunur", t6)
    run("T7", "Maliyet dönüşümü", "Fiyat 0 ise tasarruf 0; fiyatla doğrusal artar", t7)
    run("T8", "Uçtan uca çalıştırma (sentetik Azure + Google)", "Çıktı dosyaları üretilir", e2e)
    run("T9", "Tekrarlanabilirlik", "Aynı girdi iki çalıştırmada aynı çıktıyı verir", det)
    if json_out:
        json.dump(res, open(json_out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    return all(r["ok"] for r in res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("azure"); a.add_argument("--vmtable", required=True); a.add_argument("--out", required=True)
    a.add_argument("--sample", type=int, default=None); a.add_argument("--target-util", type=float, default=0.6)
    a.add_argument("--price-per-vcpu-hour", type=float, default=None)
    a.add_argument("--server-cores", type=float, default=64.0, help="Varsayımsal homojen sunucu çekirdek sayısı (consolidation)")
    g = sp.add_parser("google"); g.add_argument("--events", required=True); g.add_argument("--usage", required=True)
    g.add_argument("--out", required=True); g.add_argument("--job-mod", type=int, default=10); g.add_argument("--min-windows", type=int, default=10)
    g.add_argument("--split", type=float, default=0.7); g.add_argument("--headrooms", type=float, nargs="+", default=[0, .1, .2, .3, .5])
    g.add_argument("--target-util", type=float, default=0.6)
    g.add_argument("--server-capacity", type=float, default=1.0, help="Normalize sunucu kapasitesi (consolidation)")
    for p in (a, g):
        p.add_argument("--max-server-util", type=float, default=0.8, help="Yerleştirmede azami sunucu kullanım oranı")
        p.add_argument("--snapshots", type=int, default=24, help="Consolidation için zaman anı sayısı")
    s = sp.add_parser("selftest"); s.add_argument("--json", default=None)
    A = ap.parse_args()
    if A.cmd == "azure":
        print(azure_run(A.vmtable, A.out, A.sample, A.target_util, A.price_per_vcpu_hour,
                        server_cores=A.server_cores, max_util=A.max_server_util, n_snapshots=A.snapshots))
    elif A.cmd == "google":
        print(google_run(A.events, A.usage, A.out, A.job_mod, A.min_windows, A.split, A.headrooms,
                         u_star=A.target_util, server_capacity=A.server_capacity,
                         max_util=A.max_server_util, n_snapshots=A.snapshots))
    else:
        sys.exit(0 if selftest(A.json) else 1)


if __name__ == "__main__":
    main()
