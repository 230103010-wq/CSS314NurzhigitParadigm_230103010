import argparse, json, os, platform, re, subprocess, sys
from pathlib import Path
from salt import get_student_params

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def parse_size(s):
    m = re.match(r"(\d+)\s*([KMG]?)", s.strip().upper())
    n, u = int(m.group(1)), m.group(2)
    return n * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[u]


def linux_info():
    info = {"cpu_model": None, "base_freq_mhz": None}
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name") and not info["cpu_model"]:
            info["cpu_model"] = line.split(":", 1)[1].strip()
        if line.startswith("cpu MHz") and not info["base_freq_mhz"]:
            info["base_freq_mhz"] = float(line.split(":", 1)[1])
    caches = {}
    for d in sorted(Path("/sys/devices/system/cpu/cpu0/cache").glob("index*")):
        level = (d / "level").read_text().strip()
        typ = (d / "type").read_text().strip()
        size = parse_size((d / "size").read_text())
        caches["line_size_bytes"] = int((d / "coherency_line_size").read_text())
        if level == "1" and typ == "Data":
            caches["l1d_bytes"] = size
        elif level == "2":
            caches["l2_bytes"] = size
        elif level == "3":
            caches["l3_bytes"] = size
    info["caches"] = caches
    return info


def mac_info():
    g = lambda k: run(["sysctl", "-n", k])
    caches = {"line_size_bytes": int(g("hw.cachelinesize") or 64)}
    for key, name in [("hw.l1dcachesize", "l1d_bytes"), ("hw.l2cachesize", "l2_bytes"), ("hw.l3cachesize", "l3_bytes")]:
        v = g(key)
        caches[name] = int(v) if v else None
    freq = g("hw.cpufrequency")
    return {"cpu_model": g("machdep.cpu.brand_string"),
            "base_freq_mhz": int(freq) / 1e6 if freq else None, "caches": caches}


def windows_info():
    ps = ("Get-CimInstance Win32_Processor | Select-Object Name,MaxClockSpeed,L2CacheSize,L3CacheSize"
          " | ConvertTo-Json")
    out = run(["powershell", "-NoProfile", "-Command", ps])
    d = json.loads(out) if out else {}
    if isinstance(d, list):
        d = d[0]
    caches = {"line_size_bytes": 64,
              "l1d_bytes": None,  # Windows не отдаёт L1: впишите из спецификации CPU
              "l2_bytes": d.get("L2CacheSize", 0) * 1024 or None,
              "l3_bytes": d.get("L3CacheSize", 0) * 1024 or None}
    return {"cpu_model": d.get("Name"), "base_freq_mhz": d.get("MaxClockSpeed"), "caches": caches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--microarch", default=None, help="например 'Zen 3' или 'Alder Lake'")
    ap.add_argument("--l1d-kb", type=int, default=None, help="если L1d не определился автоматически")
    a = ap.parse_args()

    sysname = platform.system()
    info = {"Linux": linux_info, "Darwin": mac_info, "Windows": windows_info}[sysname]()
    if a.l1d_kb:
        info["caches"]["l1d_bytes"] = a.l1d_kb * 1024

    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
        logi = psutil.cpu_count(logical=True)
    except ImportError:
        phys, logi = None, os.cpu_count()

    profile = {
        "cache_line_bytes": info["caches"].get("line_size_bytes") or 64,
        "cpu_model": info["cpu_model"],
        "microarchitecture": a.microarch,
        "base_freq_mhz": info["base_freq_mhz"],
        "physical_cores": phys,
        "logical_threads": logi,
        "smt_enabled": bool(phys and logi and logi > phys),
        "caches": info["caches"],
        "os": sysname,
        "kernel": platform.release(),
        "os_version": platform.version(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
    RES.mkdir(exist_ok=True)
    (RES / "host_profile.json").write_text(json.dumps(profile, indent=2))
    (RES / "salted_params.json").write_text(json.dumps(get_student_params(a.student_id), indent=2))
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()