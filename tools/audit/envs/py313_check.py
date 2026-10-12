"""Python 3.13 compatibility probe: byte-compile every .py under poker/ and tools/ and import every poker.* module
in a fresh subprocess (catches PEP 649 deferred-annotation reliance, 3.14-only stdlib, syntax). Usage: py313_check.py OUT.json"""
import json, os, pathlib, subprocess, sys, py_compile
root = pathlib.Path(".")
res = {"python": sys.version.split()[0], "compile_errors": [], "imports": {}}
for p in sorted(list(root.glob("poker/**/*.py")) + list(root.glob("tools/**/*.py"))):
    if "audit" in p.parts: continue
    try:
        compile(p.read_text(), str(p), "exec")
    except py_compile.PyCompileError as e:
        res["compile_errors"].append(f"{p}: {e.msg[:200]}")
mods = sorted({".".join(p.with_suffix("").parts).replace(".__init__", "") for p in root.glob("poker/**/*.py")})
env = dict(os.environ)
for m in mods:
    r = subprocess.run([sys.executable, "-c", f"import {m}"], capture_output=True, text=True, timeout=300, env=env)
    res["imports"][m] = "ok" if r.returncode == 0 else r.stderr.strip().splitlines()[-1][:300]
res["import_failures"] = {k: v for k, v in res["imports"].items() if v != "ok"}
print(json.dumps({k: v for k, v in res.items() if k != "imports"}, indent=1)); print(len(mods), "modules")
json.dump(res, open(sys.argv[1], "w"), indent=1)
