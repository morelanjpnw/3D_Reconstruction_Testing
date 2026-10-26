"""Windows-only fix: gsplat passes the GCC flag -Wno-attributes to MSVC's cl.exe, which fails the JIT build.
Idempotent. Run with the project env's python (setup_windows.ps1 does this)."""
import gsplat
from pathlib import Path

p = Path(gsplat.__file__).parent / "cuda" / "_backend.py"
s = p.read_text()
old = '        extra_cflags = [opt_level, "-Wno-attributes"]'
new = ('        # local Windows patch: MSVC rejects the GCC flag -Wno-attributes\n'
       '        extra_cflags = [opt_level] if os.name == "nt" else [opt_level, "-Wno-attributes"]')
if new in s:
    print("gsplat already patched")
elif old in s:
    p.write_text(s.replace(old, new))
    print("gsplat patched:", p)
else:
    print("WARNING: expected line not found in", p, "- gsplat version may differ; check the build if it fails")
