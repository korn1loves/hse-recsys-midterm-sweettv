# Environment

- recorded: 2026-10-24 13:52 (local, MSK)
- machine: Apple M4, 10 cores, 16 GB RAM
- OS: macOS 26.6.2 (macOS-26.6.2-arm64-arm-64bit-Mach-O)
- Python 3.13.5 in .venv

| package | version |
|---|---|
| pandas | 3.0.6 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| sklearn | 1.9.1 |
| matplotlib | 3.11.2 |
| pyarrow | 26.0.0 |
| tabulate | 0.10.0 |
| pytest | 9.1.1 |
| jupyter_core | 5.9.1 |
| jupyterlab | 4.6.4 |

Full list: `results/requirements-freeze.txt`.

Seeds: Phase A methods are deterministic (ties broken by tv_show_id asc); the only RNG is `np.random.default_rng(0)` used to pick the hand-checkable join example.
