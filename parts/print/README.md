# Test parts for the real printer

Three small parts (about 7-8 g of PLA for all nine test prints together) where a hand-made dent
mid-print either only changes the **form** or breaks the **function**. Each folder has the STL to
print, a `diagram.png`, print settings and test steps (`README.md`), and the function spec our
pipeline checks (`.yaml`).

| Part | Function (physical test) | Cosmetic dent | Breaking dent | Hardware |
|---|---|---|---|---|
| [1. two-hole block](two-hole-block/) | 2 M5 bolts pass through, 30 mm apart | middle of the block | next to a hole | 2 × M5 bolt |
| [2. water tray](water-tray/) | holds water | gentle flat push | poke through | water |
| [3. knob](star-knob/) | M5 nut fits, bolt turns | tip of a lobe | bottom of a valley | M5 nut + bolt |

Each part: 3 runs (clean, cosmetic dent, breaking dent), dent at half height, `watch` running.
How `watch` works: [two-hole-block/README.md](two-hole-block/README.md). Print part 1 first.

These folders are kept out of `parts/*.yaml` on purpose, so the simulation benchmarks' default
part list does not change. Load a spec with `--part parts/print/<name>/<name>.yaml`.
