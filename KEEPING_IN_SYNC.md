# Keeping Python and Rust in sync

DSPkit has a native Rust port, `../dspkit-rs` (it also powers the single-exe
app). **A change here is not done until `dspkit-rs` has it too.**

- New option, fix or function in `dspkit/<module>.py` -> the same in
  `../dspkit-rs/src/<module>.rs` (and a CLI op in `src/main.rs` if new).
- Add a case to `../dspkit-rs/tools/parity.py`, then from `dspkit-rs/`:
  `cargo build --release` and `py tools/parity.py` must pass.
- User-facing text (errors, interpretations, advice) must be identical on
  both sides: the parity harness compares it exactly.
- If the two disagree, decide which is right, fix it with a test here, then
  port. Don't loosen a tolerance to hide a difference.

Full guide, with what-to-change-where and every check:
`../dspkit-rs/KEEPING_IN_SYNC.md`.
