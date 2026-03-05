# GEM Baseline Lock

This folder stores deterministic baseline artifacts for migration validation.

## Fixture
- Name: `tiny_v1`
- Netlist: `baseline/tiny_gatelevel.gv`
- Partition file: `baseline/tiny.gemparts`
- Locked script hash: `14926125099726623616`

## Verify
Run from GEM root:

```sh
cargo run --bin baseline_lock -- \
  baseline/tiny_gatelevel.gv \
  baseline/tiny.gemparts \
  1 \
  --expected-script-hash 14926125099726623616
```

The command exits non-zero if the flatten script hash changes.

## Regenerate fixture artifacts
```sh
cargo run --bin cut_map_interactive -- \
  baseline/tiny_gatelevel.gv \
  baseline/tiny.gemparts
```
