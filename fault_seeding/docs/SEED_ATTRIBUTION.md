# Seed Program Attribution and Licensing

Deep4ge includes 59 corrected seed programs adapted from StackOverflow
questions. These seed programs are not original Deep4ge-authored source code.
They are redistributed as adapted StackOverflow content for reproducibility.

Per-seed attribution is recorded in:

- `data/seed_programs/ATTRIBUTION.csv`

That table records the StackOverflow question URL, title, author display name,
author profile URL when available, creation date, content license reported by
the Stack Exchange API, local seed-program path, and an adaptation note.

## License Boundary

- Deep4ge framework code under `src/`, `scripts/`, and `analysis/` is licensed
  under MIT.
- Generated Deep4ge data products, including `data/manifest.csv`,
  `data/training_logs/`, and metadata authored for this dataset, are licensed
  under CC BY 4.0.
- Seed programs under `data/seed_programs/{fnn,cnn,rnn}/` are adapted from
  StackOverflow posts and remain subject to the source post licenses reported in
  `data/seed_programs/ATTRIBUTION.csv` (for example, CC BY-SA 3.0 or CC BY-SA
  4.0).

Two seed IDs (`51981187`, `52782432`) did not resolve through the Stack Exchange
API during the May 5, 2026 attribution refresh. Their source URLs are retained,
but their license metadata is marked as unknown and should be manually verified
before redistribution decisions that depend on exact source licensing.

## Attribution Format

When citing or reusing seed programs, attribute both Deep4ge and the originating
StackOverflow post. A suitable format is:

`Adapted from StackOverflow question <so_id>, "<source_title>", by
<source_author_display_name>, <source_url>, licensed as <source_content_license>;
adapted in Deep4ge.`
