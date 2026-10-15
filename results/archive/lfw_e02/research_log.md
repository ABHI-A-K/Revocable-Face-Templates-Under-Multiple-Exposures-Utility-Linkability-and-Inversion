# LFW-E02-seeded-RO research log

**Researcher:** Abhishek A K  
**Institution:** IIIT Dharwad / MIT Manipal  
**Run started (UTC):** 2026-08-11T06:25:30.702584+00:00

## Purpose

Evaluate whether a seeded random-orthogonal 128-bit BioHash preserves LFW verification performance relative to the frozen 512-D E01 baseline.

## Hypothesis

The protected representation will retain strong recognition performance while reducing stored template size from 2,048 bytes to 16 bytes.

## Notes

- Controlled change: raw 512-D cosine matching -> 128-bit Hamming matching.
- Five fixed experimental seeds are evaluated independently.
- One common transform is used for all samples within a seed trial.
- Cross-key unlinkability and inversion resistance are outside E02 recognition scope.

## Completion

Run completed (UTC): 2026-08-11T06:33:07.036448+00:00.  
See `RESULTS_README.md` for the automatic result summary and `experiment_record.json` for the machine-readable record.
