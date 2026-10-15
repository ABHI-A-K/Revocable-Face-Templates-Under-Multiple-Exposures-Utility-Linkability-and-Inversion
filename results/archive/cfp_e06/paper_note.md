# CFP-FP-E06-PROJ — research paper note

This is the LFW-E06 projection comparison extended to the screened CFP subset. The template-only attacker does not receive raw embeddings, keys or projection operators. The 128-bit PCA branch matches the LFW construction; the 512-bit branch retains all principal components and does not compress. Different datasets, partitions and seeds prevent direct pooled estimates.

512-bit results: PCA_GUIDED_RO: TAR 82.38%, FAR 0.2002%, K10 AUC 0.5643; RO: TAR 82.52%, FAR 0.1715%, K10 AUC 0.5667; SRHT: TAR 81.95%, FAR 0.1670%, K10 AUC 0.5629

Report the subset mapping limitation and observed FAR alongside TAR. Interpret differences as exploratory; failure of one linkage attacker is not proof against known-key recovery. E07 is reserved for reconstruction and fresh-key impersonation.
