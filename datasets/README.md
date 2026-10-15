# Dataset inputs

LFW original reference: http://vis-www.cs.umass.edu/lfw/ . Use the deep-funneled variant and original 6000 official pairs. Current Torchvision documentation describes manual acquisition: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.LFWPairs.html ; its listed mirror is https://www.kaggle.com/datasets/jessicali9530/lfw-dataset . Do not substitute another alignment variant.

CFP project: https://www.cfpw.io/ . Its endpoint timed out during documentation checking on 14 October 2026; current availability is not asserted. Obtain authorized CFP images and FP folds. Official recognition uses the exact aligned cfp_fp.bin from the supplied source audit. Custom attacks use the frozen 5704-crop screened subset, retaining 186 exclusions. Do not infer identities from embedding similarity.

LFW extension expects `experiments/datasets/lfw-dataset.zip`; hashes are in `models/lfw_input_hashes.json`. Historical Kaggle notebooks discover `/kaggle/input` archives and write `/kaggle/working` outputs. Use their stated predecessor inputs. Dataset terms govern redistribution. This package includes no photographs; CSV identity/path labels remain to preserve exact protocol definitions.
