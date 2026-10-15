# Frozen checkpoints

Use the official InsightFace model-zoo documentation: https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md . Historical official downloads recorded in the experiments are:

- `w600k_mbf.onnx`: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
- `w600k_r50.onnx`: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
- SCRFD `det_500m.onnx`: buffalo_s/buffalo_sc detector; exact recorded hash in `lfw_input_hashes.json`.

`model_locks.json` contains encoder hashes, byte sizes and source records. Verify the extracted ONNX bytes using `python scripts/hash_file.py PATH`, not the name alone. Do not substitute newer similarly named weights. Weights are excluded; obtain them under the publisher's terms. For LFW extraction use `experiments/models/`. CFP notebooks document their own paths.
