# Environment provenance

Root requirements.txt pins the analysis environment used for packaging, numerical checks and figure generation. It is not a retrospective experiment lock. Runtime JSON files preserve observed experiment environments.

CFP-ENC-E01A records Python 3.13.15, Tesla T4, torch 2.11.0+cu128, CUDA build 12.8 and ONNX Runtime GPU 1.26.0. The LFW extension used CPU inference, ONNX Runtime 1.28.0 and OpenCV 4.13.0.92 with explicit version/hash checks. Follow each run's records rather than treating these as interchangeable.

Inspect notebook installation cells before using an existing environment. Record pip freeze, CPU/GPU, driver, CUDA and BLAS details for fresh full runs. Hardware provenance is not a performance benchmark.
