#!/bin/sh

export CFLAGS="-I$(python - <<EOF
import numpy as np; print(np.get_include())
EOF
)"; python -m Cython.Build.Cythonize -a -i sparseoperations.pyx
