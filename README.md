# Why do we regularise in every iteration

### Installation

```
conda create --name why_skip -c conda-forge python=3.12 cmake scipy six cython numba pillow jupyterlab scikit-learn dask cvxpy zarr pywavelets astra-toolbox tqdm nb_conda_kernels

conda activate why_skip
git clone https://github.com/epapoutsellis/StochasticCIL.git
cd StochasticCIL
git checkout svrg
git tag -a v1.0 -m "Version 1.0"
mkdir build
cmake ../ -DCONDA_BUILD=OFF -DCMAKE_BUILD_TYPE="Release" -DLIBRARY_LIB=$CONDA_PREFIX/lib -DLIBRARY_INC=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make install
```

For windows: `cmake ../ -DCONDA_BUILD=OFF`, `cmake --build . --target install`


## Acknowledgements:



