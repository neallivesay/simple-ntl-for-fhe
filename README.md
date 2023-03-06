# simple Number Theoretic Library for FHE

A library to support the implementation of high-performance Number Theoretic Transforms (i.e., the Fast Fourier Transforms for a field of prime size), particularly in the context of Fully Homomorphic Encryption (FHE).

This library is designed to be small, easy to understand, easy to use, and have minimal dependencies. Simplicity has been prioritized over both performance and generality for all functions that do not have a direct impact on the performance of the Number Theoretic Transform. Such functions include primality tests, "NTT friendly" prime generation, and primitive root tests and generation.

For much more comprehensive high-performance number theoretic libraries, see [Shoup's NTL](https://github.com/libntl/ntl) or [Intel's HEXL library](https://github.com/intel/hexl).

## Contents

`ntl.py`: number theoretic library in Python
`ntt.py`: Number Theoretic Transform implementations in Python (in-progress)