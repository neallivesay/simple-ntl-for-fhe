# simple Number Theoretic Library for FHE

A small library to support the implementation of high-performance Number Theoretic Transforms (i.e., the Fast Fourier Transforms for a field of prime size), especially in the context of Fully Homomorphic Encryption (FHE).

This library is designed to be easy to understand, easy to use, small, and have minimal dependencies. Simplicity has been favored over both performance and generality for functions that do not have a direct impact on the performance of the Number Theoretic Transform. Such functions include primality tests, "NTT friendly" prime generation, and primitive root tests and generation. For more general high-performance number theoretic libraries, see [Shoup's NTL](https://github.com/libntl/ntl) or [Intel's HEXL library](https://github.com/intel/hexl).
