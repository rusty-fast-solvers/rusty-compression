# rusty-compression
A low-rank compression library in Rust.

This library provides various routines for the low-rank compression of
linear operators. The algorithms are mostly adapted from the book
*Fast direct solvers for elliptic PDEs* by Gunnar Martinsson.

For examples see the two example programs in the directory `examples`.
In application use the library requires the `ndarray-linalg` dependency
to be linked against `Openblas` or another Lapack library. For details
see the corresponding documentation of [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg).

