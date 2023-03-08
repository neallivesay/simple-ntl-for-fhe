#!/usr/bin/env python3
r"""
A simple Number Theoretic Library for FHE (Python version)

A library to support the implementation of high-performance Number
Theoretic Transforms (NTTs)---i.e., the Fast Fourier Transforms for a
field of prime size---especially in the context of Fully Homomorphic
Encryption (FHE). This library is designed to be small, easy to
understand, easy to use, and have minimal dependencies. Simplicity has
been prioritized over both performance and generality for functions
that do not have a direct impact on the performance of the Number 
Theoretic Transform. Such functions include primality tests, "NTT
friendly" prime generation, and primitive root tests and generation.
All algorithms used are widely known. For much more comprehensive
high-performance number theoretic libraries, see Shoup's NTL or Intel's
HEXL library.

"""

##### functions related to binary representations  #####

def is_power_of_two(n: int) -> bool:
    r""" Tests if integer is a power of two.

    Determines if `n` is a power of two. Note `n` is a power of two if
    and only if its Hamming weight equals one.
    
    Examples
    --------
    >>> is_power_of_two(7)
    False
    >>> is_power_of_two(8)
    True
    >>> is_power_of_two(9)
    False
    
    """
    return not n & (n-1)

def bit_reverse(x: int, bit_length: int):
    r""" Reverses the order of bits in binary representation.

    Assume `x` is an integer with length at most `bit_length`. Bit-
    reversal maps `x` to the integer whose binary representation has 
    the same bits, but in reverse order. For example, a 3-bit bit-
    reversal maps the integer 1 (i.e., 001 in binary) to 4 (100), 
    2 (010) to 2 (010), and 3 (011) to 6 (110).

    Examples
    --------
    >>> bit_reverse(1, 3)
    4
    >>> bit_reverse(3, 3)
    6
    >>> bit_reverse(5, 4)
    10

    """
    result = 0
    while bit_length > 0:
        result <<= 1
        result = (x & 1) | result
        x >>= 1
        bit_length -= 1
    return result

def bit_reverse_permute(x: list) -> list:
    r""" Bit-reversal permutation.

    Assumes the length of the list `x` is a power, say `k`, of two.
    Permutes the entries `x` by mapping `x[i]` to `x[br(i)]`, where 
    `br` denotes the `k`-bit bit-reversal. Modifies `x` in-place.

    Examples
    --------
    >>> bit_reverse_permute([0, 1, 2, 3])
    [0, 2, 1, 3]
    >>> bit_reverse_permute([0, 1, 2, 3, 4, 5, 6, 7])
    [0, 4, 2, 6, 1, 5, 3, 7]

    """
    if not is_power_of_two(len(x)):
        raise ValueError('x must be a list of length a power of two')
    k = len(x).bit_length() - 1
    for i in range(len(x)):
        br_i = bit_reverse(i, k)
        if i < br_i:
            x[i], x[br_i] = x[br_i], x[i]
    return x

##### functions related to primality testing and generation #####

def isqrt(k: int) -> int:
    r""" The integer square root function.

    Returns the integer floor of the square root of a positive integer.

    Notes
    -----
    Heron's method.

    Examples
    --------
    >>> isqrt(8)
    2
    >>> isqrt(9)
    3
    >>> isqrt(10)
    3
    
    """
    x = k
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + k//x) // 2
    return x

def is_prime(x: int) -> bool:
    r""" Primality test.

    Determines if an integer (greater than two) is prime.

    Notes
    -----
    Generates primes with bit-length `leng` that are "NTT friendly" for
    Brute-force implementation. Note that probabilistic primality tests
    (such as Miller--Rabin) can offer vastly superior runtime but
    introduces a small likelihood of returning a false positive.

    Examples
    --------
    >>> is_prime(6)
    False
    >>> is_prime(7)
    True
    >>> is_prime(8)
    False
    >>> is_prime(341)
    False

    """
    return x%2!=0 and \
            all(x%divisor!=0 for divisor in range(3, isqrt(x)+1, 2))

def is_probably_prime(x: int) -> bool:
    r""" Probabilistic primality test.

    Determines if an integer (greater than two) is "probably" prime.
    May return a false positive, but never returns false negatives.

    Notes
    -----
    A simple probabilistic primality test based on Fermat's Little
    Theorem. An example false positive is `341`. A more accurate
    probabilistic primality test is Miller--Rabin.

    Examples
    --------
    >>> is_prime(6)
    False
    >>> is_prime(7)
    True
    >>> is_prime(8)
    False

    """
    return pow(2, x-1, x) == 1

def generate_ntt_friendly_prime(N: int, leng: int, start=1) -> int:
    r""" Generates NTT friendly primes of a given bit-length.

    Generates primes with bit-length `leng` that are "NTT friendly" for
    a power of two `N`, i.e., primes `q` with the property that `N`
    divides `q-1`. Yields all such primes in ascending order, starting
    with the `start`th such prime.

    Examples
    --------
    >>> g = generate_ntt_friendly_prime(2**16, 30)
    >>> next(g)
    537133057
    >>> next(g)
    537591809
    
    """
    if not start>0:
        raise ValueError('input `start` must be positive integer')
    if not is_power_of_two(N):
        raise ValueError('input `N` must be power of two')
    if not leng>0:
        raise ValueError('input `leng` must be positive')

    # start with the smallest `leng`-bit candidate for an NTT prime
    ntt_prime = (1<<(leng-1))+1 
    while start > 0:
        while not is_prime(ntt_prime):
            ntt_prime += N
        start -= 1
    while ntt_prime.bit_length() == leng:
        yield ntt_prime
        ntt_prime += N
        while ntt_prime.bit_length()==leng and not is_prime(ntt_prime):
            ntt_prime += N

##### modular arithmetic functions #####

def modular_inverse_prime(x: int, q: int) -> int:
    r""" Modular multiplicative inverse, specialized for prime modulus.

    Returns multiplicative inverse of `x` modulo a prime integer `q`.

    Notes
    -----
    Simple implementation based on Fermat's Little Theorem. Modular 
    inversion for general, not-necessarily-prime moduli can be
    implemented via the Extended Euclidean Algorithm.

    Examples
    --------
    >>> modular_inverse_prime(1,7)
    1
    >>> modular_inverse_prime(2,7)
    4
    >>> modular_inverse_prime(3,17)
    6

    """
    return pow(x, q-2, q)

##### functions related to roots of unity #####

def is_primitive(x: int, N: int, q: int) -> bool:
    r""" A (special case) primitive root test.

    Tests if an integer `x` is a primitive `N`th root of unity modulo a
    prime `q`. Assumes `N` is a power of two integer dividing `q-1` (a
    standard baseline assumption when implementating of the NTT).

    Examples
    --------
    >>> is_primitive(9, 8, 17)
    True
    >>> is_primitive(13, 8, 17)
    False

    """
    return pow(x, N//2, q) == q-1

def primitive_root(N: int, q: int) -> int:
    r""" Returns a primitive root of unity.

    Returns a primitive `N`th root of unity modulo a prime `q`. Assumes
    `N` is a power of two dividing `q-1` (a standard baseline 
    assumption when implementing the NTT).
    
    Notes
    -----
    Given the assumptions on `N` and `q` above, there are exactly `N`
    `N`th roots of unity modulo `q`, with exactly half being primitive.
    A brute-force approach to finding one of these primitive roots
    might involve searching through all integers from `1` to `q-1` and
    testing if it is a primitive root. As the likelihood of any given
    integer being a root is `(q-1)/N`, this search can be slow when `q`
    is large relative to `N`. The implementation below uses an 
    alternative approach, leveraging the fact that if `x` is nonzero,
    then `x` raised to the power of `(q-1)/N` is always a root. The 
    likelihood of a root being primitive (given the above assumptions)
    is `1/2`.

    Examples
    --------
    >>> is_primitive(primitive_root(16, 97), 16, 97)
    True

    """
    if not is_prime(q): raise ValueError('q must be prime')
    if not is_power_of_two(N): raise ValueError('N must be power of 2')
    if (q-1) % N != 0: raise ValueError('N must divide q-1')

    def make_root(x: int) -> int:
        return pow(x, (q-1)//N, q)

    for x in range(2, q):
        if is_primitive(make_root(x), N, q):
            return make_root(x)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
