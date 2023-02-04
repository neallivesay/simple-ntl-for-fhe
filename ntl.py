r"""
A simple Number Theoretic Library for FHE (Python version)

A small library of functions for implementing negacyclic convolution
(aka "polynomial multiplication"), the main operation in (most) Fully
Homomorphic Encryption (FHE) schemes. This library is designed to be
easy to understand and use, and may be used by beginners before
transitioning to high-performance libraries such as HEXL or Shoup's
NTL.

References
----------
TODO

"""

import numpy as np

def is_power_of_two(n: int) -> bool:
    r""" Determines if an integer is a power of two.
    
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

def leng(n: int) -> int:
    r""" Bit-length of an integer.

    Returns the bit-length---i.e., the number of bits in the binary
    representation---of a nonnegative integer.

    Examples
    --------
    >>> leng(7)
    3
    >>> leng(8)
    4
    >>> leng(9)
    4

    """
    return 1+leng(n>>1) if n else 0

def isqrt(k: int) -> int:
    r""" The integer square root function.

    Returns the integer floor of the square root of a positive integer.

    Notes
    -----
    Implementation based on Heron's method.

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
    Slow, brute-force implementation. Note that probabilistic primality
    tests (such as Miller--Rabin) can offer vastly superior 
    performance at the cost of introducing a small likelihood of 
    returning a false positive.

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

    Determines if an integer (greater than two) is "probably" prime;
    returns some false positives (called "Fermat pseudoprimes"), and
    never returns false negatives

    Notes
    -----
    A simple probabilistic primality test based on Fermat's Little
    Theorem. An example Fermat pseudoprime is `341`. A more accurate
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

def generate_ntt_friendly_prime(N: int, length: int, k: int) -> int:
    r""" Returns an NTT friendly prime.

    Returns `k`th NTT friendly prime for `N` with bit-length `length`.
    (If more than `k` NTT friendly primes exist, then `0` is returned.)

    TODO
    OUTPUT: the kth "NTT-friendly" prime for N;
    i.e., the kth prime q with the property that N divides q-1
    """
    if not k>0: raise ValueError('input `k` must be positive integer')
    if not is_power_of_two(N): raise ValueError('input `N` must be ' +
                                        'power of `2` (e.g. `2**16`)')
    if not length>0:
        raise ValueError('input `length` must be positive (e.g. `30`)')
    # TODO
    good_prime = (1<<(length-1))+1
    if k==1 and is_prime(good_prime):
        return good_prime
    while k>0 and leng(good_prime)==length:
        good_prime += N
        while not is_prime(good_prime):
            good_prime += N
        k -= 1
    return good_prime if (k==0 and leng(good_prime)==length) else 0

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

##### functions related to primitive roots of unity modulo q #####

def is_primitive(x: int, N: int, q: int) -> bool:
    r""" A special case primitive root test.

    Tests whether an integer `x` is a primitive `N`th root of unity 
    modulo a prime `q`. Assumes that `q` is an "NTT friendly prime"
    with respect to `N`, i.e., that `N` is a power of two integer 
    dividing `q-1`.

    Notes
    -----
    TODO - is N dividing q-1 necessary?

    Examples
    --------
    >>> is_primitive(9, 8, 17)
    True 
    >>> is_primitive(13, 8, 17)
    False

    """
    return pow(x, N//2, q) == q-1

def gen_primitive_root(n: int, q: int) -> int:
    r""" INPUT: q prime, n a power of 2, with n dividing q-1
    OUTPUT: returns a primitive nth root of unity modulo q
    REFERENCE: https://crypto.stackexchange.com/questions/63614
    >>> gen_primitive_root(8, 17)==9 and gen_primitive_root(16, 97)==8
    True
    """
    assert is_prime(q), "q must be prime"
    assert is_power_of_two(n), "n must be power of 2"
    assert (q-1) % n == 0, "n must divide q-1"
    def make_root(i: int) -> int:
        ''' NOTE: i**((q-1)/n) is always a root modulo q '''
        return pow_mod(i, (q-1)//n, q)

    for p in range(2,q):
        if pow_mod(make_root(p), n//2, q) != 1:
            return make_root(p)
    assert False, "primitive root not found. there exists a critical error in implementation"

##### functions related to bit-reversal #####

def bit_reverse(x: int, bit_length: int):
    r""" Reverses the order of bits in binary representation.

    Assume `x` is an integer with length at most `bit_length`. Bit-
    reversal maps `x` to the integer whose binary representation has 
    the same bits, but in reverse order. For example, a 3-bit bit-
    reversal maps the integer 1 (i.e., 001 in binary) to 4 (100), 
    2 (010) to itself, and 3 (011) to 6 (110).

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

def bit_reverse_permute(x: np.array):
    '''in-place bit reverse permutation
    >>> bit_reverse_permute([0, 1, 2, 3, 4, 5, 6, 7])
    [0, 4, 2, 6, 1, 5, 3, 7]
    '''
    assert is_power_of_two(len(x)), "x must have length a power of 2"
    for i in range(len(x)):
        bit_reversed_index = bit_reverse(i, leng(len(x))-1)
        if i < bit_reversed_index:
            x[i], x[bit_reversed_index] = x[bit_reversed_index], x[i]
    return x

# if this file is run directly from command line, then doctests are run
if __name__ == "__main__":
    import doctest
    doctest.testmod()

