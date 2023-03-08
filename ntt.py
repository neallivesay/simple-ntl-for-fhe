#!/usr/bin/env python3
r"""
Basic implementations of the Number Theoretic Transform (NTT)

TODO:
- format docstrings
- include implementations of both classical and merged NTTs
- create test modules

"""

# Number Theoretic Transform and Poly Mult (aka negacyclic convolution)
def ntt(a: np.array, omega_n: int, n: int, q: int) -> np.array:
    r""" NTT implemented via "decimation-in-time" (DIT) FFT
    INPUT: coefficient vector a = [a0, a1,...,a(n-1)]
    n a power of 2, q a prime with n dividing q-1
    omega_n a primitive nth root of unity mod q
    REFERENCE: ITERATIVE-FFT on page 917 in CLRS
    >>> ntt([1,2,3,7,5,4,1,2], 9, 8, 17)
    [8, 11, 14, 2, 12, 16, 7, 6]
    """
    # BIT-REVERSE-COPY(a, A) (see CLRS, page 917)
    bit_reverse_permute(a)
    # apply Cooley--Tukey butterflies
    lgn = leng(n)-1
    m = 1
    for i in range(lgn):
        m *= 2
        omega_m = pow_mod(omega_n, 1<<(lgn-i-1), q)
        k = 0
        while k<n:
            omega = 1
            for j in range(m//2):
                t = omega*a[k+j+m//2] % q
                u = a[k+j]
                a[k+j] = (u+t) % q
                a[k+j+m//2] = (u-t) % q
                omega = (omega * omega_m) % q
            k += m
    return a

def intt(a: np.array, omega_n: int, n: int, q: int) -> np.array:
    # scale 'a' by 1/N mod q
    # double-check 'a' is an np.array and not a list; otherwise scaling causes problems!
    a = modular_inverse_prime(n, q) * np.array(a)
    return ntt(a, modular_inverse_prime(omega_n, q), n, q)

def poly_mult(a: np.array, b: np.array, n: int, q: int) -> np.array:
    ''' INPUT: two coefficient vectors 'a' and 'b'
    n power of 2, q prime with (2n) dividing q-1
    OUTPUT: negacyclic convolution of 'a' and 'b'
    '''
    psi = gen_primitive_root(2*n, q) # prim (2n)th root of unity
    ipsi = modular_inverse_prime(psi, q) # inverse of psi modulo q
    omega = (psi*psi)%q

    # generate PowMul_psi and PowMul_ipsi
    pow_mul_psi = [1]*n
    pow_mul_ipsi = [1]*n
    for i in range(n-1):
        pow_mul_psi[i+1] = (pow_mul_psi[i] * psi) % q
        pow_mul_ipsi[i+1] = (pow_mul_ipsi[i] * ipsi) % q
    pow_mul_psi = np.array(pow_mul_psi)
    pow_mul_ipsi = np.array(pow_mul_ipsi)

    # modular hadamard products with PowMul_psi
    A = (a*pow_mul_psi) % q
    B = (b*pow_mul_psi) % q
    # NTT's
    nttA = ntt(A, omega, n, q)
    nttA = ntt(B, omega, n, q)
    # modular hadamard product
    C = (A*B) % q
    # inverse NTT
    inttC = intt(C, omega, n, q)
    # modular hadamard products with PowMul_ipsi
    product = (pow_mul_ipsi*inttC) % q
    return product

