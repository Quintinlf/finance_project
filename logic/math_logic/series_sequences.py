"""
Sequences and Series Calculator
Implements formulas for arithmetic and geometric sequences/series
"""

import math

class ArithmeticSequence:
    """Arithmetic sequence: a, a+d, a+2d, a+3d, ..."""
    
    def __init__(self, a, d):
        """
        Initialize arithmetic sequence
        a: first term
        d: common difference
        """
        self.a = a
        self.d = d
    
    def nth_term(self, n):
        """Calculate nth term: a_n = a + (n-1)d"""
        return self.a + (n - 1) * self.d
    
    def sum_n_terms(self, n):
        """Calculate sum of first n terms: S_n = n/2 * (2a + (n-1)d)"""
        return n / 2 * (2 * self.a + (n - 1) * self.d)
    
    def sum_n_terms_alt(self, n):
        """Alternative sum formula: S_n = n/2 * (a + l) where l is last term"""
        l = self.nth_term(n)
        return n / 2 * (self.a + l)


class GeometricSequence:
    """Geometric sequence: a, ar, ar², ar³, ..."""
    
    def __init__(self, a, r):
        """
        Initialize geometric sequence
        a: first term
        r: common ratio
        """
        self.a = a
        self.r = r
    
    def nth_term(self, n):
        """Calculate nth term: a_n = ar^(n-1)"""
        return self.a * (self.r ** (n - 1))
    
    def sum_n_terms(self, n):
        """
        Calculate sum of first n terms
        S_n = a(1 - r^n) / (1 - r) for r ≠ 1
        S_n = na for r = 1
        """
        if self.r == 1:
            return n * self.a
        return self.a * (1 - self.r ** n) / (1 - self.r)
    
    def sum_n_terms_alt(self, n):
        """
        Alternative formula: S_n = a(r^n - 1) / (r - 1) for r ≠ 1
        """
        if self.r == 1:
            return n * self.a
        return self.a * (self.r ** n - 1) / (self.r - 1)
    
    def sum_to_infinity(self):
        """
        Sum to infinity: S_∞ = a / (1 - r) for |r| < 1
        Returns None if |r| >= 1 (series diverges)
        """
        if abs(self.r) >= 1:
            return None
        return self.a / (1 - self.r)


def demonstrate():
    """Demonstrate usage of the classes"""
    print("=" * 50)
    print("ARITHMETIC SEQUENCE EXAMPLE")
    print("=" * 50)
    
    # Example: 2, 5, 8, 11, ... (a=2, d=3)
    arith = ArithmeticSequence(a=2, d=3)
    print(f"First term (a): {arith.a}")
    print(f"Common difference (d): {arith.d}")
    print(f"\n5th term: {arith.nth_term(5)}")
    print(f"10th term: {arith.nth_term(10)}")
    print(f"\nSum of first 10 terms: {arith.sum_n_terms(10)}")
    print(f"Sum (alternative formula): {arith.sum_n_terms_alt(10)}")
    
    print("\n" + "=" * 50)
    print("GEOMETRIC SEQUENCE EXAMPLE")
    print("=" * 50)
    
    # Example: 3, 6, 12, 24, ... (a=3, r=2)
    geom = GeometricSequence(a=3, r=2)
    print(f"First term (a): {geom.a}")
    print(f"Common ratio (r): {geom.r}")
    print(f"\n5th term: {geom.nth_term(5)}")
    print(f"8th term: {geom.nth_term(8)}")
    print(f"\nSum of first 10 terms: {geom.sum_n_terms(10)}")
    print(f"Sum (alternative formula): {geom.sum_n_terms_alt(10)}")
    print(f"\nSum to infinity: {geom.sum_to_infinity()}")
    
    # Example with convergent series: 1, 1/2, 1/4, 1/8, ... (a=1, r=1/2)
    print("\n" + "=" * 50)
    print("CONVERGENT GEOMETRIC SERIES")
    print("=" * 50)
    geom2 = GeometricSequence(a=1, r=0.5)
    print(f"First term (a): {geom2.a}")
    print(f"Common ratio (r): {geom2.r}")
    print(f"\nSum of first 10 terms: {geom2.sum_n_terms(10)}")
    print(f"Sum to infinity: {geom2.sum_to_infinity()}")


if __name__ == "__main__":
    demonstrate()