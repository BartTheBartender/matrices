pub mod cyclic;
pub mod integers;

use std::{
    hash::Hash,
    iter::Sum,
    num::NonZero,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

pub use integers::Integer;
//pub use cyclic::Natural;

pub trait Ring:
    Add<Output = Self>
    + Sub<Output = Self>
    + Sized
    + PartialEq
    + Eq
    + Sub
    + Neg<Output = Self>
    + Copy
    + Mul<Output = Self>
    + Sum
{
    const ZERO: Self;
    const ONE: Self;

    /// For two iterators of the same length, return their dot product.
    /// # Safety
    /// The caller must ensure that the iterators are of the same length.
    fn dot_product(
        left_iterator: impl Iterator<Item = Self>,
        right_iterator: impl Iterator<Item = Self>,
    ) -> Self {
        left_iterator
            .zip(right_iterator)
            .map(|(left, right)| left * right)
            .sum()
    }

    /// Returns `c` such that `c * b = a`, if exists.
    fn try_left_divide(self, b: Self) -> Option<Self>;

    /// Returns `c` such that `b * c = a`, if exists.
    fn try_right_divide(self, b: Self) -> Option<Self>;

    /// determines if there exists `c` such that `b = a * c`
    fn left_divides(self, b: Self) -> bool {
        Self::try_left_divide(b, self).is_some()
    }

    /// determines if there exists `c` such that `b = a * c`
    fn right_divides(self, b: Self) -> bool {
        Self::try_right_divide(b, self).is_some()
    }

    /// This function returns a pair `(a_canon, to_canon, from_canon)` such that
    /// - `a_canon` is associated to `a`
    /// - `to_canon * a = a_canon`
    /// - `from_canon * a_canon = a`
    /// - `canonize(a_canon) = (a_canon, _, _)`.
    /// - `Self::ONE` is `canonize`d.
    fn canonize(a: Self) -> (Self, Self, Self);

    /// Determines if `a` is canonized (see ``Self::canonize``).
    fn is_canonized(a: Self) -> bool {
        let (a_canon, _, _) = Self::canonize(a);
        a_canon == a
    }
}

pub trait CommutativeRing: Ring {
    fn try_divide(self, b: Self) -> Option<Self> {
        self.try_left_divide(b)
    }

    fn divides(self, b: Self) -> bool {
        Self::try_divide(b, self).is_some()
    }

    fn is_unit(a: Self) -> bool {
        Self::divides(a, Self::ONE)
    }
}

/// The ring is commutative since now notions of left and right ideal coincide.
/// Note that all finitely generated ideals in `Gcd` are principal.
/// There is no assumption that the ring is an integral domain.
pub trait Gcd: CommutativeRing {
    /// Returns `(g,x,y)` such that `a * x + b * y = g` and `g = gcd(a,b)`.
    /// Since the greatest common divisor is unique up to association,
    /// we return the `canonize`d one.
    fn extended_gcd(a: Self, b: Self) -> (Self, Self, Self);
    fn gcd(a: Self, b: Self) -> Self {
        let (g, _, _) = Self::extended_gcd(a, b);
        g
    }
}

/// A Ring in which all ideals are finitely generated.
/// There is no assumption that the ring is an integral domain.
pub trait Noetherian: Ring {}

/// A Ring in which all ideals are principal.
/// It is a type alias since it is equivalent to showing
/// that all ideals are finitely generated and that all finitely generated ideals are principal.
///
/// There is no assumption that the ring is an integral domain.
pub trait Principal = Noetherian + Gcd;

/// An euclidian domain. It is expected that for `a`, `b` there exist `q = Self::div(a,b)` and
/// `r = Self::rem(a,b)` such that `r = Self::zero()` or `Self::norm(r) < Self::norm(b)`.
pub trait Euclidean: CommutativeRing + Div<Output = Self> + Rem<Output = Self> {
    /// The euclidian norm.
    fn norm(a: Self) -> Option<NonZero<usize>>;

    fn try_divide(a: Self, b: Self) -> Option<Self> {
        (b != Self::ZERO).then(|| a / b)
    }
}

#[allow(clippy::many_single_char_names, reason = "these are math functions")]
impl<R: Euclidean> Gcd for R {
    fn extended_gcd(a: Self, b: Self) -> (Self, Self, Self) {
        #[inline]
        fn extended_euclidean_algorithm<R: Euclidean>(a: R, b: R) -> (R, R, R) {
            if b == R::ZERO {
                (a, R::ONE, R::ZERO)
            } else {
                let (g, x, y) = extended_euclidean_algorithm(b, a % b);
                (g, y, x - (a / b) * y)
            }
        }

        let (gcd, x, y) = extended_euclidean_algorithm(a, b);
        let (gcd_canon, to_canon, _) = Self::canonize(gcd);
        (gcd_canon, x * to_canon, y * to_canon)
    }

    fn gcd(a: Self, b: Self) -> Self {
        #[inline]
        fn euclidean_algorithm<R: Euclidean>(a: R, b: R) -> R {
            if b == R::ZERO {
                a
            } else {
                euclidean_algorithm(b, a % b)
            }
        }

        let g = euclidean_algorithm(a, b);
        let (g_canon, _, _) = Self::canonize(g);
        g_canon
    }
}

/// Iterator over all the elements of a set.
pub trait Finite: Sized + Copy + Hash {
    type Output: Copy + Hash = Self;
    fn elements() -> impl ExactSizeIterator<Item = Self::Output>;
}
