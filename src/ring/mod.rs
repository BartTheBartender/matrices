pub mod cyclic;
pub mod integers;

macro_rules! unsigned_type {
    (i8) => {
        u8
    };
    (i16) => {
        u16
    };
    (i32) => {
        u32
    };
    (i64) => {
        u64
    };
    (i128) => {
        u128
    };
    (isize) => {
        usize
    };
}

pub type Integer = i8;
pub type Natural = unsigned_type!(i8);

use std::{
    hash::Hash,
    num::NonZero,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

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
{
    const ZERO: Self;
    const ONE: Self;

    /// For two iterators of the same length, return their dot product.
    /// # Safety
    /// The caller must ensure that the iterators are of the same length.
    fn dot_product(
        left_iterator: impl Iterator<Item = Self>,
        right_iterator: impl Iterator<Item = Self>,
    ) -> Self
    where
        Self: std::iter::Sum,
    {
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
    fn right_divides(a: Self, b: Self) -> bool {
        Self::try_left_divide(a, b).is_some()
    }
    
    /// This function returns a pair `(a_canon, to_canon)` such that
    /// - `a_canon` is associated to `a`
    /// - `a * to_canon = a_canon`
    /// - `canonize(a_canon) = (a_canon, Self::ONE)`.
    /// - `Self::ONE` is `canonize`d.
    fn canonize(a: Self) -> (Self, Self);

    /// Determines if `a` is canonized (see ``Self::canonize``).
    fn is_canonized(a: Self) -> bool {
        let (a_canon, _) = Self::canonize(a);
        a_canon == a
    }
}

pub trait CommutativeRing: Ring {
    fn try_divide(a: Self, b: Self) -> Option<Self> {
        <Self as Ring>::try_left_divide(a, b)
    }

    fn divides(a: Self, b: Self) -> bool {
        <Self as Ring>::left_divides(a, b)
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
    fn norm(a: Self) -> Option<NonZero<Natural>>;

    fn try_divide(a: Self, b: Self) -> Option<Self> {
        (b != Self::ZERO).then(|| a / b)
    }
}

impl<R: Euclidean> EuclideanAlgorithm for R {
    const ZERO: Self = R::ZERO;
    const ONE: Self = R::ONE;
}

impl<R: Euclidean> Gcd for R {
    #[inline]
    fn extended_gcd(a: Self, b: Self) -> (Self, Self, Self) {
        let (gcd, x, y) = Self::extended_euclidean_algorithm(a, b);
        let (gcd_canon, to_canon) = Self::canonize(gcd);
        (gcd_canon, x * to_canon, y * to_canon)
    }

    #[inline]
    fn gcd(a: Self, b: Self) -> Self {
        let g = Self::euclidean_algorithm(a, b);
        let (g_canon, _) = Self::canonize(g);
        g_canon
    }
}

/// Iterator over all the elements of a set.
pub trait Finite: Sized + Copy + Hash {
    type Output: Copy + Hash = Self;
    fn elements() -> impl ExactSizeIterator<Item = Self::Output>;
}

/// This trait is private, since it is used only to implement generic
/// Euclidean algorithm for Integer and Natural.
trait EuclideanAlgorithm:
    PartialEq
    + Eq
    + Copy
    + Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;

    /// Returns `(g,x,y)` such that `g = gcd(a,b)` and `a * x + b * y = g`.
    fn extended_euclidean_algorithm(a: Self, b: Self) -> (Self, Self, Self) {
        if b == Self::ZERO {
            (a, Self::ONE, Self::ZERO)
        } else {
            let (g, x, y) = Self::extended_euclidean_algorithm(b, a % b);
            (g, y, x - (a / b) * y)
        }
    }

    /// Returns `gcd(a,b)`.
    fn euclidean_algorithm(a: Self, b: Self) -> Self {
        if b == Self::ZERO {
            a
        } else {
            Self::euclidean_algorithm(b, a % b)
        }
    }
}

impl EuclideanAlgorithm for Natural {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}
