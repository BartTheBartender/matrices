pub mod cyclic;
pub mod integers;

pub use cyclic::Cyclic;
pub use integers::Integer;
pub use integers::Nat;

use std::{
    hash::Hash,
    iter::Sum,
    num::NonZero,
    ops::{Add, Mul, Neg, Sub},
};

pub trait AbelianGroup:
    Add<Output = Self> + Neg<Output = Self> + Sub<Output = Self> + Sum + Sized + PartialEq + Eq + Copy
{
    fn zero() -> Self;
}

#[allow(
    clippy::arithmetic_side_effects,
    reason = "this is a ring with an arbitrary multiplication"
)]
pub trait Ring: AbelianGroup + Mul<Output = Self> + From<Integer> {
    fn one() -> Self;
    fn dot_product(
        left_iterator: impl Iterator<Item = Self>,
        right_iterator: impl Iterator<Item = Self>,
    ) -> Self {
        left_iterator
            .zip(right_iterator)
            .map(|(left, right)| left * right)
            .sum()
    }

    fn try_divide(a: Self, b: Self) -> Option<Self>;

    /// determines if there exists c such that b = a * c
    fn divides(a: Self, b: Self) -> bool {
        let o = Self::zero();
        if b == o {
            true
        } else if a == o {
            false
        } else {
            Self::try_divide(b, a).is_some()
        }
    }

    /// The group of units $U(R)$ acts on $R$ by left-multiplication.
    /// This function returns a pair ``(a_canon, to_canon)`` such that
    /// - ``a_canon`` is the element representing $U(R) * a$
    /// - ``a * to_canon = to_canon``
    fn canonize(a: Self) -> (Self, Self);

    fn is_canonized(a: Self) -> bool {
        let (a_canon, _) = Self::canonize(a);
        a_canon == a
    }
}

pub trait Bezout: Ring {
    /// Returns (g,x,y) such that ax + by = g and g = gcd(a,b)
    fn gcd(a: Self, b: Self) -> (Self, Self, Self);
}

/// In Rust this can be only a formal statement without real code.
pub trait Noetherian: Ring {}

pub trait Euclidian: Noetherian + Bezout {
    /// The euclidian norm
    fn norm(a: Self) -> Option<NonZero<Nat>>;

    /// returns (q,r) such that a = qb + r and norm(r) < norm(b) or r = 0
    fn divide_with_reminder(a: Self, b: Self) -> (Self, Self);
}

pub trait Finite: Copy + Hash {
    type Output: Copy + Hash = Self;
    /// Iterator over all the elements of the ring
    fn elements() -> impl ExactSizeIterator<Item = Self::Output>;
}
