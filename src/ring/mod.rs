pub mod integers;
pub mod cyclic;

pub use integers::Integer as Integer;
pub use integers::Nat as Nat;
pub use cyclic::Cyclic as Cyclic;


use std::{
    iter::Sum,
    num::NonZero,
    ops::{Add, Mul, Neg, Sub},
    hash::Hash,
};

pub trait AbelianGroup:
    Add<Output = Self> + Neg<Output = Self> + Sub<Output = Self> + Sum + Sized + PartialEq + Eq + Copy
{
    fn zero() -> Self;
}

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
}

pub trait PID: Ring {
    fn gcd(x: Self, y: Self) -> (Self, Self, Self);
}

pub trait Euclidian: PID {
    fn valuation(self) -> Option<NonZero<Nat>>;
}

pub trait Field: Euclidian {
    fn valuation(self) -> Option<NonZero<Nat>> {
        (self == Self::zero()).then(|| unsafe { NonZero::new_unchecked(1) })
    }
}

pub trait Finite: Copy + Hash {
    type Output: Copy;
    fn elements() -> impl ExactSizeIterator<Item = Self::Output>;
}
