pub mod integers;
pub mod cyclic;

pub use integers::Integer as Integer;
pub use integers::Nat as Nat;
pub use cyclic::Cyclic as Cyclic;


use std::{
    iter::Sum,
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

pub trait Bezout: Ring {
    /// Returns (g,x,y) such that ax + by = g and g = gcd(a,b)
    fn gcd(a: Self, b: Self) -> (Self, Self, Self);
    fn try_divide(a: Self, b: Self) -> Option<Self>;
}


pub trait Finite: Copy + Hash {
    type Output: Copy;
    fn elements() -> impl ExactSizeIterator<Item = Self::Output>;
}
