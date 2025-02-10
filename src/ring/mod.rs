pub mod integers;

use std::{
    iter::Sum,
    ops::{Add, Mul, Neg, Sub},
};

pub trait AbelianGroup:
    Add<Output = Self> + Neg<Output = Self> + Sub<Output = Self> + Sum + Sized + PartialEq + Eq + Copy
{
    fn zero() -> Self;
}

pub trait Ring: AbelianGroup + Mul<Output = Self> {
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
