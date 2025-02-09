pub mod integers;

use std::ops::{Add, Mul, Neg, Sub};

pub trait AbelianGroup:
    Add<Output = Self> + Neg<Output = Self> + Sub<Output = Self> + Sized + PartialEq + Eq
{
    fn zero() -> Self;
}

pub trait Ring: AbelianGroup + Mul<Output = Self> {
    fn one() -> Self;
}

pub trait PID: Ring {
    fn gcd(x: Self, y: Self) -> (Self, Self, Self);
}
