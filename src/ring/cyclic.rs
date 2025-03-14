use super::{Integer, Natural};
use std::{
    cmp::Ordering,
    ops::{Add, Mul, Neg, Sub},
};

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct Cyclic<const N: Natural> {
    value: Natural,
}

impl<const N: Natural> Add<Self> for Cyclic<N> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self {
            value: (self.value + other.value) % N,
        }
    }
}
impl<const N: Natural> Neg for Cyclic<N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            value: N - self.value,
        }
    }
}
impl<const N: Natural> Sub<Self> for Cyclic<N> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}
impl<const N: Natural> Mul<Self> for Cyclic<N> {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Self {
            value: (self.value * other.value) % N,
        }
    }
}
impl<const N: Natural> From<Integer> for Cyclic<N> {
    fn from(int: Integer) -> Self {
        match int.cmp(&0) {
            Ordering::Equal => Self { value: 0 },
            Ordering::Greater => Self {
                value: int.unsigned_abs() % N,
            },
            Ordering::Less => Self {
                value: N - (int.unsigned_abs() % N),
            },
        }
    }
}
