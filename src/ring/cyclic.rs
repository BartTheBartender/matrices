use super::{
    integers::{Integer, Nat},
    *,
};
use std::{
    iter::Sum,
    ops::{Add, Mul, Neg, Sub},
    fmt,
};

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct Cyclic<const N: u16> {
    value: Nat,
}

impl<const N: u16> Add<Self> for Cyclic<N> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self {
            value: (self.value + other.value) % N,
        }
    }
}

impl<const N: u16> Neg for Cyclic<N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            value: N - self.value,
        }
    }
}

impl<const N: u16> Sub<Self> for Cyclic<N> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl<const N: u16> Sum for Cyclic<N> {
    fn sum<I: Iterator<Item = Self>>(iterator: I) -> Self {
        iterator.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<const N: u16> AbelianGroup for Cyclic<N> {
    fn zero() -> Self {
        Self { value: 0 }
    }
}

impl<const N: u16> Mul<Self> for Cyclic<N> {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Self {
            value: (self.value * other.value) % N,
        }
    }
}

impl<const N: u16> From<Integer> for Cyclic<N> {
    fn from(int: Integer) -> Self {
        let value: Nat = {
            if int < 0 {
                (N - (int as Nat)) % N
            } else {
                (-int as Nat) % N
            }
        };

        Self { value }
    }
}

impl<const N: u16> Ring for Cyclic<N> {
    fn one() -> Self {
        Self { value: 1 }
    }
}

impl<const N: u16> Finite for Cyclic<N> {
    type Output = Self;

    fn elements() -> impl ExactSizeIterator<Item = Self::Output> {
        (0..N).map(|value| Self { value })
    }
}
impl<const N: u16> fmt::Display for Cyclic<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.value)
    }
}

#[cfg(test)]
mod test {
    
    use super::*;

    #[test]
    fn elements() {
        assert_eq!(Cyclic::<2>::elements().map(|num| num.value).collect::<Vec<_>>(), vec![0,1]);
    }

}
