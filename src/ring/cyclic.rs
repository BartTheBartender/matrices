use super::{Gcd, Integer, Natural, Ring};
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
    #[allow(
        clippy::as_conversions,
        reason = "the result of div_euclid will be in 0..N"
    )]
    fn from(int: Integer) -> Self {
        let value = match int.cmp(&0) {
            Ordering::Equal => 0,
            Ordering::Greater | Ordering::Less => {
                Integer::div_euclid(int, Integer::from(N)).unsigned_abs() as Natural
            }
        };

        Self { value }
    }
}
impl<const N: Natural> From<Cyclic<N>> for Integer {
    fn from(cyclic: Cyclic<N>) -> Self {
        Self::from(cyclic.value)
    }
}

impl<const N: Natural> Ring for Cyclic<N> {
    const ZERO: Self = Self { value: 0 };
    const ONE: Self = Self { value: 1 };

    /// The canonized element up to association is `b = gcd(a, N)`.
    fn canonize(a: Self) -> (Self, Self) {
        let (b, x, _) = Integer::extended_gcd(Integer::from(a), Integer::from(N));
        (Self::from(b), Self::from(x))
    }

    fn try_left_divide(self, b: Self) -> Option<Self> {
        let self_integer = Integer::from(self);
        let b_integer = Integer::from(b);

        // First, find common factor of `a` and `b`.
        let common = Integer::gcd(self_integer, b_integer);

        // It is necessary and sufficient for `b / common` to be coprime with `N`, since it is then
        // invertible.
        let (g, b_common_inv, _) = Integer::extended_gcd(b_integer / common, Integer::from(N));

        (Self::from(g) == Self::ONE).then(|| Self::from(self_integer / common * b_common_inv))
    }

    fn try_right_divide(self, b: Self) -> Option<Self> {
        self.try_left_divide(b)
    }
}
