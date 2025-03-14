/// I cannot use const generic parameter of type given by a type alias.
macro_rules! impl_cyclic {
    ($natural_type: tt) => {
use super::{CommutativeRing, Gcd, Integer, Noetherian, Ring};
use std::{
    cmp::Ordering,
    ops::{Add, Mul, Neg, Sub},
    fmt,
};
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct Cyclic<const N: $natural_type> {
    value: $natural_type,
}

impl<const N: $natural_type> Add<Self> for Cyclic<N> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self {
            value: (self.value + other.value) % N,
        }
    }
}
impl<const N: $natural_type> Neg for Cyclic<N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            value: match self.value {
                0 => 0,
                positive => N - positive,
            },
        }
    }
}
impl<const N: $natural_type> Sub<Self> for Cyclic<N> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}
impl<const N: $natural_type> Mul<Self> for Cyclic<N> {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Self {
            value: (self.value * other.value) % N,
        }
    }
}
impl<const N: $natural_type> From<Integer> for Cyclic<N> {
    #[allow(
        clippy::as_conversions,
        reason = "the result of div_euclid will be in 0..N"
    )]
    fn from(int: Integer) -> Self {
        let value = match int.cmp(&0) {
            Ordering::Equal => 0,
            _ => Integer::rem_euclid(int, Integer::from(N)).unsigned_abs() as $natural_type,
        };

        Self { value }
    }
}
impl<const N: $natural_type> From<Cyclic<N>> for Integer {
    fn from(cyclic: Cyclic<N>) -> Self {
        Self::from(cyclic.value)
    }
}

impl<const N: $natural_type> Ring for Cyclic<N> {
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

        // It is necessary and sufficient for `b / common` to be coprime with `N`,
        // since it is then invertible.
        let (g, b_common_inv, _) = Integer::extended_gcd(b_integer / common, Integer::from(N));

        (Self::from(g) == Self::ONE).then(|| Self::from(self_integer / common * b_common_inv))
    }

    fn try_right_divide(self, b: Self) -> Option<Self> {
        self.try_left_divide(b)
    }
}

impl<const N: $natural_type> CommutativeRing for Cyclic<N> {}
impl<const N: $natural_type> Noetherian for Cyclic<N> {}

impl<const N: $natural_type> fmt::Display for Cyclic<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }

}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn from_integer() {
        type R = Cyclic<5>;
        assert_eq!(R::from(1).value, 1);
        assert_eq!(R::from(5).value, 0);
        assert_eq!(R::from(-17).value, 3);
    }

    #[test]
    fn add() {
        type R = Cyclic<5>;

        assert_eq!(R::from(2) + R::from(2), R::from(4));
        assert_eq!(R::from(3) + R::from(3), R::from(1));
        assert_eq!(R::from(3) + R::from(2), R::from(0));
    }

    #[test]
    fn neg() {
        type R = Cyclic<5>;

        assert_eq!(-R::from(0), R::from(0));
        assert_eq!(-R::from(1), R::from(4));
        assert_eq!(-R::from(2), R::from(3));
        assert_eq!(-R::from(3), R::from(2));
        assert_eq!(-R::from(4), R::from(1));
    }

    #[test]
    fn mul() {
        type R = Cyclic<5>;

        assert_eq!(R::from(2) * R::from(2), R::from(4));
        assert_eq!(R::from(3) * R::from(3), R::from(4));
        assert_eq!(R::from(3) * R::from(2), R::from(1));
    }

    #[test]
    fn divides() {
        type R = Cyclic<5>;
        for (i, j) in itertools::iproduct!(1..5, 0..5) {
            assert!(R::from(i).divides(R::from(j)))
        }
    }

}
    };
}

pub type Natural = u16;
impl_cyclic!(u16);
