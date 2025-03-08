use super::{integers::Integer, *};
use std::{
    fmt,
    iter::Sum,
    ops::{Add, Mul, Neg, Sub},
};

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct Cyclic<const N: u64> {
    value: u64,
}

impl<const N: u64> Add<Self> for Cyclic<N> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self {
            value: (self.value + other.value) % N,
        }
    }
}

impl<const N: u64> Neg for Cyclic<N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            value: N - self.value,
        }
    }
}

impl<const N: u64> Sub<Self> for Cyclic<N> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl<const N: u64> Sum for Cyclic<N> {
    fn sum<I: Iterator<Item = Self>>(iterator: I) -> Self {
        iterator.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<const N: u64> AbelianGroup for Cyclic<N> {
    fn zero() -> Self {
        Self { value: 0 }
    }
}

impl<const N: u64> Mul<Self> for Cyclic<N> {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Self {
            value: (self.value * other.value) % N,
        }
    }
}

impl<const N: u64> From<Integer> for Cyclic<N> {
    fn from(int: Integer) -> Self {
        let shifted: u64 = if int < 0 {
            let n_as_integer = Integer::try_from(N).expect("This overflow should not occur");
            u64::try_from(((int % n_as_integer) + n_as_integer) % n_as_integer)
                .expect("This overflow cannot occur.")
        } else {
            int.try_into().expect("The overflow cannot occur")
        };

        Self { value: shifted % N }
    }
}

impl<const N: u64> Ring for Cyclic<N> {
    fn one() -> Self {
        Self { value: 1 }
    }
}

impl<const N: u64> Bezout for Cyclic<N> {
    fn gcd(a: Self, b: Self) -> (Self, Self, Self) {
        let (g, x, y) = Integer::gcd(
            a.value.try_into().expect("This should be convertable."),
            b.value.try_into().expect("This should be convertable."),
        );
        (Self::from(g), Self::from(x), Self::from(y))
    }

    fn try_divide(a: Self, b: Self) -> Option<Self> {
        let a_v = a.value.try_into().expect("The overflow should not occur.");
        let b_v = b.value.try_into().expect("The overflow should not occur.");
        let (d, _, _) = Integer::gcd(a_v, b_v);

        let (g, x, _) = Integer::gcd(
            b_v / d,
            N.try_into().expect("The overflow should not occur."),
        );
        // a/b = a_v / b_v = (a_v / d) / (b_v / d)
        // it suffices for b_v / d to be invertible mod N
        (g == 1).then(|| Self::from(a_v / d) * Self::from(x))
    }
}

//impl<const N: u64> Finite for Cyclic<N> {
//    type Output = Self;
//
//    fn elements() -> impl ExactSizeIterator<Item = Self::Output> {
//        (0..N).map(|value| Self { value })
//    }
//}
impl<const N: u64> fmt::Display for Cyclic<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.value)
    }
}

//#[cfg(test)]
//mod test {
//
//    use super::*;
//
//    #[test]
//    fn elements() {
//        assert_eq!(
//            Cyclic::<2>::elements()
//                .map(|num| num.value)
//                .collect::<Vec<_>>(),
//            vec![0, 1]
//        );
//    }
//}
