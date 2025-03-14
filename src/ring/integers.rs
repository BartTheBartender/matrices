use super::{CommutativeRing, Euclidean, Integer, NonZero, Ring};

impl Ring for Integer {
    const ZERO: Self = 0;
    const ONE: Self = 1;

    fn canonize(a: Self) -> (Self, Self) {
        (a.abs(), a.signum())
    }

    fn try_left_divide(self, b: Self) -> Option<Self> {
        (b != 0).then(|| self / b)
    }

    fn try_right_divide(self, b: Self) -> Option<Self> {
        (b != 0).then(|| self / b)
    }
}

impl CommutativeRing for Integer {}
impl Euclidean for Integer {
    #[allow(
        clippy::as_conversions,
        reason = "I use 64 bit architecture"
    )]
    fn norm(a: Self) -> Option<NonZero<usize>> {
        NonZero::new(a.unsigned_abs() as usize)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ring::Gcd;
    #[test]
    fn gcd() {
        assert_eq!(Integer::extended_gcd(2, 3), (1, -1, 1));
    }

    #[test]
    fn extended_gcd_negative() {
        assert_eq!(Integer::extended_gcd(-2, -3), (1, 1, -1));
    }

    #[test]
    fn canonize() {
        let x : Integer = -58;
        let (x_canon, to_canon) = Integer::canonize(x);
        assert_eq!(x_canon, 58);
        assert_eq!(to_canon, -1);
    }

    #[test]
    fn dot_product_1() {
        assert_eq!(
            Integer::dot_product(vec![1, 2, 3].into_iter(), vec![3, 2, 1].into_iter()),
            10
        );
    }

    #[test]
    fn dot_product_2() {
        assert_eq!(
            Integer::dot_product(vec![1, 2, 3].into_iter(), vec![3, 2, 1].into_iter()),
            10
        );
    }
}
