use super::*;
pub type Integer = i32;
pub type Nat = u32;

impl AbelianGroup for Integer {
    fn zero() -> Self {
        0
    }
}

impl Ring for Integer {
    fn one() -> Self {
        1
    }
}

impl Bezout for Integer {
    fn gcd(a: Self, b: Self) -> (Self, Self, Self) {
        fn gcd_helper(a: Integer, b: Integer) -> (Integer, Integer, Integer) {
            match b {
                0 => (a, 1, 0),
                c => {
                    let (gcd, x, y) = gcd_helper(c, a % c);
                    (gcd, y, x - (a / c) * y)
                }
            }
        }

        let (gcd, x, y) = gcd_helper(a, b);
        (
            gcd * Integer::signum(gcd),
            x * Integer::signum(gcd),
            y * Integer::signum(gcd),
        )
    }

    fn try_divide(a: Self, b: Self) -> Option<Self> {
        (b != 0 && a % b == 0).then(|| a / b)
    }
}

impl Noetherian for Integer {}

impl Euclidian for Integer {
    fn norm(a: Self) -> Option<NonZero<Nat>> {
        NonZero::new(a.unsigned_abs())
    }

    fn divide_with_reminder(a: Self, b: Self) -> (Self, Self) {
        (a / b, a % b)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn gcd() {
        assert_eq!(Integer::gcd(2, 3), (1, -1, 1));
    }

    #[test]
    fn gcd_negative() {
        assert_eq!(Integer::gcd(-2, -3), (1, 1, -1));
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

    #[test]
    fn divide_with_reminder() {
        use rand::random;
        (0..1000)
            .map(|_| (random::<Integer>(), random::<Integer>()))
            .for_each(|(a, b)| {
                let (q, r) = Integer::divide_with_reminder(a, b);
                assert_eq!(a, q * b + r)
            });
    }
}
