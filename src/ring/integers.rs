use super::*;
pub type Integer = i16;

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

impl PID for Integer {
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

        let (gcd, x, y) = gcd_helper(a,b);
        (gcd * Integer::signum(gcd), x * Integer::signum(gcd), y * Integer::signum(gcd))
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
}
