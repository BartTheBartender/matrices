mod vec2d;

use crate::ring::Ring;
use itertools::iproduct;
use std::ops::{Add, Mul, Neg, Sub};
use vec2d::Vec2d;

pub type Matrix<R: Ring> = Vec2d<R>;

impl<R: Ring> Matrix<R> {
    pub fn zero(nof_rows: usize, nof_cols: usize) -> Self {
        Self {
            nof_cols,
            nof_rows,
            buffer: vec![R::zero(); nof_cols * nof_rows],
        }
    }

    pub fn is_lower_triangular(&self) -> bool {
        self.is_square()
            && (0..self.nof_cols())
                .map(|col_idx| {
                    self.col(col_idx)
                        .expect("This should be well-defined.")
                        .take(col_idx)
                })
                .flatten()
                .all(|&entry| entry == R::zero())
    }

    pub fn is_upper_triangular(&self) -> bool {
        self.is_square()
            && (0..self.nof_cols())
                .map(|col_idx| {
                    self.col(col_idx)
                        .expect("This should be well-defined.")
                        .skip(col_idx + 1)
                })
                .flatten()
                .all(|&entry| entry == R::zero())
    }

    pub fn is_diagonal(&self) -> bool {
        self.is_lower_triangular() && self.is_upper_triangular()
    }
}

/// Addition
impl<R: Ring> Add<&Matrix<R>> for Matrix<R> {
    type Output = Matrix<R>;
    fn add(self, other: &Matrix<R>) -> Self::Output {
        debug_assert_eq!(
            self.shape(),
            other.shape(),
            "Incorrect shapes: self.shape() = {:?}, other.shape() = {:?}.",
            self.shape(),
            other.shape()
        );

        let mut buffer = self.buffer;

        buffer
            .iter_mut()
            .zip(other.buffer.iter())
            .for_each(|(self_entry, other_entry)| *self_entry = *self_entry + *other_entry);
        Self::Output {
            nof_cols: other.nof_cols(),
            nof_rows: other.nof_rows(),
            buffer,
        }
    }
}

impl<R: Ring> Add<Matrix<R>> for &Matrix<R> {
    type Output = Matrix<R>;
    fn add(self, other: Matrix<R>) -> Self::Output {
        other.add(self)
    }
}

impl<R: Ring> Add<&Matrix<R>> for &Matrix<R> {
    type Output = Matrix<R>;
    fn add(self, other: &Matrix<R>) -> Self::Output {
        self.clone().add(other)
    }
}

impl<R: Ring> Add<Matrix<R>> for Matrix<R> {
    type Output = Matrix<R>;
    fn add(self, other: Matrix<R>) -> Self::Output {
        self.add(&other)
    }
}

/// Substraction
impl<R: Ring> Sub<&Matrix<R>> for Matrix<R> {
    type Output = Matrix<R>;
    fn sub(self, other: &Matrix<R>) -> Self::Output {
        debug_assert_eq!(
            self.shape(),
            other.shape(),
            "Incorrect shapes: self.shape() = {:?}, other.shape() = {:?}.",
            self.shape(),
            other.shape()
        );

        let mut buffer = self.buffer;

        buffer
            .iter_mut()
            .zip(other.buffer.iter())
            .for_each(|(self_entry, other_entry)| *self_entry = *self_entry - *other_entry);
        Self::Output {
            nof_cols: other.nof_cols(),
            nof_rows: other.nof_rows(),
            buffer,
        }
    }
}

impl<R: Ring> Sub<Matrix<R>> for &Matrix<R> {
    type Output = Matrix<R>;
    fn sub(self, other: Matrix<R>) -> Self::Output {
        self.clone().sub(&other)
    }
}

impl<R: Ring> Sub<&Matrix<R>> for &Matrix<R> {
    type Output = Matrix<R>;
    fn sub(self, other: &Matrix<R>) -> Self::Output {
        self.clone().sub(other)
    }
}

impl<R: Ring> Sub<Matrix<R>> for Matrix<R> {
    type Output = Matrix<R>;
    fn sub(self, other: Matrix<R>) -> Self::Output {
        self.sub(&other)
    }
}

/// Negation
impl<R: Ring> Neg for Matrix<R> {
    type Output = Matrix<R>;
    fn neg(self) -> Self::Output {
        let nof_cols = self.nof_cols();
        let nof_rows = self.nof_rows();
        let mut buffer = self.buffer;
        buffer.iter_mut().for_each(|entry| *entry = -(*entry));
        Self::Output {
            nof_cols,
            nof_rows,
            buffer,
        }
    }
}

impl<R: Ring> Neg for &Matrix<R> {
    type Output = Matrix<R>;
    fn neg(self) -> Self::Output {
        self.clone().neg()
    }
}

///Mutliplication
impl<R: Ring> Mul<&Matrix<R>> for &Matrix<R> {
    type Output = Matrix<R>;
    fn mul(self, other: &Matrix<R>) -> Self::Output {
        debug_assert_eq!(
            self.row_len(),
            other.col_len(),
            "Incorrect shapes: self.row_len() = {:?}, other.col_len() = {:?}.",
            self.row_len(),
            other.col_len()
        );

        let mut buffer = Vec::<R>::with_capacity(self.col_len() * other.row_len());

        iproduct!(0..other.nof_cols(), 0..self.nof_rows())
            .map(|(other_col_idx, self_row_idx)| {
                (
                    other
                        .col(other_col_idx)
                        .expect("This should be well-defined")
                        .copied(),
                    self.row(self_row_idx)
                        .expect("The row should br well-defined.")
                        .copied(),
                )
            })
            .map(|(other_col, self_row)| R::dot_product(self_row, other_col))
            .collect_into(&mut buffer);

        Self::Output {
            nof_cols: other.nof_cols(),
            nof_rows: self.nof_rows(),
            buffer,
        }
    }
}

impl<R: Ring> Mul<&Matrix<R>> for Matrix<R> {
    type Output = Matrix<R>;
    fn mul(self, other: &Matrix<R>) -> Self::Output {
        <&Matrix<R>>::mul(&self, other)
    }
}

impl<R: Ring> Mul<Matrix<R>> for &Matrix<R> {
    type Output = Matrix<R>;
    fn mul(self, other: Matrix<R>) -> Self::Output {
        <&Matrix<R>>::mul(self, &other)
    }
}

impl<R: Ring> Mul<Matrix<R>> for Matrix<R> {
    type Output = Matrix<R>;
    fn mul(self, other: Matrix<R>) -> Self::Output {
        <&Matrix<R>>::mul(&self, &other)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ring::integers::Integer;

    type M = Matrix<Integer>;

    #[test]
    fn zero() {
        assert_eq!(
            M::zero(3, 2).into_rows_vec(),
            vec![vec![0, 0], vec![0, 0], vec![0, 0]]
        );
    }

    #[test]
    fn add() {
        let a = M::from_rows_vec(vec![vec![1, 2, 3], vec![4, 5, 6]])
            .expect("This should be well-defined.");
        let b = M::from_rows_vec(vec![vec![3, 2, 1], vec![6, 5, 4]])
            .expect("This should be well-defined.");

        assert_eq!(
            (a + b).into_rows_vec(),
            vec![vec![4, 4, 4], vec![10, 10, 10]]
        );
    }

    #[test]
    #[should_panic(expected = "Incorrect shapes: self.shape() = (2, 3), other.shape() = (2, 4).")]
    fn add_wrong_shapes() {
        let a = M::from_rows_vec(vec![vec![1, 2, 3], vec![4, 5, 6]])
            .expect("This should be well-defined.");
        let b = M::from_rows_vec(vec![vec![3, 2, 1, 17], vec![6, 5, 4, 13]])
            .expect("This should be well-defined.");
        let _ = a + b;
    }

    #[test]
    fn neg() {
        let a = M::from_rows_vec(vec![vec![5, 7, 8], vec![7, 8, 9]])
            .expect("This should be well-defined.");
        assert_eq!(
            (-a).into_rows_vec(),
            vec![vec![-5, -7, -8], vec![-7, -8, -9]]
        );
    }

    #[test]
    fn sub() {
        let a = M::from_rows_vec(vec![vec![1, 2, 3], vec![4, 5, 6]])
            .expect("This should be well-defined.");
        let b = M::from_rows_vec(vec![vec![3, 2, 1], vec![6, 5, 4]])
            .expect("This should be well-defined.");

        assert_eq!(
            (a - b).into_rows_vec(),
            vec![vec![-2, 0, 2], vec![-2, 0, 2]]
        );
    }

    #[test]
    #[should_panic(expected = "Incorrect shapes: self.shape() = (2, 3), other.shape() = (2, 4).")]
    fn sub_wrong_shapes() {
        let a = M::from_rows_vec(vec![vec![1, 2, 3], vec![4, 5, 6]])
            .expect("This should be well-defined.");
        let b = M::from_rows_vec(vec![vec![3, 2, 1, 17], vec![6, 5, 4, 13]])
            .expect("This should be well-defined.");
        let _ = a - b;
    }

    #[test]
    fn mul() {
        let a = M::from_rows_vec(vec![vec![1, 2, 3], vec![4, 5, 6]])
            .expect("This should be well-defined.");
        let b = M::from_rows_vec(vec![vec![1, 2, 3, 4], vec![4, 5, 6, 7], vec![7, 8, 9, 0]])
            .expect("This should be well-defined.");

        let c = a * b;

        assert_eq!(
            c.into_rows_vec(),
            vec![vec![30, 36, 42, 18], vec![66, 81, 96, 51]]
        );
    }

    #[test]
    fn mul_by_zero() {
        let a = M::from_rows_vec(vec![vec![1; 2]; 8]).expect("This should be well-defined");

        assert_eq!(M::zero(8, 8) * (&a), M::zero(8, 2));
        assert_eq!((&a) * M::zero(2, 7), M::zero(8, 7));
    }

    #[test]
    #[should_panic(expected = "Incorrect shapes: self.row_len() = 3, other.col_len() = 2.")]
    fn mul_wrong_shapes() {
        let a = M::from_rows_vec(vec![vec![1, 2, 3], vec![4, 5, 6]])
            .expect("This should be well-defined.");
        let b = M::from_rows_vec(vec![vec![3, 2, 1, 17], vec![6, 5, 4, 13]])
            .expect("This should be well-defined.");

        let _ = a * b;
    }
}
