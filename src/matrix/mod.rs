pub mod algorithms;
pub mod vec2d;

use crate::ring::Ring;
use custom_error::custom_error;
use itertools::iproduct;
use std::ops::{Add, Mul, Neg, Sub};
use vec2d::{Vec2d, Vec2dError};

custom_error! {
    pub MatrixError
    Vec2d{vec_2d_error: Vec2dError} = "Error of the underlying Vec2d: {vec_2d_error}.",
    AddedRowToItself{idx: usize} = "Trying to add row {idx} to itself.",
    AddedColToItself{idx: usize} = "Trying to add col {idx} to itself.",
}

pub type Matrix<R: Ring> = Vec2d<R>;

impl<R: Ring> Matrix<R> {
    /// Determines if for `i` < `j` < `self.nof_cols()` there is a non-zero entry.
    #[must_use]
    pub fn is_upper_triangular(&self) -> bool {
        (1..self.nof_cols()).all(|j| {
            unsafe { self.col_unchecked(j) }
                .take(j.saturating_sub(1)) // all such aij with i < j
                .all(|aij| *aij == R::ZERO)
        })
    }

    /// Determines if for `i` < `j` < `self.nof_cols()` there is a non-zero entry.
    #[must_use]
    pub fn is_lower_triangular(&self) -> bool {
        (0..self.nof_cols()).all(|j| {
            unsafe { self.col_unchecked(j) }
                .skip(j.saturating_add(1)) // all such aij with i > j
                .all(|aij| *aij == R::ZERO)
        })
    }

    /// Determines if a `Matrix` is both lower and upper triangular. Note that it does not require
    /// that it is square.
    #[must_use]
    pub fn is_diagonal(&self) -> bool {
        self.is_upper_triangular() && self.is_lower_triangular()
    }

    /// A zero `nof_rows` x `nof_cols` matrix.
    #[must_use]
    pub fn zero(nof_rows: usize, nof_cols: usize) -> Self {
        Self {
            nof_cols,
            nof_rows,
            buffer: vec![R::ZERO; nof_cols.saturating_mul(nof_rows)],
        }
    }

    /// An identity `nof_rows` x `nof_rows` matrix.
    #[must_use]
    pub fn identity(nof_rows: usize) -> Self {
        let mut id = Self::zero(nof_rows, nof_rows);
        (0..nof_rows).for_each(|i| unsafe {
            *id.get_unchecked_mut(i, i) = R::ONE;
        });
        id
    }

    /// Multiplies `col_idx`-th collumn by `r`.
    /// # Errors
    /// If `col_idx >= self.nof_cols()`, function returns error.
    pub fn mul_col_by(&mut self, col_idx: usize, r: R) -> Result<(), MatrixError> {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Ring operations are defined using math symbols."
        )]
        self.col_mut(col_idx)
            .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })
            .map(|col_mut| col_mut.for_each(|entry| *entry = *entry * r))
    }

    /// Multiplies `row_idx`-th row by `r`.
    /// # Errors
    /// If `row_idx >= self.nof_rows()`, function returns error.
    pub fn mul_row_by(&mut self, row_idx: usize, r: R) -> Result<(), MatrixError> {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Ring operations are defined using math symbols."
        )]
        self.row_mut(row_idx)
            .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })
            .map(|row_mut| row_mut.for_each(|entry| *entry = *entry * r))
    }

    /// Adds `col_idx_1`-th collumn to `col_idx_2`-th collumn. The collumns must be different.
    /// # Errors
    /// If `col_idx_1 == col_idx_2`, or `col_idx_1` or `col_idx_2` is greater than `self.nof_cols()`, function returns error.
    pub fn add_col_to_col(
        &mut self,
        col_idx_1: usize,
        col_idx_2: usize,
    ) -> Result<(), MatrixError> {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Ring operations are defined using math symbols."
        )]
        (col_idx_1 != col_idx_2)
            .then(|| {
                let col_1 = self
                    .col(col_idx_1)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })
                    .map(|col_1| col_1.copied().collect::<Vec<_>>().into_iter())?;

                let col_2 = self
                    .col_mut(col_idx_2)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })?;

                col_2.zip(col_1).for_each(|(col_2_entry, col_1_entry)| {
                    *col_2_entry = *col_2_entry + col_1_entry;
                });
                Ok(())
            })
            .ok_or(MatrixError::AddedColToItself { idx: col_idx_1 })
            .flatten()
    }

    /// Adds `col_idx_1`-th collumn multiplied by 'r' to `col_idx_2`-th collumn. The collumns must
    /// be different.
    /// # Errors
    /// If `col_idx_1 == col_idx_2`, or `col_idx_1` or `col_idx_2` is greater than `self.nof_cols()`, function returns error.
    pub fn add_muled_col_to_col(
        &mut self,
        r: R,
        col_idx_1: usize,
        col_idx_2: usize,
    ) -> Result<(), MatrixError> {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Ring operations are defined using math symbols."
        )]
        (col_idx_1 != col_idx_2)
            .then(|| {
                let col_1 = self
                    .col(col_idx_1)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })
                    .map(|col_1| {
                        col_1
                            .copied()
                            .collect::<Vec<_>>()
                            .into_iter()
                            .map(|col_1_entry| col_1_entry * r)
                    })?;

                let col_2 = self
                    .col_mut(col_idx_2)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })?;

                col_2.zip(col_1).for_each(|(col_2_entry, col_1_entry)| {
                    *col_2_entry = *col_2_entry + col_1_entry;
                });
                Ok(())
            })
            .ok_or(MatrixError::AddedColToItself { idx: col_idx_1 })
            .flatten()
    }

    /// Adds `row_idx_1`-th row to `row_idx_2`-th row. The rows must be different.
    /// # Errors
    /// If `row_idx_1 == row_idx_2`, or `row_idx_1` or `row_idx_2` is greater than `self.nof_rows()`, function returns error.
    pub fn add_row_to_row(
        &mut self,
        row_idx_1: usize,
        row_idx_2: usize,
    ) -> Result<(), MatrixError> {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Ring operations are defined using math symbols."
        )]
        (row_idx_1 != row_idx_2)
            .then(|| {
                let row_1 = self
                    .row(row_idx_1)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })
                    .map(|row_1| row_1.copied().collect::<Vec<_>>().into_iter())?;

                let row_2 = self
                    .row_mut(row_idx_2)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })?;

                row_2.zip(row_1).for_each(|(row_2_entry, row_1_entry)| {
                    *row_2_entry = *row_2_entry + row_1_entry;
                });
                Ok(())
            })
            .ok_or(MatrixError::AddedRowToItself { idx: row_idx_1 })
            .flatten()
    }

    /// Adds `row_idx_1`-th row multiplied by `r` to `row_idx_2`-th row. THe rows must be
    /// different.
    /// # Errors
    /// If `row_idx_1 == row_idx_2`, or `row_idx_1` or `row_idx_2` is greater than `self.nof_rows()`, function returns error.
    pub fn add_muled_row_to_row(
        &mut self,
        r: R,
        row_idx_1: usize,
        row_idx_2: usize,
    ) -> Result<(), MatrixError> {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Ring operations are defined using math symbols."
        )]
        (row_idx_1 != row_idx_2)
            .then(|| {
                let row_1 = self
                    .row(row_idx_1)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })
                    .map(|row_1| {
                        row_1
                            .copied()
                            .collect::<Vec<_>>()
                            .into_iter()
                            .map(|row_1_entry| row_1_entry * r)
                    })?;

                let row_2 = self
                    .row_mut(row_idx_2)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })?;

                row_2.zip(row_1).for_each(|(row_2_entry, row_1_entry)| {
                    *row_2_entry = *row_2_entry + row_1_entry;
                });
                Ok(())
            })
            .ok_or(MatrixError::AddedRowToItself { idx: row_idx_1 })
            .flatten()
    }

    /// For given matrix `a`, returns `a`^`n` such that `a`^`n`=`a`^`n+1`
    /// or None if such power doesn't exists.
    #[must_use]
    pub fn infinite_power(&self) -> Option<Self> {
        let powers = [self.clone()].to_vec();
        self.infinite_power_helper(powers)
    }
    fn infinite_power_helper(&self, mut powers: Vec<Self>) -> Option<Self> {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Matrix multiplication uses math symbols."
        )]
        debug_assert!(self.is_square(), "Cannot iterate non-square matrix.");
        let last_idx = powers.len() - 1; // powers.len() > 0
        let self_nth_power: &Self = unsafe { powers.get_unchecked(last_idx) };
        let self_n_plus_one_power = self_nth_power * self;

        if let Some(repeated_power_idx) = powers
            .iter()
            .position(|self_kth_power| self_kth_power == &self_n_plus_one_power)
        {
            (repeated_power_idx == last_idx).then_some(self_n_plus_one_power)
        } else {
            powers.push(self_n_plus_one_power);
            self.infinite_power_helper(powers)
        }
    }
}

/// Addition
impl<R: Ring> Add<&Self> for Matrix<R> {
    type Output = Self;
    fn add(self, other: &Self) -> Self::Output {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Ring operations are defined using math symbols."
        )]
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

impl<R: Ring> Add<Self> for Matrix<R> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        self.add(&other)
    }
}

/// Substraction
impl<R: Ring> Sub<&Self> for Matrix<R> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self::Output {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Ring operations are defined using math symbols."
        )]
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

impl<R: Ring> Sub<Self> for Matrix<R> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        self.sub(&other)
    }
}

/// Negation
impl<R: Ring> Neg for Matrix<R> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Ring operations are defined using math symbols."
        )]
        let nof_cols = self.nof_cols();
        let nof_rows = self.nof_rows();
        let mut buffer = self.buffer;
        buffer.iter_mut().for_each(|entry| *entry = -(*entry));
        Self::Output {
            buffer,
            nof_rows,
            nof_cols,
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
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Ring operations are defined using math symbols."
        )]
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

impl<R: Ring> Mul<&Self> for Matrix<R> {
    type Output = Self;
    fn mul(self, other: &Self) -> Self::Output {
        <&Matrix<R>>::mul(&self, other)
    }
}

impl<R: Ring> Mul<Matrix<R>> for &Matrix<R> {
    type Output = Matrix<R>;
    fn mul(self, other: Matrix<R>) -> Self::Output {
        <&Matrix<R>>::mul(self, &other)
    }
}

impl<R: Ring> Mul<Self> for Matrix<R> {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        <&Matrix<R>>::mul(&self, &other)
    }
}

///Mutliplication by a vector
impl<R: Ring> Mul<&Vec<R>> for &Matrix<R> {
    type Output = Vec<R>;
    fn mul(self, vec: &Vec<R>) -> Self::Output {
        debug_assert_eq!(
            self.row_len(),
            vec.len(),
            "Incorrect shapes: matrix.row_len() = {}, vec.len() = {}",
            self.row_len(),
            vec.len()
        );
        self.rows()
            .map(|row| R::dot_product(row.copied(), vec.iter().copied()))
            .collect::<Vec<_>>()
    }
}

impl<R: Ring> Mul<Vec<R>> for &Matrix<R> {
    type Output = Vec<R>;
    fn mul(self, vec: Vec<R>) -> Self::Output {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Matrix-vector multiplication uses math symbols."
        )]
        self * (&vec)
    }
}

impl<R: Ring> Mul<&Vec<R>> for Matrix<R> {
    type Output = Vec<R>;
    fn mul(self, vec: &Vec<R>) -> Self::Output {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Matrix-vector multiplication uses math symbols."
        )]
        (&self) * vec
    }
}

impl<R: Ring> Mul<Vec<R>> for Matrix<R> {
    type Output = Vec<R>;
    fn mul(self, vec: Vec<R>) -> Self::Output {
        #![allow(
            clippy::arithmetic_side_effects,
            reason = "Matrix-vector multiplication uses math symbols."
        )]
        (&self) * (&vec)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ring::integers::Integer;

    type M = Matrix<Integer>;

    #[test]
    fn upper_triangular() {
        assert!(
            M::from_rows_arr([[1, 0, 0, 0], [2, 1, 0, 0], [3, 78, 9, 0]]).is_upper_triangular()
        );
    }

    #[test]
    fn lower_triangular() {
        assert!(
            M::from_cols_arr([[1, 0, 0, 0], [2, 1, 0, 0], [3, 78, 9, 0]]).is_upper_triangular()
        );
    }

    #[test]
    fn zero() {
        assert_eq!(
            M::zero(3, 2).into_rows_vec(),
            vec![vec![0, 0], vec![0, 0], vec![0, 0]]
        );
    }

    #[test]
    fn identity() {
        assert_eq!(
            M::identity(3).into_rows_vec(),
            vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]
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

    #[test]
    fn mul_matrix_vector_1() {
        let a =
            M::from_rows_vec(vec![vec![3, 0], vec![0, 4]]).expect("This should be well-defined");
        let v = vec![1, 2];

        assert_eq!(a * v, vec![3, 8]);
    }

    #[test]
    fn mul_matrix_vector_2() {
        let a = M::from_rows_vec(vec![vec![3, 0], vec![0, 4], vec![1, -1]])
            .expect("This should be well-defined");
        let v = vec![1, 2];

        assert_eq!(a * v, vec![3, 8, -1]);
    }

    #[test]
    #[should_panic(expected = "Incorrect shapes: matrix.row_len() = 2, vec.len() = 3")]
    fn mul_matrix_vector_wrong_shapes() {
        let a =
            M::from_rows_vec(vec![vec![3, 0], vec![0, 4]]).expect("This should be well-defined");
        let _ = a * vec![1, 2, 3];
    }

    #[test]
    fn add_col_to_col() {
        let mut matrix =
            M::from_rows_vec(vec![vec![1, 2, 3, 4], vec![2, 4, 6, 8], vec![3, 6, 9, 12]])
                .expect("This should be well-defined.");
        matrix
            .add_col_to_col(0, 2)
            .expect("This should be well-defined");
        assert_eq!(
            matrix.into_rows_vec(),
            vec![vec![1, 2, 4, 4], vec![2, 4, 8, 8], vec![3, 6, 12, 12]]
        );
    }

    #[test]
    fn add_muled_col_to_col() {
        let mut matrix =
            M::from_rows_vec(vec![vec![1, 2, 3, 4], vec![2, 4, 6, 8], vec![3, 6, 9, 12]])
                .expect("This should be well-defined.");
        matrix
            .add_muled_col_to_col(-3, 0, 2)
            .expect("This should be well-defined");
        assert_eq!(
            matrix.into_rows_vec(),
            vec![vec![1, 2, 0, 4], vec![2, 4, 0, 8], vec![3, 6, 0, 12]]
        );
    }

    #[test]
    fn add_row_to_row() {
        let mut matrix =
            M::from_cols_vec(vec![vec![1, 2, 3, 4], vec![2, 4, 6, 8], vec![3, 6, 9, 12]])
                .expect("This should be well-defined.");
        matrix
            .add_row_to_row(0, 2)
            .expect("This should be well-defined");
        assert_eq!(
            matrix.into_cols_vec(),
            vec![vec![1, 2, 4, 4], vec![2, 4, 8, 8], vec![3, 6, 12, 12]]
        );
    }

    #[test]
    fn add_muled_row_to_row() {
        let mut matrix =
            M::from_cols_vec(vec![vec![1, 2, 3, 4], vec![2, 4, 6, 8], vec![3, 6, 9, 12]])
                .expect("This should be well-defined.");
        matrix
            .add_muled_row_to_row(-3, 0, 2)
            .expect("This should be well-defined");
        assert_eq!(
            matrix.into_cols_vec(),
            vec![vec![1, 2, 0, 4], vec![2, 4, 0, 8], vec![3, 6, 0, 12]]
        );
    }

    #[test]
    fn infinite_power_1() {
        let matrix = M::from_rows_vec(vec![vec![0, 1, 0], vec![0, 0, 1], vec![1, 0, 0]])
            .expect("This should be well-defined.");

        let matrix_inf_power = matrix.infinite_power();
        assert!(matrix_inf_power.is_none());
    }

    #[test]
    fn infinite_power_2() {
        let matrix = M::from_rows_vec(vec![vec![2, -2, -4], vec![-1, 3, 4], vec![1, -2, -3]])
            .expect("This should be well-defined.");
        assert_eq!(
            matrix.infinite_power().expect("This matrix is idempotent."),
            matrix
        );
    }
}
