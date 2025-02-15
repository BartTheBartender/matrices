mod gaussian_elimination;
mod smith_normal_form;
mod vec2d;

use crate::ring::{Finite, Ring};
use custom_error::custom_error;
use itertools::{iproduct, Itertools};
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
    pub fn zero(nof_rows: usize, nof_cols: usize) -> Self {
        Self {
            nof_cols,
            nof_rows,
            buffer: vec![R::zero(); nof_cols * nof_rows],
        }
    }

    pub fn mul_col_by(&mut self, col_idx: usize, r: R) -> Result<(), MatrixError> {
        self.col_mut(col_idx)
            .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })
            .map(|col_mut| col_mut.for_each(|entry| *entry = *entry * r))
    }

    pub fn mul_row_by(&mut self, row_idx: usize, r: R) -> Result<(), MatrixError> {
        self.row_mut(row_idx)
            .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })
            .map(|row_mut| row_mut.for_each(|entry| *entry = *entry * r))
    }

    pub fn add_col_to_col(
        &mut self,
        col_idx_1: usize,
        col_idx_2: usize,
    ) -> Result<(), MatrixError> {
        (col_idx_1 != col_idx_2)
            .then(|| {
                let col_1 = self
                    .col(col_idx_1)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })
                    .map(|col_1| col_1.copied().collect::<Vec<_>>().into_iter())?;

                let col_2 = self
                    .col_mut(col_idx_2)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })?;

                Ok(col_2.zip(col_1).for_each(|(col_2_entry, col_1_entry)| {
                    *col_2_entry = *col_2_entry + col_1_entry
                }))
            })
            .ok_or(MatrixError::AddedColToItself { idx: col_idx_1 })
            .flatten()
    }

    pub fn add_muled_col_to_col(
        &mut self,
        r: R,
        col_idx_1: usize,
        col_idx_2: usize,
    ) -> Result<(), MatrixError> {
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

                Ok(col_2.zip(col_1).for_each(|(col_2_entry, col_1_entry)| {
                    *col_2_entry = *col_2_entry + col_1_entry
                }))
            })
            .ok_or(MatrixError::AddedColToItself { idx: col_idx_1 })
            .flatten()
    }

    pub fn add_row_to_row(
        &mut self,
        row_idx_1: usize,
        row_idx_2: usize,
    ) -> Result<(), MatrixError> {
        (row_idx_1 != row_idx_2)
            .then(|| {
                let row_1 = self
                    .row(row_idx_1)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })
                    .map(|row_1| row_1.copied().collect::<Vec<_>>().into_iter())?;

                let row_2 = self
                    .row_mut(row_idx_2)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })?;

                Ok(row_2.zip(row_1).for_each(|(row_2_entry, row_1_entry)| {
                    *row_2_entry = *row_2_entry + row_1_entry
                }))
            })
            .ok_or(MatrixError::AddedRowToItself { idx: row_idx_1 })
            .flatten()
    }

    pub fn add_muled_row_to_row(
        &mut self,
        r: R,
        row_idx_1: usize,
        row_idx_2: usize,
    ) -> Result<(), MatrixError> {
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

                Ok(row_2.zip(row_1).for_each(|(row_2_entry, row_1_entry)| {
                    *row_2_entry = *row_2_entry + row_1_entry
                }))
            })
            .ok_or(MatrixError::AddedRowToItself { idx: row_idx_1 })
            .flatten()
    }

    /////////

    pub fn sub_col_from_col(
        &mut self,
        col_idx_1: usize,
        col_idx_2: usize,
    ) -> Result<(), MatrixError> {
        (col_idx_1 != col_idx_2)
            .then(|| {
                let col_1 = self
                    .col(col_idx_1)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })
                    .map(|col_1| col_1.copied().collect::<Vec<_>>().into_iter())?;

                let col_2 = self
                    .col_mut(col_idx_2)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })?;

                Ok(col_2.zip(col_1).for_each(|(col_2_entry, col_1_entry)| {
                    *col_2_entry = *col_2_entry - col_1_entry
                }))
            })
            .ok_or(MatrixError::AddedColToItself { idx: col_idx_1 })
            .flatten()
    }

    pub fn sub_muled_col_from_col(
        &mut self,
        r: R,
        col_idx_1: usize,
        col_idx_2: usize,
    ) -> Result<(), MatrixError> {
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

                Ok(col_2.zip(col_1).for_each(|(col_2_entry, col_1_entry)| {
                    *col_2_entry = *col_2_entry - col_1_entry
                }))
            })
            .ok_or(MatrixError::AddedColToItself { idx: col_idx_1 })
            .flatten()
    }

    pub fn sub_row_from_row(
        &mut self,
        row_idx_1: usize,
        row_idx_2: usize,
    ) -> Result<(), MatrixError> {
        (row_idx_1 != row_idx_2)
            .then(|| {
                let row_1 = self
                    .row(row_idx_1)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })
                    .map(|row_1| row_1.copied().collect::<Vec<_>>().into_iter())?;

                let row_2 = self
                    .row_mut(row_idx_2)
                    .map_err(|vec_2d_error| MatrixError::Vec2d { vec_2d_error })?;

                Ok(row_2.zip(row_1).for_each(|(row_2_entry, row_1_entry)| {
                    *row_2_entry = *row_2_entry - row_1_entry
                }))
            })
            .ok_or(MatrixError::AddedRowToItself { idx: row_idx_1 })
            .flatten()
    }

    pub fn sub_muled_row_from_row(
        &mut self,
        r: R,
        row_idx_1: usize,
        row_idx_2: usize,
    ) -> Result<(), MatrixError> {
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

                Ok(row_2.zip(row_1).for_each(|(row_2_entry, row_1_entry)| {
                    *row_2_entry = *row_2_entry - row_1_entry
                }))
            })
            .ok_or(MatrixError::AddedRowToItself { idx: row_idx_1 })
            .flatten()
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
        self * (&vec)
    }
}

impl<R: Ring> Mul<&Vec<R>> for Matrix<R> {
    type Output = Vec<R>;
    fn mul(self, vec: &Vec<R>) -> Self::Output {
        (&self) * vec
    }
}

impl<R: Ring> Mul<Vec<R>> for Matrix<R> {
    type Output = Vec<R>;
    fn mul(self, vec: Vec<R>) -> Self::Output {
        (&self) * (&vec)
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

    #[test]
    fn mul_matrix_vector_1() {
        let a = M::from_rows_vec(vec![vec![3, 0], vec![0,4]]).expect("This should be well-defined");
        let v = vec![1,2];
        
        assert_eq!(a * v, vec![3,8]);

    }

    #[test]
    fn mul_matrix_vector_2() {
        let a = M::from_rows_vec(vec![vec![3, 0], vec![0,4], vec![1,-1]]).expect("This should be well-defined");
        let v = vec![1,2];
        
        assert_eq!(a * v, vec![3,8,-1]);

    }

    #[test]
    #[should_panic(expected = "Incorrect shapes: matrix.row_len() = 2, vec.len() = 3")]
    fn mul_matrix_vector_wrong_shapes() {
        let a = M::from_rows_vec(vec![vec![3, 0], vec![0,4]]).expect("This should be well-defined");
        let _ = a * vec![1,2,3];
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
    fn sub_col_from_col() {
        let mut matrix =
            M::from_rows_vec(vec![vec![1, 2, 3, 4], vec![2, 4, 6, 8], vec![3, 6, 9, 12]])
                .expect("This should be well-defined.");
        matrix
            .sub_col_from_col(0, 2)
            .expect("This should be well-defined");
        assert_eq!(
            matrix.into_rows_vec(),
            vec![vec![1, 2, 2, 4], vec![2, 4, 4, 8], vec![3, 6, 6, 12]]
        );
    }

    #[test]
    fn sub_muled_col_from_col() {
        let mut matrix =
            M::from_rows_vec(vec![vec![1, 2, 3, 4], vec![2, 4, 6, 8], vec![3, 6, 9, 12]])
                .expect("This should be well-defined.");
        matrix
            .sub_muled_col_from_col(3, 0, 2)
            .expect("This should be well-defined");
        assert_eq!(
            matrix.into_rows_vec(),
            vec![vec![1, 2, 0, 4], vec![2, 4, 0, 8], vec![3, 6, 0, 12]]
        );
    }

    #[test]
    fn sub_row_from_row() {
        let mut matrix =
            M::from_cols_vec(vec![vec![1, 2, 3, 4], vec![2, 4, 6, 8], vec![3, 6, 9, 12]])
                .expect("This should be well-defined.");
        matrix
            .sub_row_from_row(0, 2)
            .expect("This should be well-defined");
        assert_eq!(
            matrix.into_cols_vec(),
            vec![vec![1, 2, 2, 4], vec![2, 4, 4, 8], vec![3, 6, 6, 12]]
        );
    }

    #[test]
    fn sub_muled_row_from_row() {
        let mut matrix =
            M::from_cols_vec(vec![vec![1, 2, 3, 4], vec![2, 4, 6, 8], vec![3, 6, 9, 12]])
                .expect("This should be well-defined.");
        matrix
            .sub_muled_row_from_row(3, 0, 2)
            .expect("This should be well-defined");
        assert_eq!(
            matrix.into_cols_vec(),
            vec![vec![1, 2, 0, 4], vec![2, 4, 0, 8], vec![3, 6, 0, 12]]
        );
    }
}
