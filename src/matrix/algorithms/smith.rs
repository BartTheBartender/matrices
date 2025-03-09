use crate::{matrix::Matrix, ring::Euclidian};
use std::{
    fmt::{Debug, Display},
    ops::Not,
};

#[allow(
    non_snake_case,
    clippy::arithmetic_side_effects,
    reason = "this is a matrix algorithm"
)]
//impl<R: Euclidian> Matrix<R> {
impl<R: Euclidian + Debug + Display> Matrix<R> {
    /// Returns the element in the principal minor in the submatrix ``A[minor_idx..][minor_idx..]`` with the smallest ``R::norm()``.
    fn pivot(&self, minor_idx: usize) -> Option<(usize, usize)> {
        self.cols()
            .skip(minor_idx)
            .zip(minor_idx..self.nof_cols())
            .flat_map(|(col_iterator, col_idx)| {
                col_iterator
                    .skip(minor_idx)
                    .zip(minor_idx..self.col_len())
                    .map(move |(entry, row_idx)| ((row_idx, col_idx), entry))
            })
            .filter_map(|(idx, entry)| R::norm(*entry).map(|norm| (idx, norm)))
            .min_by(|&(_, norm_a), &(_, norm_b)| norm_a.cmp(&norm_b))
            .map(|(idx, _)| idx)
    }

    /// Assuming that the matrix was updated by ``Self::clear_row`` and ``Self::clear_col``
    /// find an entry ``(i,j)`` such that ``pivot`` does not divide ``self[i,j]``.
    fn non_divisible_entry(&self, minor_idx: usize) -> Option<(usize, usize)> {
        let pivot = *self
            .get(minor_idx, minor_idx)
            .expect("The pivot index is in proper bounds.");
        self.cols()
            .skip(minor_idx)
            .zip(minor_idx..self.nof_cols())
            .flat_map(|(col_iterator, col_idx)| {
                col_iterator
                    .skip(minor_idx)
                    .zip(minor_idx..self.col_len())
                    .map(move |(entry, row_idx)| ((row_idx, col_idx), entry))
            })
            .find_map(|(idx, &entry)| R::divides(pivot, entry).not().then_some(idx))
    }

    fn clear_row(A: &mut Self, Q: &mut Self, minor_idx: usize) {
        debug_assert_ne!(
            *A.get(minor_idx, minor_idx)
                .expect("The indices are in proper bounds."),
            R::zero(),
            "The pivot in clear_col cannot be zero."
        );
        for col_idx in minor_idx.saturating_add(1)..A.nof_cols() {
            while *A
                .get(minor_idx, col_idx)
                .expect("The indices are in proper bounds.")
                != R::zero()
            {
                let (q, r) = R::divide_with_reminder(
                    *A.get(minor_idx, col_idx)
                        .expect("The indices in A are in proper bounds."),
                    *A.get(minor_idx, minor_idx)
                        .expect("The indices in T are in proper bounds."),
                );
                A.add_muled_col_to_col(-q, minor_idx, col_idx)
                    .expect("Addition of col to col in A will succeed.");
                Q.add_muled_col_to_col(-q, minor_idx, col_idx)
                    .expect("Addition of col to col in T will succeed.");
                if *A
                    .get(minor_idx, col_idx)
                    .expect("The indices in A are in proper bounds.")
                    != R::zero()
                {
                    A.swap_cols(minor_idx, col_idx)
                        .expect("Swapping cols in A will succeed.");
                    Q.swap_cols(minor_idx, col_idx)
                        .expect("Swapping cols in Q will succeed.");
                    debug_assert_eq!(
                        *A.get(minor_idx, minor_idx)
                            .expect("The pivot should be in proper bounds."),
                        r,
                        "After performing the reduction this should be the pivot."
                    );
                }
            }
        }
    }

    fn clear_col(A: &mut Self, S: &mut Self, minor_idx: usize) {
        A.transpose();
        S.transpose();
        Self::clear_row(A, S, minor_idx); // ``A^T * S^T = (SA) ^T ``
        A.transpose();
        S.transpose();
    }

    /// # Panics
    #[must_use]
    pub fn smith(self) -> (Self, Self, Self) {
        #[cfg(debug_assertions)]
        let A_old = self.clone();

        let mut A = self;
        let mut P = Self::identity(A.nof_rows());
        let mut Q = Self::identity(A.nof_cols());

        //debug_assert_eq!(
        //    (&P) * (&A_old),
        //    A,
        //    "Matrix P should be the proper identity at the start."
        //);
        //debug_assert_eq!(
        //    (&A_old) * (&Q),
        //    A,
        //    "Matrix Q should be the proper identity at the start."
        //);

        for minor_idx in 0..std::cmp::min(A.nof_rows(), A.nof_cols()) {
            loop {
                if let Some((piv_row, piv_col)) = A.pivot(minor_idx) {
                    {
                        A.swap_rows(minor_idx, piv_row)
                            .expect("Indices of swapped rows of A are in proper bounds.");
                        P.swap_rows(minor_idx, piv_row)
                            .expect("Indices of swapped rows of Q are in proper bounds.");
                        A.swap_cols(minor_idx, piv_col)
                            .expect("Indices of swapped rows of A are in proper bounds.");
                        Q.swap_cols(minor_idx, piv_col)
                            .expect("Indices of swapped rows of Q are in proper bounds.");
                    }
                    //debug_assert_eq!(
                    //    (&P) * (&A_old) * (&Q),
                    //    A,
                    //    "Matrices should be the proper change of basis after swapping rows."
                    //);
                    //debug_assert_eq!(
                    //    (&P) * (&A_old) * (&Q),
                    //    A,
                    //    "Matrices should be the proper change of basis after swapping cols."
                    //);

                    Self::clear_row(&mut A, &mut Q, minor_idx);
                    Self::clear_col(&mut A, &mut P, minor_idx);

                    //debug_assert_eq!(
                    //    (&P) * (&A_old) * (&Q),
                    //    A,
                    //    "Matrices should be the proper change of basis after clearing a col."
                    //);
                    //debug_assert_eq!(
                    //    (&P) * (&A_old) * (&Q),
                    //    A,
                    //    "Matrices should be the proper change of basis after clearing a row."
                    //);

                    if let Some((bad_idx, _)) = A.non_divisible_entry(minor_idx) {
                        A.add_col_to_col(bad_idx, minor_idx)
                            .expect("The bad and the minor indices in A should be well-defined.");
                        Q.add_col_to_col(bad_idx, minor_idx)
                            .expect("The bad and the minor indices in Q should be well-defined.");
                        //debug_assert_eq!(
                        //(&P) * (&A_old) * (&Q),
                        //    A,
                        //    "Matrices P and Q should be the proper change of basis after detection of bad entry."
                        //);
                    } else {
                        let (_, to_canon) = R::canonize(
                            *A.get(minor_idx, minor_idx)
                                .expect("The pivot in canonizing should be well-defined."),
                        );

                        A.mul_col_by(minor_idx, to_canon)
                            .expect("Multiplying the col of A to canonize cannot fail.");
                        Q.mul_col_by(minor_idx, to_canon)
                            .expect("Multiplying the col of Q to canonize cannot fail.");

                        break;
                    }
                } else {
                    debug_assert!(
                        A.is_in_smith_normal_form(),
                        "The matrix should be in the smith normal form."
                    );
                    debug_assert_eq!(
                        (&P) * (&A_old) * (&Q),
                        A,
                        "The matrices P and Q are not a desired change of basis in snf."
                    );
                    return (P, A, Q);
                }
            }
        }
        debug_assert!(
            A.is_in_smith_normal_form(),
            "The matrix should be in the smith normal form."
        );
        debug_assert_eq!(
            (&P) * (&A_old) * (&Q),
            A,
            "The matrices P and Q are not a desired change of basis in snf."
        );
        (P, A, Q)
    }

    /// Matrix is in a smith normal form if
    /// * it is diagonal
    /// * ``d_i`` divides ``d_{i+1}`` for the diagonal
    /// # Panics
    #[must_use]
    pub fn is_in_smith_normal_form(&self) -> bool {
        use std::cmp::min;
        (0..min(self.nof_rows(), self.nof_cols()))
            .zip(1..min(self.nof_rows(), self.nof_cols()))
            .map(|(curr, next)| {
                (
                    self.get(curr, curr)
                        .expect("The curr index is in proper bounds."),
                    self.get(next, next)
                        .expect("The curr index is in proper bounds."),
                )
            })
            .all(|(&curr, &next)| R::is_canonized(curr) && R::divides(curr, next))
    }
}

#[allow(non_snake_case, reason = "this is a matrix algorithm")]
#[cfg(test)]
mod test {
    use super::*;
    use crate::ring::{AbelianGroup, Integer};
    type M = Matrix<Integer>;

    fn matrix() -> M {
        M::from_rows_arr([
            [1, -1, 5, 7, -2, 4],
            [0, 2, -3, 6, 1, 8],
            [9, -4, 1, 0, 5, -7],
            [2, 3, 8, -5, -6, 0],
            [4, 1, -2, 9, 3, 6],
        ])
    }

    fn random_matrix(nof_rows: usize, nof_cols: usize) -> M {
        use rand::Rng;
        M::from_cols(
            (0..nof_cols)
                .map(move |_| (0..nof_rows).map(move |_| rand::rng().random_range(1_i64..100_i64))),
        )
        .expect("This is well-defined since all inner iterators have the same length.")
    }

    #[test]
    fn pivot() {
        assert_eq!(matrix().pivot(0), Some((0, 0)));
        assert_eq!(matrix().pivot(1), Some((4, 1)));
        assert_eq!(matrix().pivot(2), Some((2, 2)));
        assert_eq!(M::zero(6, 7).pivot(1), None);
    }

    #[test]
    fn clear_row() {
        let mut A = matrix();
        let mut Q = M::identity(A.nof_cols());
        assert_eq!(matrix() * (&Q), A);
        M::clear_row(&mut A, &mut Q, 0);
        assert_eq!(matrix() * Q, A);
        A.row(0)
            .expect("The 0-th row exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, <Integer as AbelianGroup>::zero()));
    }

    #[test]
    fn clear_col() {
        let mut A = matrix();
        let mut P = M::identity(A.nof_rows());
        assert_eq!((&P) * matrix(), A);
        M::clear_col(&mut A, &mut P, 0);
        assert_eq!(P * matrix(), A);
        A.col(0)
            .expect("The 0-th row exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, <Integer as AbelianGroup>::zero()));
    }

    #[test]
    fn clear_row_many() {
        for _ in 0_i32..100_i32 {
            let A_old = random_matrix(5, 6);
            let mut A = A_old.clone();
            let mut Q = M::identity(A.nof_cols());
            M::clear_row(&mut A, &mut Q, 0);
            assert_eq!(A_old * Q, A);
            A.row(0)
                .expect("The 0-th row exists.")
                .skip(1)
                .for_each(|row_entry| assert_eq!(*row_entry, <Integer as AbelianGroup>::zero()));
        }
    }

    #[test]
    fn clear_col_many() {
        for _ in 0_i32..100_i32 {
            let A_old = random_matrix(5, 6);
            let mut A = A_old.clone();
            let mut P = M::identity(A.nof_rows());
            M::clear_col(&mut A, &mut P, 0);
            assert_eq!(P * A_old, A);
            A.col(0)
                .expect("The 0-th col exists.")
                .skip(1)
                .for_each(|row_entry| assert_eq!(*row_entry, <Integer as AbelianGroup>::zero()));
        }
    }

    #[test]
    fn non_divisible_entry() {
        let mut A = matrix();
        let mut Q = M::identity(A.nof_cols());
        let mut P = M::identity(A.nof_rows());
        M::clear_row(&mut A, &mut Q, 0);
        M::clear_col(&mut A, &mut P, 0);
        assert_eq!(M::non_divisible_entry(&A, 0), None);
        assert_eq!(P * matrix() * Q, A);
    }

    #[test]
    fn clear_row_2nd_time() {
        let mut A = matrix();
        let mut Q = M::identity(A.nof_cols());
        let mut P = M::identity(A.nof_rows());
        M::clear_row(&mut A, &mut Q, 0);
        M::clear_col(&mut A, &mut P, 0);
        M::clear_row(&mut A, &mut Q, 1);
        assert_eq!(P * matrix() * Q, A);

        A.row(0)
            .expect("The 0-th row exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, <Integer as AbelianGroup>::zero()));

        A.col(0)
            .expect("The 0-th col exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, <Integer as AbelianGroup>::zero()));

        A.row(1)
            .expect("The 1-th row exists.")
            .skip(2)
            .for_each(|row_entry| assert_eq!(*row_entry, <Integer as AbelianGroup>::zero()));
    }

    #[test]
    fn clear_col_2nd_time() {
        let mut A = matrix();
        let mut Q = M::identity(A.nof_cols());
        let mut P = M::identity(A.nof_rows());
        M::clear_row(&mut A, &mut Q, 0);
        M::clear_col(&mut A, &mut P, 0);
        M::clear_col(&mut A, &mut P, 1);
        assert_eq!(P * matrix() * Q, A);

        A.row(0)
            .expect("The 0-th row exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, <Integer as AbelianGroup>::zero()));

        A.col(0)
            .expect("The 0-th col exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, <Integer as AbelianGroup>::zero()));

        A.col(1)
            .expect("The 1-th row exists.")
            .skip(2)
            .for_each(|row_entry| assert_eq!(*row_entry, <Integer as AbelianGroup>::zero()));
    }

    #[test]
    fn smith() {
        let (P, D, Q) = matrix().smith();
        assert!(D.is_diagonal());
        assert_eq!(P * matrix() * Q, D);
    }

    #[test]
    fn smith_many() {
        for _ in 0_i32..100_i32 {
            let A = random_matrix(5, 6);
            match std::panic::catch_unwind(|| A.clone().smith()) {
                Ok((P, D, Q)) => {
                    assert!(D.is_diagonal());
                    assert_eq!(P * A * Q, D);
                }
                Err(err) => {
                    let message = err
                        .downcast_ref::<&str>()
                        .expect("The only possibility to fail is to multiply by overflow.");
                    assert!(
                        (message == &"attempt to multiply with overflow")
                            || (message == &"attempt to add with overflow")
                    );
                }
            }
        }
    }
}
