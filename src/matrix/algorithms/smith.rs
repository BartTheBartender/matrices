use crate::{matrix::Matrix, ring::Euclidean};
use std::ops::Not;

#[allow(
    non_snake_case,
    clippy::arithmetic_side_effects,
    reason = "this is a matrix algorithm"
)]
impl<R: Euclidean + std::fmt::Debug + std::fmt::Display> Matrix<R> {
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

    fn clear_row(A: &mut Self, Q: &mut Self, Q_inv: &mut Self, minor_idx: usize) {
        debug_assert!(
            *A.get(minor_idx, minor_idx)
                .expect("The indices are in proper bounds.")
                != R::ZERO,
            "The pivot in clear_row cannot be zero."
        );
        debug_assert_eq!(
            (&*Q) * (&*Q_inv),
            Self::identity(Q.nof_cols()),
            "Q_inv should be an inverse of Q at the start of clear_row."
        );
        debug_assert_eq!(
            (&*Q_inv) * (&*Q),
            Self::identity(Q.nof_cols()),
            "Q_inv should be an inverse of Q at the start of clear_row."
        );

        for col_idx in minor_idx.saturating_add(1)..A.nof_cols() {
            while *A
                .get(minor_idx, col_idx)
                .expect("The indices are in proper bounds.")
                != R::ZERO
            {
                let amc = *A
                    .get(minor_idx, col_idx)
                    .expect("The indices in A are in proper bounds.");
                let amm = *A
                    .get(minor_idx, minor_idx)
                    .expect("The indices in T are in proper bounds.");
                let q = amc / amm;

                A.add_muled_col_to_col(-q, minor_idx, col_idx)
                    .expect("Addition of col to col in A will succeed.");
                Q.add_muled_col_to_col(-q, minor_idx, col_idx)
                    .expect("Addition of col to col in Q will succeed.");
                Q_inv
                    .add_muled_row_to_row(q, col_idx, minor_idx)
                    .expect("Addition of col to col in Q_inv will succeed.");

                debug_assert_eq!(
                    (&*Q) * (&*Q_inv),
                    Self::identity(Q.nof_cols()),
                    "Q_inv should be an inverse of Q after adding col to col."
                );
                debug_assert_eq!(
                    (&*Q_inv) * (&*Q),
                    Self::identity(Q.nof_cols()),
                    "Q_inv should be an inverse of Q after adding col to col."
                );

                if *A
                    .get(minor_idx, col_idx)
                    .expect("The indices in A are in proper bounds.")
                    != R::ZERO
                {
                    A.swap_cols(minor_idx, col_idx)
                        .expect("Swapping cols in A will succeed.");
                    Q.swap_cols(minor_idx, col_idx)
                        .expect("Swapping cols in Q will succeed.");
                    Q_inv
                        .swap_rows(minor_idx, col_idx)
                        .expect("Swapping cols in Q_inv will succeed.");

                    debug_assert_eq!(
                        (&*Q) * (&*Q_inv),
                        Self::identity(Q.nof_cols()),
                        "Q_inv should be an inverse of Q after swapping cols."
                    );
                    debug_assert_eq!(
                        (&*Q_inv) * (&*Q),
                        Self::identity(Q.nof_cols()),
                        "Q_inv should be an inverse of Q after swapping cols."
                    );
                    debug_assert!(
                        *A.get(minor_idx, minor_idx)
                            .expect("The pivot should be in proper bounds.")
                            == amc % amm,
                        "After performing the reduction this should be the pivot."
                    );
                }
            }
        }
    }

    fn clear_col(A: &mut Self, P: &mut Self, P_inv: &mut Self, minor_idx: usize) {
        debug_assert!(
            *A.get(minor_idx, minor_idx)
                .expect("The indices are in proper bounds.")
                != R::ZERO,
            "The pivot in clear_col cannot be zero."
        );
        debug_assert_eq!(
            (&*P) * (&*P_inv),
            Self::identity(P.nof_cols()),
            "P_inv should be an inverse of P at the start of clear_col."
        );
        debug_assert_eq!(
            (&*P_inv) * (&*P),
            Self::identity(P.nof_cols()),
            "P_inv should be an inverse of P at the start of clear_col."
        );

        for row_idx in minor_idx.saturating_add(1)..A.nof_rows() {
            while *A
                .get(row_idx, minor_idx)
                .expect("The indices are in proper bounds.")
                != R::ZERO
            {
                let arm = *A
                    .get(row_idx, minor_idx)
                    .expect("The indices in A are in proper bounds.");
                let amm = *A
                    .get(minor_idx, minor_idx)
                    .expect("The indices in T are in proper bounds.");

                let q = arm / amm;

                A.add_muled_row_to_row(-q, minor_idx, row_idx)
                    .expect("Addition of row to row in A will succeed.");
                P.add_muled_row_to_row(-q, minor_idx, row_idx)
                    .expect("Addition of row to row in P will succeed.");
                P_inv
                    .add_muled_col_to_col(q, row_idx, minor_idx)
                    .expect("Addition of col to col in P_inv will succeed.");

                debug_assert_eq!(
                    (&*P) * (&*P_inv),
                    Self::identity(P.nof_cols()),
                    "P_inv should be an inverse of P after adding col to col."
                );
                debug_assert_eq!(
                    (&*P_inv) * (&*P),
                    Self::identity(P.nof_cols()),
                    "P_inv should be an inverse of P after adding col to col."
                );

                if *A
                    .get(row_idx, minor_idx)
                    .expect("The indices in A are in proper bounds.")
                    != R::ZERO
                {
                    A.swap_rows(minor_idx, row_idx)
                        .expect("Swapping cols in A will succeed.");
                    P.swap_rows(minor_idx, row_idx)
                        .expect("Swapping cols in P will succeed.");
                    P_inv
                        .swap_cols(minor_idx, row_idx)
                        .expect("Swapping cols in P_inv will succeed.");

                    debug_assert_eq!(
                        (&*P) * (&*P_inv),
                        Self::identity(P.nof_cols()),
                        "P_inv should be an inverse of P after swapping cols."
                    );
                    debug_assert_eq!(
                        (&*P_inv) * (&*P),
                        Self::identity(P.nof_cols()),
                        "P_inv should be an inverse of P after swapping cols."
                    );
                    debug_assert!(
                        *A.get(minor_idx, minor_idx)
                            .expect("The pivot should be in proper bounds.")
                            == arm % amm,
                        "After performing the reduction this should be the pivot."
                    );
                }
            }
        }
    }

    #[must_use]
    #[allow(
        clippy::missing_panics_doc,
        reason = "elementary operations and getters cannot panic since indices are in proper bounds"
    )]
    pub fn smith(self) -> (Self, Self, Self) {
        #[cfg(debug_assertions)]
        let A_old = self.clone();

        let mut A = self;

        let mut P = Self::identity(A.nof_rows());
        let mut P_inv = Self::identity(A.nof_rows());

        let mut Q = Self::identity(A.nof_cols());
        let mut Q_inv = Self::identity(A.nof_cols());

        'minor_idx: for minor_idx in 0..std::cmp::min(A.nof_rows(), A.nof_cols()) {
            debug_assert_eq!(
                (&P) * (&P_inv),
                Self::identity(A.nof_rows()),
                "The matrix P_inv should be the inverse of P."
            );
            debug_assert_eq!(
                (&Q) * (&Q_inv),
                Self::identity(A.nof_cols()),
                "The matrix Q_inv should be the inverse of Q."
            );
            'updating_minor: loop {
                if let Some((piv_row, piv_col)) = A.pivot(minor_idx) {
                    {
                        A.swap_rows(minor_idx, piv_row)
                            .expect("Indices of swapped rows of A are in proper bounds.");
                        P.swap_rows(minor_idx, piv_row)
                            .expect("Indices of swapped rows of P are in proper bounds.");
                        P_inv
                            .swap_cols(piv_row, minor_idx)
                            .expect("Indices of swapped cols of P_inv are in proper bounds.");

                        A.swap_cols(minor_idx, piv_col)
                            .expect("Indices of swapped cols of A are in proper bounds.");
                        Q.swap_cols(minor_idx, piv_col)
                            .expect("Indices of swapped cols of Q are in proper bounds.");
                        Q_inv
                            .swap_rows(minor_idx, piv_col)
                            .expect("Indices of swapped rows of Q_inv are in proper bounds.");
                    }
                    Self::clear_row(&mut A, &mut Q, &mut Q_inv, minor_idx);
                    Self::clear_col(&mut A, &mut P, &mut P_inv, minor_idx);

                    if let Some((bad_idx, _)) = A.non_divisible_entry(minor_idx) {
                        A.add_col_to_col(bad_idx, minor_idx)
                            .expect("The bad and the minor indices in A should be well-defined.");
                        Q.add_col_to_col(bad_idx, minor_idx)
                            .expect("The bad and the minor indices in Q should be well-defined.");
                        Q_inv
                            .add_muled_row_to_row(-R::ONE, minor_idx, bad_idx)
                            .expect("The bad and the minor indices in Q should be well-defined.");
                    } else {
                        let (_, to_canon, from_canon) = R::canonize(
                            *A.get(minor_idx, minor_idx)
                                .expect("The pivot in canonizing should be well-defined."),
                        );

                        debug_assert_eq!(
                            from_canon * to_canon,
                            R::ONE,
                            "Canonizing should be doune by invertible ring elements."
                        );

                        A.mul_col_by(minor_idx, to_canon)
                            .expect("Multiplying the col of A to canonize cannot fail.");
                        Q.mul_col_by(minor_idx, to_canon)
                            .expect("Multiplying the col of Q to canonize cannot fail.");
                        Q_inv
                            .mul_row_by(minor_idx, from_canon)
                            .expect("Multiplying the row of Q_inv to canonize cannot fail.");

                        break 'updating_minor;
                    }
                } else {
                    break 'minor_idx;
                }
            }
        }
        debug_assert!(
            A.is_in_smith_normal_form(),
            "The matrix should be in the smith normal form."
        );
        debug_assert!(
            (&P) * (&A_old) * (&Q) == A,
            "The matrices P and Q are not a desired change of basis in snf."
        );
        debug_assert_eq!(
            (&P) * (&P_inv),
            Self::identity(A.nof_rows()),
            "The matrix P_inv should be the inverse of P."
        );
        debug_assert_eq!(
            (&Q) * (&Q_inv),
            Self::identity(A.nof_cols()),
            "The matrix Q_inv should be the inverse of Q."
        );
        (P, A, Q)
    }

    #[must_use]
    #[allow(
        clippy::missing_panics_doc,
        reason = "curr and next indices are in proper bounds"
    )]
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
    use crate::ring::{Integer, Ring};
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
                .map(move |_| (0..nof_rows).map(move |_| rand::rng().random_range(1_i32..100_i32))),
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
        let A_old = matrix();
        let mut A = A_old.clone();

        let mut Q = M::identity(A.nof_cols());
        let mut Q_inv = M::identity(A.nof_cols());

        M::clear_row(&mut A, &mut Q, &mut Q_inv, 0);

        assert_eq!(A_old * (&Q), A);
        assert_eq!((&Q_inv) * (&Q), M::identity(A.nof_cols()));
        assert_eq!((&Q) * (&Q_inv), M::identity(A.nof_cols()));

        A.row(0)
            .expect("The 0-th row exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, Integer::ZERO));
    }

    #[test]
    fn clear_row_random() {
        let A_old = random_matrix(4, 3);
        let mut A = A_old.clone();

        let mut Q = M::identity(A.nof_cols());
        let mut Q_inv = M::identity(A.nof_cols());

        M::clear_row(&mut A, &mut Q, &mut Q_inv, 0);

        assert_eq!(A_old * (&Q), A);
        assert_eq!((&Q_inv) * (&Q), M::identity(A.nof_cols()));
        assert_eq!((&Q) * (&Q_inv), M::identity(A.nof_cols()));

        A.row(0)
            .expect("The 0-th row exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, Integer::ZERO));
    }

    #[test]
    fn clear_col() {
        let A_old = matrix();
        let mut A = A_old.clone();

        let mut P = M::identity(A.nof_rows());
        let mut P_inv = M::identity(A.nof_rows());

        M::clear_col(&mut A, &mut P, &mut P_inv, 0);

        assert_eq!((&P) * A_old, A);
        assert_eq!((&P_inv) * (&P), M::identity(A.nof_rows()));
        assert_eq!((&P) * (&P_inv), M::identity(A.nof_rows()));

        A.col(0)
            .expect("The 0-th col exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, Integer::ZERO));
    }

    #[test]
    fn clear_col_random() {
        let A_old = random_matrix(5, 6);
        let mut A = A_old.clone();

        let mut P = M::identity(A.nof_rows());
        let mut P_inv = M::identity(A.nof_rows());

        M::clear_col(&mut A, &mut P, &mut P_inv, 0);

        assert_eq!((&P) * A_old, A);
        assert_eq!((&P_inv) * (&P), M::identity(A.nof_rows()));
        assert_eq!((&P) * (&P_inv), M::identity(A.nof_rows()));

        A.col(0)
            .expect("The 0-th col exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, Integer::ZERO));
    }

    #[test]
    fn non_divisible_entry() {
        let mut A = matrix();

        let mut Q = M::identity(A.nof_cols());
        let mut Q_inv = M::identity(A.nof_cols());

        let mut P = M::identity(A.nof_rows());
        let mut P_inv = M::identity(A.nof_rows());

        M::clear_row(&mut A, &mut Q, &mut Q_inv, 0);
        M::clear_col(&mut A, &mut P, &mut P_inv, 0);

        assert_eq!(M::non_divisible_entry(&A, 0), None);
    }

    #[test]
    fn clear_row_2nd_time() {
        let A_old = matrix();
        let mut A = A_old.clone();

        let mut Q = M::identity(A.nof_cols());
        let mut Q_inv = M::identity(A.nof_cols());

        let mut P = M::identity(A.nof_rows());
        let mut P_inv = M::identity(A.nof_rows());

        M::clear_row(&mut A, &mut Q, &mut Q_inv, 0);
        M::clear_col(&mut A, &mut P, &mut P_inv, 0);
        M::clear_row(&mut A, &mut Q, &mut Q_inv, 1);

        assert_eq!((&P) * A_old * (&Q), A);
        assert_eq!((&Q_inv) * (&Q), M::identity(A.nof_cols()));
        assert_eq!((&Q) * (&Q_inv), M::identity(A.nof_cols()));
        assert_eq!((&P_inv) * (&P), M::identity(A.nof_rows()));
        assert_eq!((&P) * (&P_inv), M::identity(A.nof_rows()));

        A.row(0)
            .expect("The 0-th row exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, Integer::ZERO));

        A.col(0)
            .expect("The 0-th col exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, Integer::ZERO));

        A.row(1)
            .expect("The 1-th row exists.")
            .skip(2)
            .for_each(|row_entry| assert_eq!(*row_entry, Integer::ZERO));
    }

    #[test]
    fn clear_col_2nd_time() {
        let A_old = matrix();
        let mut A = A_old.clone();

        let mut Q = M::identity(A.nof_cols());
        let mut Q_inv = M::identity(A.nof_cols());

        let mut P = M::identity(A.nof_rows());
        let mut P_inv = M::identity(A.nof_rows());

        M::clear_row(&mut A, &mut Q, &mut Q_inv, 0);
        M::clear_col(&mut A, &mut P, &mut P_inv, 0);
        M::clear_col(&mut A, &mut P, &mut P_inv, 1);

        assert_eq!((&P) * A_old * (&Q), A);
        assert_eq!((&Q_inv) * (&Q), M::identity(A.nof_cols()));
        assert_eq!((&Q) * (&Q_inv), M::identity(A.nof_cols()));
        assert_eq!((&P_inv) * (&P), M::identity(A.nof_rows()));
        assert_eq!((&P) * (&P_inv), M::identity(A.nof_rows()));

        A.row(0)
            .expect("The 0-th row exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, Integer::ZERO));

        A.col(0)
            .expect("The 0-th col exists.")
            .skip(1)
            .for_each(|row_entry| assert_eq!(*row_entry, Integer::ZERO));

        A.col(1)
            .expect("The 1-th row exists.")
            .skip(2)
            .for_each(|row_entry| assert_eq!(*row_entry, Integer::ZERO));
    }

    #[test]
    fn smith() {
        let (P, D, Q) = matrix().smith();
        assert!(D.is_diagonal());
        assert_eq!(P * matrix() * Q, D);
    }

    #[test]
    fn smith_random() {
        'try_smith: loop {
            let A = random_matrix(2, 3);
            match std::panic::catch_unwind(|| A.clone().smith()) {
                Ok((P, D, Q)) => {
                    assert!(D.is_diagonal());
                    assert_eq!(P * A * Q, D);
                    break 'try_smith;
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
