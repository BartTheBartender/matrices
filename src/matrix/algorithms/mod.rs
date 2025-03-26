pub mod smith;
use super::*;
use crate::ring::Integer;

#[allow(
    non_snake_case,
    clippy::arithmetic_side_effects,
    reason = "this is a matrix algorithm"
)]
impl<R: Ring + From<Integer> + Into<Integer>> Matrix<R> {
    pub fn smith_from_integer(self) -> (Self, Self, Self, Self, Self) {
        let (D, P, P_inv, Q, Q_inv) = Matrix::<Integer>::from_vec_2d(self).smith();
        (
            Matrix::<R>::from_vec_2d(D),
            Matrix::<R>::from_vec_2d(P),
            Matrix::<R>::from_vec_2d(P_inv),
            Matrix::<R>::from_vec_2d(Q),
            Matrix::<R>::from_vec_2d(Q_inv),
        )
    }

    // / Returns a matrix `K` wich embeds the kernel into the domain.
    pub fn kernel(self) -> Self {
        let (D, _, _, Q, _) = self.smith_from_integer();
        let ker_D = D.kernel_diagonal();
        debug_assert!(
            (&D * &ker_D).is_zero(),
            "This should be a kernel of a diagonal matrix"
        );
        Q * ker_D
    }

    // / Returns a matrix `K` wich embeds the kernel into the domain, assuming `self` is diagonal.
    // Basically, collums of `K` form a basis of the kernel of `self`.
    pub fn kernel_diagonal(&self) -> Self {
        debug_assert!(self.is_diagonal(), "The matrix should be diagonal.");

        Self::from_cols(
            self.cols()
                .enumerate()
                .filter_map(|(idx, mut col)| col.all(|entry| *entry == R::ZERO).then_some(idx))
                .map(|idx| {
                    (0..self.row_len())
                        .map(move |col_idx| if col_idx == idx { R::ONE } else { R::ZERO })
                }),
        )
        .expect("The kernel of a diagonal matrix should be well-defined.")
    }

    /// Assumes that `A` is diagonal, counts non-zero collumns. Possibly optimize
    pub fn rank_diagonal(&self) -> usize {
        self.cols().fold(0, |count, mut col| {
            if col.any(|&entry| entry != R::ZERO) {
                count.saturating_add(1)
            } else {
                count
            }
        })
    }

    /// Computes rank of a not_diagonal matrix.
    pub fn rank(self) -> usize {
        let (D, _, _, _, _) = self.smith_from_integer();
        D.rank_diagonal()
    }

    pub fn is_mono(self) -> bool {
        self.row_len() <= self.col_len() && self.row_len() == self.rank()
    }

    pub fn is_epi(self) -> bool {
        self.col_len() <= self.row_len() && self.col_len() == self.rank()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type R = crate::ring::Integer;

    #[test]
    fn kernel_diagonal() {
        let diagonal_matrix = Matrix::<R>::from_rows_arr([[0, 0, 0], [0, 0, 0], [0, 0, 123]]);
        let kernel = diagonal_matrix.kernel_diagonal();
        assert_eq!(kernel, Matrix::<R>::from_cols_arr([[1, 0, 0], [0, 1, 0]]));
        assert!((diagonal_matrix * kernel).is_zero());
    }

    #[test]
    #[should_panic(expected = "The matrix should be diagonal.")]
    fn kernel_not_diagonal() {
        Matrix::<R>::from_rows_arr([[3, 2, 7], [8, 5, 1]]).kernel_diagonal();
    }

    #[test]
    fn kernel() {
        let matrix = Matrix::<R>::from_rows_arr([[1, -1, -1], [2, -2, 1]]);
        let kernel = matrix.clone().kernel();
        assert!((matrix * kernel).is_zero());
    }

    #[test]
    fn rank_diagonal() {
        let diagonal_matrix = Matrix::<R>::from_rows_arr([[0, 0, 0], [0, 0, 0], [0, 0, 123]]);
        assert_eq!(diagonal_matrix.rank_diagonal(), 1);
    }
}
