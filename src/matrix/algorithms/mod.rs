use super::*;
use crate::ring::Bezout;

impl<R: Bezout + std::fmt::Debug + std::fmt::Display> Matrix<R> {
    /// Returns the index of a pivot in the smith normal form algorithm
    /// in the right-bottom submatrix of self
    fn pivot(&self, row_beg: usize, col_beg: usize) -> Option<(usize, usize)> {
        self.buffer
            .iter()
            .zip(itertools::iproduct!(0..self.nof_rows(), 0..self.nof_cols()))
            .filter(|(_, (row, col))| row_beg <= *row && col_beg <= *col)
            .find(|(el, _)| **el != R::zero())
            .map(|(_, idx)| idx)
    }

    /// Assuming the matrix is in the form
    /// // ----------
    /// // |diag| 0 |
    /// // ----------
    /// // |  0 | A |
    /// using elementary operations make the first row of A zeroes,
    /// except the first entry, without updating other blocks
    fn reduce_row(&mut self, pivot: usize) {
        let a_pp = *self
            .get(pivot, pivot)
            .expect("This entry should be well-defined");

        for col in pivot + 1..self.row_len() {
            let a_pc = *self
                .get(pivot, col)
                .expect("This entry should be well-defined");

            let (g, x, y) = R::gcd(a_pp, a_pc);
            let q_pp = R::try_divide(a_pp, g).expect("g is a divisor of a_pp.");
            let q_pc = R::try_divide(a_pc, g).expect("g is a divisor of a_rp.");

            let mut t = Matrix::<R>::identity(self.nof_cols());
            *t.get_mut(pivot, pivot)
                .expect("This entry should be well-defined.") = x;
            *t.get_mut(pivot, col)
                .expect("This entry should be well-defined.") = -q_pc;
            *t.get_mut(col, pivot)
                .expect("This entry should be well-defined.") = y;
            *t.get_mut(col, col)
                .expect("This entry should be well-defined.") = q_pp;
            *self = &*self * t;
        }
    }

    /// Assuming the matrix is in the form
    /// // ----------
    /// // |diag| 0 |
    /// // ----------
    /// // |  0 | A |
    /// using elementary operations make the first col of A zeroes,
    /// except the first entry, without updating other blocks
    fn reduce_col(&mut self, pivot: usize) {
        println!("{}", self);
        let a_pp = *self
            .get(pivot, pivot)
            .expect("This entry should be well-defined");

        for row in pivot + 1..self.col_len() {
            let a_rp = *self
                .get(row, pivot)
                .expect("This entry should be well-defined");
            println!("a[{},{}]={}", row, pivot, a_rp);

            let (g, x, y) = R::gcd(a_pp, a_rp);
            let q_pp = R::try_divide(a_pp, g).expect("g is a divisor of a_pp.");
            let q_rp = R::try_divide(a_rp, g).expect("g is a divisor of a_rp.");

            let mut s = Matrix::<R>::identity(self.nof_rows());
            *s.get_mut(pivot, pivot)
                .expect("This entry should be well-defined.") = x;
            *s.get_mut(row, pivot)
                .expect("This entry should be well-defined.") = -q_rp;
            *s.get_mut(pivot, row)
                .expect("This entry should be well-defined.") = y;
            *s.get_mut(row, row)
                .expect("This entry should be well-defined.") = q_pp;
            *self = s * (&*self);
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::ring::AbelianGroup;
    type R = crate::ring::Integer;

    #[test]
    fn reduce_row_1() {
        let mut matrix = Matrix::<R>::from_rows_vec(vec![
            vec![1, 2, 3],
            vec![5, 7, 1],
            vec![0, 5, 6],
            vec![12, 6, 7],
        ])
        .expect("This should be well-defined.");

        matrix.reduce_row(0);

        matrix
            .row(0)
            .expect("Row index is in proper bounds")
            .enumerate()
            .filter(|(idx, _)| *idx != 0)
            .map(|(_, el)| el)
            .copied()
            .for_each(|el| assert_eq!(el, R::zero()));
    }

    #[test]
    fn reduce_row_2() {
        let mut matrix = Matrix::<R>::from_rows_vec(vec![
            vec![1, 0, 0],
            vec![0, 7, 1],
            vec![0, 5, 6],
            vec![0, 6, 7],
        ])
        .expect("This should be well-defined.");

        matrix.reduce_row(1);

        matrix
            .row(1)
            .expect("Row index is in proper bounds")
            .enumerate()
            .filter(|(idx, _)| *idx != 1)
            .map(|(_, el)| el)
            .copied()
            .for_each(|el| assert_eq!(el, R::zero()));
    }

    #[test]
    fn reduce_col_1() {
        let mut matrix = Matrix::<R>::from_cols_vec(vec![
            vec![1, 2, 3],
            vec![5, 7, 1],
            vec![0, 5, 6],
            vec![12, 6, 7],
        ])
        .expect("This should be well-defined.");

        matrix.reduce_col(0);
        println!("{}", matrix);

        matrix
            .col(0)
            .expect("Row index is in proper bounds")
            .enumerate()
            .filter(|(idx, _)| *idx != 0)
            .map(|(_, el)| el)
            .copied()
            .for_each(|el| assert_eq!(el, R::zero()));
    }

    #[test]
    fn reduce_col_2() {
        let mut matrix = Matrix::<R>::from_cols_vec(vec![
            vec![1, 0, 0],
            vec![0, 7, 1],
            vec![0, 5, 6],
            vec![0, 6, 7],
        ])
        .expect("This should be well-defined.");

        matrix.reduce_col(1);

        matrix
            .col(1)
            .expect("Row index is in proper bounds")
            .enumerate()
            .filter(|(idx, _)| *idx != 1)
            .map(|(_, el)| el)
            .copied()
            .for_each(|el| assert_eq!(el, R::zero()));
    }
}
