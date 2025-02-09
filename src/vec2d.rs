use custom_error::custom_error;
use itertools::iproduct;
use std::{fmt, mem};

custom_error! {
    #[derive(PartialEq, Eq)]
    pub Vec2dError
    ColIteratorLen = "Cols in the iterator had different lengths.",
    RowIteratorLen = "Rows in the iterator had different lengths.",
    NoColsIterator = "Iterator of cols was empty.",
    NoRowsIterator = "Iterator of rows was empty.",
    DifferentColLens{u_col_len: usize, v_col_len: usize} = "Different col_lens (u.col_len() = {u_col_len}, v.col_len() = {v_col_len}).",
    DifferentRowLens{u_row_len: usize, v_row_len: usize} = "Different row_lens (u.row_len() = {u_row_len}, v.row_len() = {v_row_len}).",
    ColIdxOutOfBounds{col_idx: usize, col_len: usize} = "Col_idx ({col_idx}) out of bound col_len ({col_len}).",
    RowIdxOutOfBounds{row_idx: usize, row_len: usize} = "Row_idx ({row_idx}) out of bound row_len ({row_len}).",
    Unexpected = "This bug is unexpected"
}

/// Struct representing 2d vector, optimized for collumn operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vec2d<T> {
    buffer: Vec<T>,
    nof_rows: usize,
    nof_cols: usize,
}

impl<T> Vec2d<T> {
    pub const fn nof_rows(&self) -> usize {
        self.nof_rows
    }

    pub const fn nof_cols(&self) -> usize {
        self.nof_cols
    }

    pub const fn col_len(&self) -> usize {
        self.nof_rows()
    }

    pub const fn row_len(&self) -> usize {
        self.nof_cols()
    }

    pub const fn shape(&self) -> (usize, usize) {
        (self.nof_rows(), self.nof_cols())
    }

    /// Transposes the 2d vector. Unfortunately not in place.
    // Possibly optimize
    pub fn transpose(&mut self) {
        let nof_cols = self.nof_rows();
        let nof_rows = self.nof_cols();

        self.nof_cols = nof_cols;
        self.nof_rows = nof_rows;

        let mut helper_buffer = {
            let mut buffer: Vec<T> = Vec::with_capacity(self.buffer.len());
            unsafe { buffer.set_len(self.buffer.len()) }
            buffer
        };
        mem::swap(&mut helper_buffer, &mut self.buffer);

        helper_buffer
            .into_iter()
            .zip(iproduct!(0..nof_rows, 0..nof_cols))
            .map(|(value, (i, j))| (value, i + j * nof_rows))
            .for_each(|(value, idx)| self.buffer[idx] = value);
    }

    /// Collumn iterator, borrowing the 2d vec
    pub fn cols(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..self.nof_cols())
            .map(|col_idx| col_idx * self.col_len())
            .map(|col_idx| self.buffer[col_idx..col_idx + self.col_len()].iter())
    }

    //Collumn  iterator, taking ownership of the 2d vec
    pub fn into_cols(self) -> impl Iterator<Item = impl Iterator<Item = T>> {
        let nof_cols = self.nof_cols();
        let col_len = self.col_len();
        let mut buffer_iterator = self.buffer.into_iter();
        (0..nof_cols).map(move |_| {
            buffer_iterator
                .by_ref()
                .take(col_len)
                .collect::<Vec<_>>()
                .into_iter()
        })
    }

    pub fn from_cols(
        cols_iterator: impl Iterator<Item = impl Iterator<Item = T>>,
    ) -> Result<Self, Vec2dError> {
        let mut nof_rows = None;
        let mut buffer = Vec::new();

        let mut old_len = buffer.len();
        let mut new_len = buffer.len();

        for col_iterator in cols_iterator {
            buffer.extend(col_iterator);
            new_len = buffer.len();
            match nof_rows {
                Some(nof_rows) => {
                    if nof_rows != new_len - old_len {
                        return Err(Vec2dError::ColIteratorLen);
                    }
                }
                None => nof_rows = Some(new_len),
            }
            old_len = new_len;
        }

        nof_rows
            .ok_or(Vec2dError::NoColsIterator)
            .map(|nof_rows| Self {
                buffer,
                nof_rows,
                nof_cols: new_len / nof_rows,
            })
    }

    pub fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..self.nof_rows())
            .map(move |row_idx| {
                (0..self.nof_cols()).map(move |col_idx| col_idx * self.col_len() + row_idx)
            })
            .map(|curr_row_ids| curr_row_ids.map(|idx| &self.buffer[idx]))
    }

    pub fn into_rows(mut self) -> impl Iterator<Item = impl Iterator<Item = T>> {
        self.transpose();
        self.into_cols()
    }

    pub fn from_rows(
        rows_iterator: impl Iterator<Item = impl Iterator<Item = T>>,
    ) -> Result<Self, Vec2dError> {
        Self::from_cols(rows_iterator)
            .map(|mut self_transposed| {
                self_transposed.transpose();
                self_transposed
            })
            .map_err(|error| match error {
                Vec2dError::ColIteratorLen => Vec2dError::RowIteratorLen,
                Vec2dError::NoColsIterator => Vec2dError::NoRowsIterator,
                _ => Vec2dError::Unexpected,
            })
    }

    /// Merging horizontally and vertically.
    /// Given 2d vectors u and v, create a 2d vector [u|v].
    /// The caller must ensure that that the lengths of collumns are the same.
    fn merge_horizontally_unchecked(u: Self, mut v: Self) -> Self {
        let u_nof_cols = u.nof_cols();
        let mut buffer = u.buffer;
        buffer.append(&mut v.buffer);
        Self {
            buffer,
            nof_cols: u_nof_cols + v.nof_cols(),
            nof_rows: v.nof_rows(),
        }
    }

    /// Given 2d vectors u and v, create a 2d vector [u|v]
    pub fn merge_horizontally(u: Self, v: Self) -> Result<Self, Vec2dError> {
        let u_col_len = u.col_len();
        let v_col_len = v.col_len();
        if u_col_len == v_col_len {
            Ok(Self::merge_horizontally_unchecked(u, v))
        } else {
            Err(Vec2dError::DifferentColLens {
                u_col_len,
                v_col_len,
            })
        }
    }

    /// Given 2d vectors u and v, create a 2d vector [u^\u|v^T]^T
    pub fn merge_vertically(mut u: Self, mut v: Self) -> Result<Self, Vec2dError> {
        let u_row_len = u.row_len();
        let v_row_len = v.row_len();
        if u_row_len == v_row_len {
            u.transpose();
            v.transpose();
            let mut transposed = Self::merge_horizontally_unchecked(u, v);

            transposed.transpose();
            Ok(transposed)
        } else {
            Err(Vec2dError::DifferentRowLens {
                u_row_len,
                v_row_len,
            })
        }
    }
}

/// Returns vector of collumns
impl<T> From<Vec2d<T>> for Vec<Vec<T>> {
    fn from(vec2d: Vec2d<T>) -> Self {
        vec2d
            .into_cols()
            .map(|col_iterator| col_iterator.collect::<Vec<_>>())
            .collect::<Vec<_>>()
    }
}

impl<T: fmt::Display> fmt::Display for Vec2d<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let mut padding = 0;

        let elementwise_stringified = self.rows().map(move |row| {
            row.map(move |entry| entry.to_string())
                .inspect(move |entry| {
                    if entry.len() > padding {
                        padding = entry.len()
                    }
                })
        });

        let stringified = elementwise_stringified
            .map(|row| {
                row.map(|entry| format!("{:<width$}", entry, width = padding,))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .map(|row_stringified| format!("[{}]", row_stringified))
            .collect::<Vec<_>>()
            .join("\n");

        write!(
            f,
            "({} x {})\n{}",
            self.nof_rows(),
            self.nof_cols(),
            stringified
        )
    }
}

#[cfg(test)]
mod test {
    // In asserts use only internal data strucures.
    use super::*;

    #[test]
    fn from_cols() {
        let cols = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12],
        ]
        .into_iter()
        .map(Vec::into_iter);
        let vec2d = Vec2d::from_cols(cols).expect("This should be well-defined.");
        assert_eq!(vec2d.shape(), (3, 4));
        assert_eq!(vec2d.buffer, (1..=12).collect::<Vec<_>>());
    }

    #[test]
    fn from_cols_empty_iterator() {
        let empty_vec: Vec<Vec<u8>> = vec![];
        let empty_vec2d = Vec2d::from_cols(empty_vec.into_iter().map(|col| col.into_iter()));
        assert_eq!(
            empty_vec2d.expect_err("This should be empty 2d vector"),
            Vec2dError::NoColsIterator
        );
    }

    #[test]
    fn from_cols_wrong_col_lens() {
        let wrong_col_lens_vec = vec![vec![1, 2, 3], vec![4, 5]];
        let wrong_col_lens_vec2d =
            Vec2d::from_cols(wrong_col_lens_vec.into_iter().map(|col| col.into_iter()));
        assert_eq!(
            wrong_col_lens_vec2d.expect_err("This should be non-empty but have wrong col lens."),
            Vec2dError::ColIteratorLen
        );
    }

    #[test]
    fn cols() {
        let vec = vec![vec!['a', 'b', 'c'], vec!['d', 'e', 'f']];
        let vec2d = Vec2d::from_cols(vec.clone().into_iter().map(|col| col.into_iter()))
            .expect("This should be well-defined");
        assert_eq!(vec2d.shape(), (3, 2));
        assert_eq!(vec2d.buffer, vec!['a', 'b', 'c', 'd', 'e', 'f']);
    }

    #[test]
    fn into_cols() {
        let vec = vec![vec!['a', 'b', 'c'], vec!['d', 'e', 'f']];
        let vec2d = Vec2d::from_cols(vec.into_iter().map(|col| col.into_iter()))
            .expect("This should be well-defined");
        assert_eq!(vec2d.shape(), (3, 2));
        assert_eq!(vec2d.buffer, vec!['a', 'b', 'c', 'd', 'e', 'f']);
    }

    #[test]
    fn transpose_1() {
        let vec_t = vec![vec!['a', 'b', 'c'], vec!['d', 'e', 'f']];
        let mut vec2d = Vec2d::from_cols(vec_t.clone().into_iter().map(|col| col.into_iter()))
            .expect("This should be well-defined");
        vec2d.transpose();
        assert_eq!(vec2d.buffer, vec!['a', 'd', 'b', 'e', 'c', 'f']);
    }

    #[test]
    fn transpose_2() {
        let vec_t = vec![vec!['a', 'd'], vec!['b', 'e'], vec!['c', 'f']];
        let mut vec2d = Vec2d::from_cols(vec_t.clone().into_iter().map(|col| col.into_iter()))
            .expect("This should be well-defined");
        vec2d.transpose();
        assert_eq!(vec2d.buffer, vec!['a', 'b', 'c', 'd', 'e', 'f']);
    }

    #[test]
    fn from_rows() {
        let vec_t = vec![vec!["a", "b", "c"], vec!["d", "e", "f"]];
        let vec2d = Vec2d::from_rows(vec_t.into_iter().map(|row| row.into_iter()))
            .expect("This should be well-defined.");
        println!("{}", vec2d);
        assert_eq!(vec2d.shape(), (2, 3));
        assert_eq!(vec2d.buffer, vec!["a", "d", "b", "e", "c", "f"]);
    }

    #[test]
    #[ignore]
    fn from_rows_empty_iterator() {
        let empty_vec: Vec<Vec<u8>> = vec![];
        let empty_vec2d = Vec2d::from_rows(empty_vec.into_iter().map(|row| row.into_iter()));
        assert_eq!(
            empty_vec2d.expect_err("This should be empty 2d vector"),
            Vec2dError::NoRowsIterator
        );
    }

    #[test]
    fn from_rows_wrong_row_lens() {
        let wrong_row_lens_vec = vec![vec![1, 2, 3], vec![4, 5]];
        let wrong_row_lens_vec2d =
            Vec2d::from_rows(wrong_row_lens_vec.into_iter().map(|row| row.into_iter()));
        assert_eq!(
            wrong_row_lens_vec2d.expect_err("This should be non-empty but have wrong row lens."),
            Vec2dError::RowIteratorLen
        );
    }

    #[test]
    fn rows() {
        let vec = vec![vec!['a', 'b', 'c'], vec!['d', 'e', 'f']];
        let vec2d = Vec2d::from_rows(vec.clone().into_iter().map(|row| row.into_iter()))
            .expect("This should be well-defined");
        assert_eq!(vec2d.shape(), (2, 3));
        assert_eq!(vec2d.buffer, vec!['a', 'd', 'b', 'e', 'c', 'f']);
    }

    #[test]
    fn into_rows() {
        let vec = vec![vec!['a', 'b', 'c'], vec!['d', 'e', 'f']];
        let vec2d = Vec2d::from_rows(vec.into_iter().map(|row| row.into_iter()))
            .expect("This should be well-defined");
        assert_eq!(vec2d.shape(), (2, 3));
        assert_eq!(vec2d.buffer, vec!['a', 'd', 'b', 'e', 'c', 'f']);
    }

    #[test]
    #[ignore]
    fn merge_horizontally() {
        todo!()
    }

    #[test]
    #[ignore]
    fn merge_horizontally_wrong_col_lens() {
        todo!()
    }

    #[test]
    #[ignore]
    fn merge_vertically() {
        todo!()
    }

    #[test]
    #[ignore]
    fn merge_vertically_wrong_row_lens() {
        todo!()
    }

    #[test]
    #[ignore]
    fn split_horizontally() {
        todo!()
    }

    #[test]
    #[ignore]
    fn split_horizontally_idx_out_of_bounds() {
        todo!()
    }

    #[test]
    #[ignore]
    fn split_vertically() {
        todo!()
    }

    #[test]
    #[ignore]
    fn split_vertically_idx_out_of_bounds() {
        todo!()
    }

    #[test]
    #[ignore]
    fn merge() {
        todo!()
    }

    #[test]
    #[ignore]
    fn split() {
        todo!()
    }

    #[test]
    #[ignore]
    fn into_vec_vec_t() {
        todo!()
    }

    #[test]
    #[ignore]
    fn from_vec_vec_t() {
        todo!()
    }
}
