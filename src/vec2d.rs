use custom_error::custom_error;
use itertools::iproduct;
use std::{fmt, mem};

custom_error! {
    #[derive(PartialEq, Eq, Clone)]
    pub Vec2dError
    DifferentLengthsColsIterator = "Cols in the iterator had different lengths.",
    DifferentLengthsRowsIterator = "Rows in the iterator had different lengths.",
    EmptyColsIterator = "Iterator of cols was empty.",
    EmptyRowsIterator = "Iterator of rows was empty.",
    DifferentColLengths{left_col_len: usize, right_col_len: usize} = "Different col_lens (left.col_len() = {left_col_len}, right.col_len() = {right_col_len}).",
    DifferentRowLengths{top_row_len: usize, bot_row_len: usize} = "Different row_lens (top.row_len() = {top_row_len}, bot.row_len() = {bot_row_len}).",
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
                        return Err(Vec2dError::DifferentLengthsColsIterator);
                    }
                }
                None => nof_rows = Some(new_len),
            }
            old_len = new_len;
        }

        nof_rows
            .ok_or(Vec2dError::EmptyColsIterator)
            .map(|nof_rows| Self {
                buffer,
                nof_rows,
                nof_cols: new_len / nof_rows,
            })
    }

    pub fn from_cols_vec(cols_vec: Vec<Vec<T>>) -> Result<Self, Vec2dError> {
        Self::from_cols(cols_vec.into_iter().map(|col| col.into_iter()))
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
                Vec2dError::DifferentLengthsColsIterator => {
                    Vec2dError::DifferentLengthsRowsIterator
                }
                Vec2dError::EmptyColsIterator => Vec2dError::EmptyRowsIterator,
                _ => Vec2dError::Unexpected,
            })
    }

    pub fn from_rows_vec(rows_vec: Vec<Vec<T>>) -> Result<Self, Vec2dError> {
        Self::from_rows(rows_vec.into_iter().map(|row| row.into_iter()))
    }

    /// Given 2d vectors left and right, create the 2d vector
    /// ------------
    /// |left|right|
    /// ------------
    /// The caller must ensure that left and right have the same col_len
    fn merge_horizontally_unchecked(left: Self, mut right: Self) -> Self {
        let left_nof_cols = left.nof_cols();
        let mut buffer = left.buffer;
        buffer.append(&mut right.buffer);
        Self {
            buffer,
            nof_cols: left_nof_cols + right.nof_cols(),
            nof_rows: right.nof_rows(),
        }
    }

    /// Given 2d vectors left and right, if they have the same col_len's, create the 2d vector
    /// ------------
    /// |left|right|
    /// ------------
    pub fn merge_horizontally(left: Self, right: Self) -> Result<Self, Vec2dError> {
        let left_col_len = left.col_len();
        let right_col_len = right.col_len();
        (left_col_len == right_col_len)
            .then(|| Self::merge_horizontally_unchecked(left, right))
            .ok_or(Vec2dError::DifferentColLengths {
                left_col_len,
                right_col_len,
            })
    }

    /// Given 2d vectors top and bot, if they have the same row_len's, create the 2d vector
    /// -----
    /// |top|
    /// -----
    /// |bot|
    /// -----
    pub fn merge_vertically(mut top: Self, mut bot: Self) -> Result<Self, Vec2dError> {
        let top_row_len = top.row_len();
        let bot_row_len = bot.row_len();
        (top_row_len == bot_row_len)
            .then(|| {
                top.transpose();
                bot.transpose();
                let mut transposed = Self::merge_horizontally_unchecked(top, bot);

                transposed.transpose();
                transposed
            })
            .ok_or(Vec2dError::DifferentRowLengths {
                top_row_len,
                bot_row_len,
            })
    }

    /// Creates a 2d vector
    /// --------------------
    /// |left_top|right_top|
    /// --------------------
    /// |left_bot|right_bot|
    /// --------------------
    /// whenever it is possible.
    pub fn merge(
        top_left: Self,
        top_right: Self,
        bot_left: Self,
        bot_right: Self,
    ) -> Result<Self, Vec2dError> {
        let left = Self::merge_vertically(top_left, bot_left)?;
        let right = Self::merge_vertically(top_right, bot_right)?;
        Self::merge_horizontally(left, right)
    }

    /// Given a col_idx, splits
    /// ------   ------------
    /// |self| = |left|right|
    /// ------   ------------
    /// in such a way left.nof_cols() = col_idx (and right.nof_cols() = self.nof_cols() - col_idx).
    /// The caller must ensure that col_idx in 1..self.nof_cols()
    fn split_horizontally_unchecked(self, col_idx: usize) -> (Self, Self) {
        let nof_rows = self.nof_rows();
        let nof_cols = self.nof_cols();

        let mut left_buffer = self.buffer;
        let right_buffer = left_buffer.split_off(col_idx * self.nof_rows);
        let left = Self {
            buffer: left_buffer,
            nof_cols: col_idx,
            nof_rows,
        };

        let right = Self {
            buffer: right_buffer,
            nof_cols: nof_cols - col_idx,
            nof_rows,
        };

        (left, right)
    }

    pub fn split_horizontally(self, col_idx: usize) -> Result<(Self, Self), Vec2dError> {
        let col_len = self.col_len();
        (0 < col_idx && col_idx < self.nof_cols())
            .then(|| self.split_horizontally_unchecked(col_idx))
            .ok_or(Vec2dError::ColIdxOutOfBounds { col_idx, col_len })
    }

    pub fn split_vertically(mut self, row_idx: usize) -> Result<(Self, Self), Vec2dError> {
        let row_len = self.row_len();
        (0 < row_idx && row_idx < self.nof_rows())
            .then(|| {
                self.transpose();
                let (mut top, mut bot) = self.split_horizontally_unchecked(row_idx);
                top.transpose();
                bot.transpose();
                (top, bot)
            })
            .ok_or(Vec2dError::RowIdxOutOfBounds { row_idx, row_len })
    }

    pub fn split(
        self,
        col_idx: usize,
        row_idx: usize,
    ) -> Result<(Self, Self, Self, Self), Vec2dError> {
        let (left, right) = self.split_horizontally(col_idx)?;
        let (top_left, bot_left) = left.split_vertically(row_idx)?;
        let (top_right, bot_right) = right.split_vertically(row_idx)?;
        Ok((top_left, top_right, bot_left, bot_right))
    }
}

impl<T: fmt::Display> fmt::Display for Vec2d<T> {
    /// Displays the 2d vector in a format
    /// (self.nof_cols() x self.nof_rows())
    /// [*, *, *]
    /// [*, *, *]
    /// The values are padded for nicer format.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let elementwise_stringified = self
            .rows()
            .map(move |row| row.map(T::to_string).collect::<Vec<String>>())
            .collect::<Vec<_>>();

        let padding: usize = elementwise_stringified
            .iter()
            .flatten()
            .map(|entry| entry.len())
            .max()
            .unwrap_or(0);

        let stringified = elementwise_stringified
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|entry| format!("{}{}", " ".repeat(padding - entry.len()), entry))
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
    fn from_cols_vec() {
        let vec = vec![vec!['a', 'b', 'c'], vec!['d', 'e', 'f']];
        let vec2d = Vec2d::from_cols_vec(vec).expect("This should be well-defined");
        assert_eq!(vec2d.shape(), (3, 2));
        assert_eq!(vec2d.buffer, vec!['a', 'b', 'c', 'd', 'e', 'f']);
    }

    #[test]
    fn from_cols_empty_iterator() {
        let empty_vec: Vec<Vec<u8>> = vec![];
        let empty_vec2d = Vec2d::from_cols(empty_vec.into_iter().map(|col| col.into_iter()));
        assert_eq!(
            empty_vec2d.expect_err("This should be empty 2d vector"),
            Vec2dError::EmptyColsIterator
        );
    }

    #[test]
    fn from_cols_different_col_lens() {
        let different_col_lens_vec = vec![vec![1, 2, 3], vec![4, 5]];
        let different_col_lens_vec2d = Vec2d::from_cols(
            different_col_lens_vec
                .into_iter()
                .map(|col| col.into_iter()),
        );
        assert_eq!(
            different_col_lens_vec2d
                .expect_err("This should be non-empty but have different col lens."),
            Vec2dError::DifferentLengthsColsIterator
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
    fn from_rows_vec() {
        let vec = vec![vec!['a', 'b', 'c'], vec!['d', 'e', 'f']];
        let vec2d = Vec2d::from_rows_vec(vec).expect("This should be well-defined");
        assert_eq!(vec2d.shape(), (2, 3));
        assert_eq!(vec2d.buffer, vec!['a', 'd', 'b', 'e', 'c', 'f']);
    }

    #[test]
    fn from_rows_empty_iterator() {
        let empty_vec: Vec<Vec<u8>> = vec![];
        let empty_vec2d = Vec2d::from_rows(empty_vec.into_iter().map(|row| row.into_iter()));
        assert_eq!(
            empty_vec2d.expect_err("This should be empty 2d vector"),
            Vec2dError::EmptyRowsIterator
        );
    }

    #[test]
    fn from_rows_different_row_lens() {
        let different_row_lens_vec = vec![vec![1, 2, 3], vec![4, 5]];
        let different_row_lens_vec2d = Vec2d::from_rows(
            different_row_lens_vec
                .into_iter()
                .map(|row| row.into_iter()),
        );
        assert_eq!(
            different_row_lens_vec2d
                .expect_err("This should be non-empty but have different row lens."),
            Vec2dError::DifferentLengthsRowsIterator
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
    fn merge_horizontally() {
        let vec2d_1 = Vec2d::from_cols_vec(vec![vec![1, 2, 3], vec![4, 5, 6]])
            .expect("This should be well-defined.");
        let vec2d_2 =
            Vec2d::from_cols_vec(vec![vec![7, 8, 9]]).expect("This should be well-defined.");
        let vec2d =
            Vec2d::merge_horizontally(vec2d_1, vec2d_2).expect("The merge should be well-defined.");

        assert_eq!(vec2d.shape(), (3, 3));
        assert_eq!(vec2d.buffer, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn merge_horizontally_different_col_lens() {
        let u = Vec2d::from_cols_vec(vec![vec![1, 2, 3], vec![4, 5, 6]])
            .expect("This should be well-defined.");
        let v =
            Vec2d::from_cols_vec(vec![vec![7, 8, 9, 10]]).expect("This should be well-defined.");
        let vec = Vec2d::merge_horizontally(u.clone(), v.clone());
        assert_eq!(
            vec.expect_err("This should fail."),
            Vec2dError::DifferentColLengths {
                left_col_len: u.col_len(),
                right_col_len: v.col_len()
            }
        );
    }

    #[test]
    fn merge_vertically() {
        let u = Vec2d::from_rows_vec(vec![vec!['a', 'b', 'c'], vec!['d', 'e', 'f']])
            .expect("This should be well-defined.");
        let v =
            Vec2d::from_rows_vec(vec![vec!['g', 'h', 'i']]).expect("This should be well-defined.");

        let vec = Vec2d::merge_vertically(u, v).expect("The merge should be well-defined.");

        assert_eq!(vec.shape(), (3, 3));
        assert_eq!(
            vec.buffer,
            vec!['a', 'd', 'g', 'b', 'e', 'h', 'c', 'f', 'i']
        );
    }

    #[test]
    fn merge_vertically_different_row_lens() {
        let u = Vec2d::from_rows_vec(vec![vec!['a', 'b', 'c'], vec!['d', 'e', 'f']])
            .expect("This should be well-defined.");
        let v = Vec2d::from_rows_vec(vec![vec!['g', 'h', 'i', 'j']])
            .expect("This should be well-defined.");

        let vec = Vec2d::merge_vertically(u.clone(), v.clone());

        assert_eq!(
            vec.expect_err("This should fail."),
            Vec2dError::DifferentRowLengths {
                top_row_len: u.row_len(),
                bot_row_len: v.row_len()
            }
        )
    }

    #[test]
    fn merge() {
        let top_left =
            Vec2d::from_cols_vec(vec![vec![1, 2, 3]]).expect("This should be well-defined.");
        let top_right = Vec2d::from_cols_vec(vec![
            vec![5, 6, 7],
            vec![9, 10, 11],
            vec![13, 14, 15],
            vec![17, 18, 19],
        ])
        .expect("This should be well-defined.");
        let bot_left = Vec2d::from_cols_vec(vec![vec![4]]).expect("This should be well-defined.");
        let bot_right =
            Vec2d::from_rows_vec(vec![vec![8, 12, 16, 20]]).expect("This should be well-defined.");

        let vec = Vec2d::merge(top_left, top_right, bot_left, bot_right)
            .expect("The merge should be well-defined.");
        assert_eq!(vec.buffer, (1..=20).collect::<Vec<_>>());
    }

    #[test]
    fn split_horizontally() {
        let rows = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let vec = Vec2d::from_rows_vec(rows).expect("This should we well-defined.");
        let (left, right) = vec
            .split_horizontally(1)
            .expect("The split should be well-defined.");

        assert_eq!(left.shape(), (3, 1));
        assert_eq!(left.buffer, vec![1, 4, 7]);

        assert_eq!(right.shape(), (3, 2));
        assert_eq!(right.buffer, vec![2, 5, 8, 3, 6, 9]);
    }

    #[test]
    fn split_horizontally_idx_out_of_bounds() {
        let rows = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let vec = Vec2d::from_rows_vec(rows).expect("This should we well-defined.");

        let split_at_zero = vec.clone().split_horizontally(0);
        assert_eq!(
            split_at_zero.expect_err("This should fail."),
            Vec2dError::ColIdxOutOfBounds {
                col_idx: 0,
                col_len: vec.col_len()
            }
        );

        let split_at_col_len = vec.clone().split_horizontally(vec.col_len());
        assert_eq!(
            split_at_col_len.expect_err("This should fail."),
            Vec2dError::ColIdxOutOfBounds {
                col_idx: vec.col_len(),
                col_len: vec.col_len()
            }
        );
    }

    #[test]
    fn split_vertically() {
        let rows = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let vec = Vec2d::from_rows_vec(rows).expect("This should we well-defined.");

        let (top, bot) = vec
            .split_vertically(2)
            .expect("The split should be well-defined.");

        assert_eq!(top.shape(), (2, 3));
        assert_eq!(top.buffer, vec![1, 4, 2, 5, 3, 6]);

        assert_eq!(bot.shape(), (1, 3));
        assert_eq!(bot.buffer, vec![7, 8, 9]);
    }

    #[test]
    fn split_vertically_idx_out_of_bounds() {
        let rows = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let vec = Vec2d::from_rows_vec(rows).expect("This should we well-defined.");

        let split_at_zero = vec.clone().split_vertically(0);
        assert_eq!(
            split_at_zero.expect_err("This should fail."),
            Vec2dError::RowIdxOutOfBounds {
                row_idx: 0,
                row_len: vec.row_len()
            }
        );

        let split_at_row_len = vec.clone().split_vertically(vec.row_len());
        assert_eq!(
            split_at_row_len.expect_err("This should fail."),
            Vec2dError::RowIdxOutOfBounds {
                row_idx: vec.row_len(),
                row_len: vec.row_len()
            }
        );
    }

    #[test]
    fn split() {
        let vec = Vec2d::from_rows((1..5).map(|i| (1..6).map(move |j| i * j)))
            .expect("This should be well-defined.");
        println!("{}", vec);
        let (top_left, top_right, bot_left, bot_right) =
            vec.split(3, 3).expect("The split should be well-defined.");

        assert_eq!(top_left.shape(), (3, 3));
        assert_eq!(top_left.buffer, vec![1, 2, 3, 2, 4, 6, 3, 6, 9]);

        assert_eq!(top_right.shape(), (3, 2));
        assert_eq!(top_right.buffer, vec![4, 8, 12, 5, 10, 15]);

        assert_eq!(bot_left.shape(), (1, 3));
        assert_eq!(bot_left.buffer, vec![4, 8, 12]);

        assert_eq!(bot_right.shape(), (1, 2));
        assert_eq!(bot_right.buffer, vec![16, 20]);
    }

    #[test]
    #[ignore]
    fn swap_cols() {
        todo!()
    }

    #[test]
    #[ignore]
    fn swap_cols_idx_out_of_bounds() {
        todo!()
    }

    #[test]
    #[ignore]
    fn swap_rows() {
        todo!()
    }

    #[test]
    #[ignore]
    fn swap_rows_idx_out_of_bounds() {
        todo!()
    }
}
