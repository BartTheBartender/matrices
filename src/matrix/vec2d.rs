use crate::ring::Finite;
use custom_error::custom_error;
use itertools::{iproduct, Itertools};
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Vec2d<T> {
    pub(super) buffer: Vec<T>,
    pub(super) nof_rows: usize,
    pub(super) nof_cols: usize,
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

    pub const fn is_square(&self) -> bool {
        self.nof_cols() == self.nof_rows()
    }

    pub unsafe fn get_unchecked(&self, i: usize, j: usize) -> &T {
        self.buffer.get_unchecked(i + self.col_len() * j)
    }

    pub fn get(&self, i: usize, j: usize) -> Result<&T, Vec2dError> {
        if i >= self.nof_rows() {
            Err(Vec2dError::ColIdxOutOfBounds {
                col_idx: i,
                col_len: self.nof_rows(),
            })
        } else if j >= self.nof_cols() {
            Err(Vec2dError::RowIdxOutOfBounds {
                row_idx: j,
                row_len: self.nof_cols(),
            })
        } else {
            Ok(unsafe { self.get_unchecked(i, j) })
        }
    }

    pub unsafe fn get_unchecked_mut<'a>(&'a mut self, i: usize, j: usize) -> &'a mut T {
        let col_len = self.col_len();
        self.buffer.get_unchecked_mut(i + col_len * j)
    }

    pub fn get_mut<'a>(&'a mut self, i: usize, j: usize) -> Result<&'a mut T, Vec2dError> {
        if i >= self.nof_rows() {
            Err(Vec2dError::ColIdxOutOfBounds {
                col_idx: i,
                col_len: self.nof_rows(),
            })
        } else if j >= self.nof_cols() {
            Err(Vec2dError::RowIdxOutOfBounds {
                row_idx: j,
                row_len: self.nof_cols(),
            })
        } else {
            Ok(unsafe { self.get_unchecked_mut(i, j) })
        }
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

    /// The caller must guaranttee that col_idx < self.nof_cols()
    pub unsafe fn col_unchecked(&self, col_idx: usize) -> impl Iterator<Item = &T> {
        let begin = col_idx * self.col_len();
        let end = begin + self.col_len();
        self.buffer[begin..end].iter()
    }

    /// The caller must guaranttee that col_idx < self.nof_cols()
    pub unsafe fn col_mut_unchecked(&mut self, col_idx: usize) -> impl Iterator<Item = &mut T> {
        let begin = col_idx * self.col_len();
        let end = begin + self.col_len();
        self.buffer[begin..end].iter_mut()
    }

    pub fn col(&self, col_idx: usize) -> Result<impl Iterator<Item = &T>, Vec2dError> {
        (col_idx < self.nof_cols())
            .then(|| unsafe { self.col_unchecked(col_idx) })
            .ok_or(Vec2dError::RowIdxOutOfBounds {
                row_idx: col_idx,
                row_len: self.nof_cols(),
            })
    }

    pub fn col_mut(&mut self, col_idx: usize) -> Result<impl Iterator<Item = &mut T>, Vec2dError> {
        let nof_cols = self.nof_cols();
        (col_idx < nof_cols)
            .then(|| unsafe { self.col_mut_unchecked(col_idx) })
            .ok_or(Vec2dError::RowIdxOutOfBounds {
                row_idx: col_idx,
                row_len: nof_cols,
            })
    }

    /// Collumn iterator, borrowing the 2d vec
    pub fn cols(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..self.nof_cols()).map(|col_idx| unsafe { self.col_unchecked(col_idx) })
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

    pub fn into_cols_vec(self) -> Vec<Vec<T>> {
        self.into_cols()
            .map(|col_iterator| col_iterator.collect::<Vec<_>>())
            .collect::<Vec<_>>()
    }

    pub fn from_cols_arr<const NOF_ROWS: usize, const NOF_COLS: usize>(
        cols_arr: [[T; NOF_ROWS]; NOF_COLS],
    ) -> Self {
        Self::from_cols(cols_arr.into_iter().map(|col| col.into_iter()))
            .expect("The arrays have correct bounds.")
    }

    /// The caller must guaranttee that row_idx < self.nof_rows()
    pub unsafe fn row_unchecked(&self, row_idx: usize) -> impl Iterator<Item = &T> {
        (0..self.row_len())
            .map(move |idx| self.col_len() * idx + row_idx)
            .map(|idx| self.buffer.get_unchecked(idx))
    }

    /// The caller must guaranttee that row_idx < self.nof_rows()
    pub unsafe fn row_mut_unchecked(&mut self, row_idx: usize) -> impl Iterator<Item = &mut T> {
        let row_len = self.row_len();
        let col_len = self.col_len();

        let buffer_ptr = self.buffer.as_mut_ptr();

        (0..row_len)
            .map(move |idx| col_len * idx + row_idx)
            .map(move |idx| &mut *buffer_ptr.add(idx))
    }

    pub fn row(&self, row_idx: usize) -> Result<impl Iterator<Item = &T>, Vec2dError> {
        (row_idx < self.nof_rows())
            .then(|| unsafe { self.row_unchecked(row_idx) })
            .ok_or(Vec2dError::ColIdxOutOfBounds {
                col_idx: row_idx,
                col_len: self.nof_rows(),
            })
    }

    pub fn row_mut(&mut self, row_idx: usize) -> Result<impl Iterator<Item = &mut T>, Vec2dError> {
        let nof_rows = self.nof_rows();
        (row_idx < nof_rows)
            .then(|| unsafe { self.row_mut_unchecked(row_idx) })
            .ok_or(Vec2dError::ColIdxOutOfBounds {
                col_idx: row_idx,
                col_len: nof_rows,
            })
    }

    pub fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..self.nof_rows()).map(move |row_idx| unsafe { self.row_unchecked(row_idx) })
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

    pub fn into_rows_vec(self) -> Vec<Vec<T>> {
        self.into_rows()
            .map(|row_iterator| row_iterator.collect::<Vec<_>>())
            .collect::<Vec<_>>()
    }

    pub fn from_rows_arr<const NOF_ROWS: usize, const NOF_COLS: usize>(
        rows_arr: [[T; NOF_COLS]; NOF_ROWS],
    ) -> Self {
        Self::from_rows(rows_arr.into_iter().map(|row| row.into_iter()))
            .expect("The arrays have correct bounds.")
    }

    /// Given 2d vectors left and right, create the 2d vector
    /// ------------
    /// |left|right|
    /// ------------
    /// The caller must ensure that left and right have the same col_len
    pub unsafe fn merge_horizontally_unchecked(left: Self, mut right: Self) -> Self {
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
            .then(|| unsafe { Self::merge_horizontally_unchecked(left, right) })
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
                let mut transposed = unsafe { Self::merge_horizontally_unchecked(top, bot) };

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
    /// in such a way ``left.nof_cols() = col_idx`` and ``right.nof_cols() = self.nof_cols() - col_idx``.
    /// # Safety: the caller must ensure that ``col_idx`` in ``1..self.nof_cols()``
    #[must_use]
    pub unsafe fn split_horizontally_unchecked(self, col_idx: usize) -> (Self, Self) {
        let nof_rows = self.nof_rows();
        let nof_cols = self.nof_cols();

        let mut left_buffer = self.buffer;
        let right_buffer = left_buffer.split_off(col_idx.saturating_mul(self.nof_rows));
        let left = Self {
            buffer: left_buffer,
            nof_cols: col_idx,
            nof_rows,
        };

        let right = Self {
            buffer: right_buffer,
            nof_cols: nof_cols.saturating_sub(col_idx),
            nof_rows,
        };

        (left, right)
    }

    /// Given a col_idx, splits
    /// ------   ------------
    /// |self| = |left|right|
    /// ------   ------------
    /// in such a way left.nof_cols() = col_idx (and right.nof_cols() = self.nof_cols() - col_idx).
    /// Otherwise returns error
    /// # Errors
    pub fn split_horizontally(self, col_idx: usize) -> Result<(Self, Self), Vec2dError> {
        let col_len = self.col_len();
        (0 < col_idx && col_idx < self.nof_cols())
            .then(|| unsafe { self.split_horizontally_unchecked(col_idx) })
            .ok_or(Vec2dError::ColIdxOutOfBounds { col_idx, col_len })
    }

    /// Given a row_idx, splits
    /// ------   -----
    /// |self| = |top|
    /// ------   -----
    /// ///      |bot|
    /// ///      -----
    /// in such a way top.nof_rows() = row_idx (and bot.nof_rows() = self.nof_rows() - rows_idx).
    /// Otherwise returns error.
    /// # Errors
    pub fn split_vertically(mut self, row_idx: usize) -> Result<(Self, Self), Vec2dError> {
        let row_len = self.row_len();
        (0 < row_idx && row_idx < self.nof_rows())
            .then(|| {
                self.transpose();
                let (mut top, mut bot) = unsafe { self.split_horizontally_unchecked(row_idx) };
                top.transpose();
                bot.transpose();
                (top, bot)
            })
            .ok_or(Vec2dError::RowIdxOutOfBounds { row_idx, row_len })
    }

    /// Given a col_idx and row_idx, splits
    /// ------   --------------------
    /// |self| = |top_left|top_right|
    /// ------   --------------------
    /// ///      |bot_left|bot_right|
    /// ///      --------------------
    /// in such a way top_left.shape() = (col_idx, row_idx)
    /// whenever it is possible.
    /// # Errors
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

    /// Modifies a 2d vector of a form
    /// -----------
    /// |*|i|*|j|*|
    /// -----------
    /// to
    /// -----------
    /// |*|j|*|i|*|
    /// -----------
    /// whenever indices are in proper bounds.
    /// # Safety
    /// Note that
    /// i * col_len + idx <= (nof_cols - 1) * col_len + col_len - 1 = nof_cols * col_len - 1
    /// and similarly for j.
    /// # Errors
    pub fn swap_cols(&mut self, i: usize, j: usize) -> Result<(), Vec2dError> {
        let nof_cols = self.nof_cols();
        let col_len = self.col_len();

        if i >= nof_cols {
            Err(Vec2dError::RowIdxOutOfBounds {
                row_idx: i,
                row_len: nof_cols,
            })
        } else if j >= nof_cols {
            Err(Vec2dError::RowIdxOutOfBounds {
                row_idx: j,
                row_len: nof_cols,
            })
        } else {
            if i != j {
                (0..col_len)
                    .map(|idx| {
                        (
                            i.saturating_mul(col_len).saturating_add(idx),
                            j.saturating_mul(col_len).saturating_add(idx),
                        )
                    })
                    .for_each(|(col_i_idx, col_j_idx)| unsafe {
                        self.buffer.swap_unchecked(col_i_idx, col_j_idx);
                    });
            }
            Ok(())
        }
    }

    /// Modifies a 2d vector of a form
    /// ---
    /// |*|
    /// ---
    /// |i|
    /// ---
    /// |*|
    /// ---
    /// |j|
    /// ---
    /// |*|
    /// ---
    /// to
    /// ---
    /// |*|
    /// ---
    /// |j|
    /// ---
    /// |*|
    /// ---
    /// |i|
    /// ---
    /// |*|
    /// ---
    /// whenever indices are in proper bounds.
    /// # Safety
    /// Note that
    /// i * row_len + idx <= (nof_rows - 1) * row_len + row_len - 1 = nof_rows * row_len - 1
    /// and similarly for j.
    /// # Errors
    pub fn swap_rows(&mut self, i: usize, j: usize) -> Result<(), Vec2dError> {
        let nof_rows = self.nof_rows();
        let row_len = self.row_len();

        if i >= nof_rows {
            Err(Vec2dError::ColIdxOutOfBounds {
                col_idx: i,
                col_len: nof_rows,
            })
        } else if j >= nof_rows {
            Err(Vec2dError::ColIdxOutOfBounds {
                col_idx: j,
                col_len: nof_rows,
            })
        } else {
            if i != j {
                (0..row_len)
                    .map(|idx| {
                        (
                            i.saturating_add(nof_rows.saturating_mul(idx)),
                            j.saturating_add(nof_rows.saturating_mul(idx)),
                        )
                    })
                    .for_each(|(row_i_idx, row_j_idx)| unsafe {
                        self.buffer.swap_unchecked(row_i_idx, row_j_idx);
                    });
            }
            Ok(())
        }
    }
    #[must_use]
    pub fn from_vec_2d<U: Into<T>>(vec2d: Vec2d<U>) -> Self {
        let nof_cols = vec2d.nof_cols();
        let nof_rows = vec2d.nof_rows();

        Self {
            nof_cols,
            nof_rows,
            buffer: vec2d
                .buffer
                .into_iter()
                .map(|value| std::convert::Into::into(value))
                .collect::<Vec<_>>(),
        }
    }

    #[must_use]
    pub fn into_vec_2d<U: From<T>>(self) -> Vec2d<U> {
        let nof_cols = self.nof_cols();
        let nof_rows = self.nof_rows();

        Vec2d::<U> {
            nof_cols,
            nof_rows,
            buffer: self
                .buffer
                .into_iter()
                .map(|value| (value).into())
                .collect::<Vec<_>>(),
        }
    }
}

impl<T: fmt::Display> fmt::Display for Vec2d<T> {
    /// Displays the 2d vector in a format
    /// (self.nof_cols() x self.nof_rows())
    /// [*, *, *]
    /// [*, *, *]
    /// The values are padded.
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

impl<T: Finite> Vec2d<T> {
    pub fn elements(
        nof_cols: usize,
        nof_rows: usize,
    ) -> impl Iterator<Item = Vec2d<<T as crate::ring::Finite>::Output>> {
        let elements = T::elements().collect::<Vec<_>>();
        (0..nof_cols * nof_rows)
            .map(|_| (0..elements.len()))
            .multi_cartesian_product()
            .map(
                move |buffer_indices: Vec<usize>| Vec2d::<<T as crate::ring::Finite>::Output> {
                    nof_cols,
                    nof_rows,
                    buffer: buffer_indices
                        .into_iter()
                        .map(|idx| elements[idx])
                        .collect::<Vec<_>>(),
                },
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
    fn col() {
        let vec = Vec2d::from_cols_vec(vec![vec![1, 1], vec![2, 2], vec![3, 3], vec![0, 0]])
            .expect("This should be well-defined.");
        assert_eq!(
            vec.col(0)
                .expect("This should be well-defined.")
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert_eq!(
            vec.col(3)
                .expect("This should be well-defined.")
                .copied()
                .collect::<Vec<_>>(),
            vec![0, 0]
        );
        assert_eq!(
            unsafe { vec.col(7).unwrap_err_unchecked() },
            Vec2dError::RowIdxOutOfBounds {
                row_idx: 7,
                row_len: vec.row_len()
            }
        );
    }

    #[test]
    fn col_mut() {
        let mut vec = Vec2d::from_cols_vec(vec![vec![1, 1], vec![2, 2], vec![3, 3], vec![4, 4]])
            .expect("This should be well-defined.");
        let mut vec1 = vec.clone();

        vec.col_mut(0)
            .expect("This should be well-defined.")
            .for_each(|x| (*x) *= 10);
        assert_eq!(
            vec.into_cols_vec(),
            vec![vec![10, 10], vec![2, 2], vec![3, 3], vec![4, 4]]
        );

        assert_eq!(
            unsafe { vec1.col_mut(8).unwrap_err_unchecked() },
            Vec2dError::RowIdxOutOfBounds {
                row_idx: 8,
                row_len: 4
            }
        );
    }

    #[test]
    fn from_cols_arr() {
        let vec = Vec2d::from_cols_arr([[1, 2, 3, 3], [4, 5, 6, 6], [7, 8, 9, 9]]);
        assert_eq!(vec.buffer, vec![1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9,]);
    }

    #[test]
    fn transpose_1() {
        let vec = Vec2d::from_cols_arr([['a', 'b', 'c'], ['d', 'e', 'f']]);
        let mut vec_t = vec.clone();
        vec_t.transpose();
        assert_eq!(vec_t.buffer, vec!['a', 'd', 'b', 'e', 'c', 'f']);
        assert_eq!(vec.nof_rows(), vec_t.nof_cols());
        assert_eq!(vec.nof_cols(), vec_t.nof_rows());
    }

    #[test]
    fn transpose_2() {
        let vec = Vec2d::from_cols_arr([['a', 'd'], ['b', 'e'], ['c', 'f']]);
        let mut vec_t = vec.clone();
        vec_t.transpose();
        assert_eq!(vec_t.buffer, vec!['a', 'b', 'c', 'd', 'e', 'f']);
        assert_eq!(vec.nof_rows(), vec_t.nof_cols());
        assert_eq!(vec.nof_cols(), vec_t.nof_rows());
    }

    #[test]
    fn from_rows() {
        let vec_t = [["a", "b", "c"], ["d", "e", "f"]];
        let vec2d = Vec2d::from_rows(vec_t.into_iter().map(|row| row.into_iter()))
            .expect("This should be well-defined.");
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
    fn from_rows_arr() {
        let vec = Vec2d::from_rows_arr([[1, 2, 3, 3], [4, 5, 6, 6], [7, 8, 9, 9]]);
        assert_eq!(vec.buffer, vec![1, 4, 7, 2, 5, 8, 3, 6, 9, 3, 6, 9]);
    }

    #[test]
    fn row() {
        let vec = Vec2d::from_rows_vec(vec![vec![1, 1], vec![2, 2], vec![3, 3], vec![0, 0]])
            .expect("This should be well-defined.");
        assert_eq!(
            vec.row(0)
                .expect("This should be well-defined.")
                .copied()
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert_eq!(
            vec.row(3)
                .expect("This should be well-defined.")
                .copied()
                .collect::<Vec<_>>(),
            vec![0, 0]
        );
        assert_eq!(
            unsafe { vec.row(7).unwrap_err_unchecked() },
            Vec2dError::ColIdxOutOfBounds {
                col_idx: 7,
                col_len: vec.col_len()
            }
        );
    }

    #[test]
    fn row_mut() {
        let mut vec = Vec2d::from_rows_vec(vec![vec![1, 1], vec![2, 2], vec![3, 3], vec![4, 4]])
            .expect("This should be well-defined.");
        let mut vec1 = vec.clone();

        vec.row_mut(0)
            .expect("This should be well-defined.")
            .for_each(|x| (*x) *= 10);
        assert_eq!(
            vec.into_rows_vec(),
            vec![vec![10, 10], vec![2, 2], vec![3, 3], vec![4, 4]]
        );

        assert_eq!(
            unsafe { vec1.row_mut(8).unwrap_err_unchecked() },
            Vec2dError::ColIdxOutOfBounds {
                col_idx: 8,
                col_len: 4
            }
        );
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
            vec.expect_err("The merge should fail."),
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
            vec.expect_err("The merge should fail."),
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
            split_at_zero.expect_err("The split should fail."),
            Vec2dError::ColIdxOutOfBounds {
                col_idx: 0,
                col_len: vec.col_len()
            }
        );

        let split_at_col_len = vec.clone().split_horizontally(vec.col_len());
        assert_eq!(
            split_at_col_len.expect_err("The split should fail."),
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
            split_at_zero.expect_err("The split should fail."),
            Vec2dError::RowIdxOutOfBounds {
                row_idx: 0,
                row_len: vec.row_len()
            }
        );

        let split_at_row_len = vec.clone().split_vertically(vec.row_len());
        assert_eq!(
            split_at_row_len.expect_err("The split should fail."),
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
    fn swap_cols() {
        let mut vec = Vec2d::from_cols_vec(vec![vec![0, 0], vec![1, 1], vec![2, 2], vec![3, 3]])
            .expect("This should be well-defined.");
        vec.swap_cols(0, 2)
            .expect("The swap should be well-defined.");
        assert_eq!(vec.buffer, vec![2, 2, 1, 1, 0, 0, 3, 3]);
    }

    #[test]
    fn swap_cols_idx_out_of_bounds() {
        let vec = Vec2d::from_cols_vec(vec![vec![0, 0], vec![1, 1], vec![2, 2], vec![3, 3]])
            .expect("This should be well-defined.");

        assert_eq!(
            vec.clone()
                .swap_cols(0, 5)
                .expect_err("The swap should fail."),
            Vec2dError::RowIdxOutOfBounds {
                row_idx: 5,
                row_len: vec.row_len()
            }
        );

        assert_eq!(
            vec.clone()
                .swap_cols(5, 3)
                .expect_err("The swap should fail."),
            Vec2dError::RowIdxOutOfBounds {
                row_idx: 5,
                row_len: vec.row_len()
            }
        );
    }

    #[test]
    fn swap_rows() {
        let mut vec = Vec2d::from_rows_vec(vec![vec![0, 0], vec![1, 1], vec![2, 2]])
            .expect("This should be well-defined.");
        vec.swap_rows(1, 2)
            .expect("The swap should be well-defined");
        assert_eq!(vec.buffer, vec![0, 2, 1, 0, 2, 1]);
    }

    #[test]
    fn swap_rows_idx_out_of_bounds() {
        let vec = Vec2d::from_rows_vec(vec![vec![0, 0], vec![1, 1], vec![2, 2]])
            .expect("This should be well-defined.");

        assert_eq!(
            vec.clone()
                .swap_rows(0, 5)
                .expect_err("The swap should fail."),
            Vec2dError::ColIdxOutOfBounds {
                col_idx: 5,
                col_len: vec.col_len()
            }
        );

        assert_eq!(
            vec.clone()
                .swap_rows(5, 2)
                .expect_err("The swap should fail."),
            Vec2dError::ColIdxOutOfBounds {
                col_idx: 5,
                col_len: vec.col_len()
            }
        );
    }

    //#[test]
    //fn elements_1() {
    //    use crate::ring::cyclic::Cyclic;
    //    use std::collections::HashSet;
    //    let output = Vec2d::<Cyclic<2>>::elements(2, 2).collect::<HashSet<_>>();
    //
    //    let correct = vec![
    //        vec![vec![0, 0], vec![0, 0]],
    //        vec![vec![0, 0], vec![0, 1]],
    //        vec![vec![0, 0], vec![1, 0]],
    //        vec![vec![0, 0], vec![1, 1]],
    //        vec![vec![0, 1], vec![0, 0]],
    //        vec![vec![0, 1], vec![0, 1]],
    //        vec![vec![0, 1], vec![1, 0]],
    //        vec![vec![0, 1], vec![1, 1]],
    //        vec![vec![1, 0], vec![0, 0]],
    //        vec![vec![1, 0], vec![0, 1]],
    //        vec![vec![1, 0], vec![1, 0]],
    //        vec![vec![1, 0], vec![1, 1]],
    //        vec![vec![1, 1], vec![0, 0]],
    //        vec![vec![1, 1], vec![0, 1]],
    //        vec![vec![1, 1], vec![1, 0]],
    //        vec![vec![1, 1], vec![1, 1]],
    //    ]
    //    .into_iter()
    //    .map(|vec| {
    //        Vec2d::from_rows_vec(vec)
    //            .expect("This should be well-defined")
    //            .into_vec_2d::<Cyclic<2>>()
    //    })
    //    .collect::<HashSet<_>>();
    //
    //    assert_eq!(correct, output);
    //}
    //
    //#[test]
    //fn elements_2() {
    //    use crate::ring::cyclic::Cyclic;
    //    assert_eq!(Vec2d::<Cyclic<8>>::elements(2, 2).count(), 4096);
    //}

    #[test]
    fn get() {
        let vec = Vec2d::from_rows_vec(vec![vec![1, 2], vec![10, 20], vec![100, 200]])
            .expect("This should be well-defined.");

        assert_eq!(
            vec.get(0, 0).expect("This should be well-defined.").clone(),
            1
        );
        assert_eq!(
            vec.get(0, 1).expect("This should be well-defined.").clone(),
            2
        );

        assert_eq!(
            vec.get(1, 0).expect("This should be well-defined.").clone(),
            10
        );
        assert_eq!(
            vec.get(1, 1).expect("This should be well-defined.").clone(),
            20
        );

        assert_eq!(
            vec.get(2, 0).expect("This should be well-defined.").clone(),
            100
        );
        assert_eq!(
            vec.get(2, 1).expect("This should be well-defined.").clone(),
            200
        );
    }

    #[test]
    fn get_2() {
        let vec = Vec2d::from_rows_arr([[2, 3, 4, 5], [6, 1, 8, 7], [9, 3, 2, 4]]);
        assert_eq!(
            vec.get(0, 0).expect("This should be well-defined.").clone(),
            2
        );
        assert_eq!(
            vec.get(0, 1).expect("This should be well-defined.").clone(),
            3
        );
        assert_eq!(
            vec.get(0, 2).expect("This should be well-defined.").clone(),
            4
        );
        assert_eq!(
            vec.get(0, 3).expect("This should be well-defined.").clone(),
            5
        );

        assert_eq!(
            vec.get(1, 0).expect("This should be well-defined.").clone(),
            6
        );
        assert_eq!(
            vec.get(1, 1).expect("This should be well-defined.").clone(),
            1
        );
        assert_eq!(
            vec.get(1, 2).expect("This should be well-defined.").clone(),
            8
        );
        assert_eq!(
            vec.get(1, 3).expect("This should be well-defined.").clone(),
            7
        );

        assert_eq!(
            vec.get(2, 0).expect("This should be well-defined.").clone(),
            9
        );
        assert_eq!(
            vec.get(2, 1).expect("This should be well-defined.").clone(),
            3
        );
        assert_eq!(
            vec.get(2, 2).expect("This should be well-defined.").clone(),
            2
        );
        assert_eq!(
            vec.get(2, 3).expect("This should be well-defined.").clone(),
            4
        );
    }

    #[test]
    fn get_mut() {
        let mut vec = Vec2d::from_rows_vec(vec![vec![1, 2], vec![10, 20], vec![100, 200]])
            .expect("This should be well-defined.");

        assert_eq!(
            vec.get_mut(0, 0)
                .expect("This should be well-defined.")
                .clone(),
            1
        );
        assert_eq!(
            vec.get_mut(0, 1)
                .expect("This should be well-defined.")
                .clone(),
            2
        );

        assert_eq!(
            vec.get_mut(1, 0)
                .expect("This should be well-defined.")
                .clone(),
            10
        );
        assert_eq!(
            vec.get_mut(1, 1)
                .expect("This should be well-defined.")
                .clone(),
            20
        );

        assert_eq!(
            vec.get_mut(2, 0)
                .expect("This should be well-defined.")
                .clone(),
            100
        );
        assert_eq!(
            vec.get_mut(2, 1)
                .expect("This should be well-defined.")
                .clone(),
            200
        );

        *vec.get_mut(0, 0).expect("This should be well-defined.") *= -1;
        *vec.get_mut(0, 0).expect("This should be well-defined.") *= -1;
    }
}
