use bitvec::prelude::*;
use custom_error::custom_error;
use itertools::Itertools;
use std::fmt;

custom_error! {
    #[derive(PartialEq, Eq, Clone)]
    pub Vec2dError
    IteratorLengthsMismatch = "Axes in the iterator had different lengths.",
    EmptyIterator = "Iterator was empty.",
    DifferentLengths{left_len: usize, right_len: usize} = "Left axis has len {left_len}, but right axis has len {right_len}.",
    IdxOutOfBounds{idx: usize, bound: usize} = "The index {idx} should be less than {bound}.",
    Unexpected = "This bug is unexpected"
}

/// Struct representing 2d vector, optimized for collumn operations.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Vec2d<T> {
    pub(super) buffer: Vec<T>,
    pub(super) nof_rows: usize,
    pub(super) nof_cols: usize,
}

impl<T> Vec2d<T> {
    /// Returns the number of rows of the `Vec2d`.
    #[must_use]
    pub const fn nof_rows(&self) -> usize {
        self.nof_rows
    }

    /// Returns the number of collums of the `Vec2d`.
    #[must_use]
    pub const fn nof_cols(&self) -> usize {
        self.nof_cols
    }

    /// Returns the length of a collumn of the `Vec2d`.
    /// Note that it is by definition its number of rows.
    #[must_use]
    pub const fn col_len(&self) -> usize {
        self.nof_rows()
    }

    /// Returns the length of a row of the `Vec2d`.
    /// Note that it is by definition its number of collums.
    #[must_use]
    pub const fn row_len(&self) -> usize {
        self.nof_cols()
    }

    /// The `shape` of the `Vec2d` is a pair `(nof_rows, nof_cols)`.
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.nof_rows(), self.nof_cols())
    }

    /// Determines if then `Vec2d` is square.
    #[must_use]
    pub const fn is_square(&self) -> bool {
        self.nof_cols() == self.nof_rows()
    }

    /// Returns an immutable reference to the element in the
    /// `i`-th row and `j`-th collumn of a `Vec2d`.
    /// # Safety
    /// The caller must guarantee that `i < nof_rows` and `j < nof_cols`.
    #[must_use]
    pub unsafe fn get_unchecked(&self, i: usize, j: usize) -> &T {
        self.buffer
            .get_unchecked(i.strict_add(self.col_len().strict_mul(j)))
    }

    /// Returns an immutable reference to the element in the
    /// `i`-th row and `j`-th collumn of a `Vec2d`.
    /// # Errors
    /// If one of the indices is out of bounds, returns error.
    pub fn get(&self, i: usize, j: usize) -> Result<&T, Vec2dError> {
        if i >= self.nof_rows() {
            Err(Vec2dError::IdxOutOfBounds {
                idx: i,
                bound: self.nof_rows(),
            })
        } else if j >= self.nof_cols() {
            Err(Vec2dError::IdxOutOfBounds {
                idx: j,
                bound: self.nof_cols(),
            })
        } else {
            Ok(unsafe { self.get_unchecked(i, j) })
        }
    }

    /// Returns a mutable reference to the element in the
    /// `i`-th row and `j`-th collumn of a `Vec2d`.
    /// # Safety
    /// The caller must guarantee that `i < nof_rows` and `j < nof_cols`.
    pub unsafe fn get_unchecked_mut(&mut self, i: usize, j: usize) -> &mut T {
        let col_len = self.col_len();
        self.buffer
            .get_unchecked_mut(i.strict_add(col_len.strict_mul(j)))
    }

    /// Returns a mutable reference to the element in the
    /// `i`-th row and `j`-th collumn of a `Vec2d`.
    /// # Errors
    /// If one of the indices is out of bounds, returns error.
    pub fn get_mut(&mut self, i: usize, j: usize) -> Result<&mut T, Vec2dError> {
        if i >= self.nof_rows() {
            Err(Vec2dError::IdxOutOfBounds {
                idx: i,
                bound: self.nof_rows(),
            })
        } else if j >= self.nof_cols() {
            Err(Vec2dError::IdxOutOfBounds {
                idx: j,
                bound: self.nof_cols(),
            })
        } else {
            Ok(unsafe { self.get_unchecked_mut(i, j) })
        }
    }

    /// Transposes the `Vec2d.`.
    /// For the correctness of the algorithm see
    /// `<https://www.geeksforgeeks.org/inplace-m-x-n-size-matrix-transpose/>`.
    pub fn transpose(&mut self) {
        #![allow(clippy::arithmetic_side_effects, reason = "This uses mod arithmetic.")]
        let row_len = self.row_len();
        let len_minus_1 = self.buffer.len() - 1;

        let mut cycle_hashing = bitvec![0; self.buffer.len()];
        cycle_hashing.set(0, true);
        cycle_hashing.set(len_minus_1, true);

        for start_idx in 0..self.buffer.len() {
            if !cycle_hashing[start_idx] {
                let mut curr = start_idx;

                'cycling: loop {
                    let next = (curr * row_len) % len_minus_1;
                    self.buffer.swap(start_idx, next);
                    cycle_hashing.set(curr, true);
                    curr = next;

                    if cycle_hashing[curr] {
                        break 'cycling;
                    }
                }
            }
        }

        self.nof_cols = self.nof_rows();
        self.nof_rows = row_len;
    }

    /// Iterator of immutable references to a collumn of a given index.
    /// # Safety
    /// The caller must guarantee that `col_idx < self.nof_cols()`.
    pub unsafe fn col_unchecked(&self, col_idx: usize) -> impl Iterator<Item = &T> {
        let begin = col_idx.strict_mul(self.col_len());
        let end = begin.strict_add(self.col_len());
        self.buffer[begin..end].iter()
    }

    /// Iterator of mutable references to a collumn of a given index.
    /// # Safety
    /// The caller must guarantee that `col_idx < self.nof_cols()`.
    pub unsafe fn col_mut_unchecked(&mut self, col_idx: usize) -> impl Iterator<Item = &mut T> {
        let begin = col_idx.strict_mul(self.col_len());
        let end = begin.strict_add(self.col_len());
        self.buffer[begin..end].iter_mut()
    }

    /// Iterator of immutable references to a collumn of a given index.
    /// # Errors
    /// If `col_idx < self.nof_cols()`, function returns error.
    pub fn col(&self, col_idx: usize) -> Result<impl Iterator<Item = &T>, Vec2dError> {
        (col_idx < self.nof_cols())
            .then(|| unsafe { self.col_unchecked(col_idx) })
            .ok_or_else(|| Vec2dError::IdxOutOfBounds {
                idx: col_idx,
                bound: self.nof_cols(),
            })
    }

    /// Iterator of mutable references to a collumn of a given index.
    /// # Errors
    /// If `col_idx < self.nof_cols()`, function returns error.
    pub fn col_mut(&mut self, col_idx: usize) -> Result<impl Iterator<Item = &mut T>, Vec2dError> {
        let nof_cols = self.nof_cols();
        (col_idx < nof_cols)
            .then(|| unsafe { self.col_mut_unchecked(col_idx) })
            .ok_or(Vec2dError::IdxOutOfBounds {
                idx: col_idx,
                bound: nof_cols,
            })
    }

    /// Iterator of iterators to collumns of the `Vec2d`, borrowing the `Vec2d`.
    /// # Panics
    /// This function never panics.
    pub fn cols(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..self.nof_cols()).map(|col_idx| unsafe { self.col_unchecked(col_idx) })
    }

    /// Iterator of iterators to collumns of the `Vec2d`, taking ownership of the `Vec2d`.
    /// # Panics
    /// This function never panics.
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

    /// Constructs a `Vec2d` from an iterator of iterators over `T`
    /// representing its collumns.
    /// # Errors
    /// If either `cols_iterator` is empty or some its elements have diffrerent
    /// lengths, the function returns error.
    pub fn from_cols<ColIt: Iterator<Item = T>, It: Iterator<Item = ColIt>>(
        mut cols_iterator: It,
    ) -> Result<Self, Vec2dError> {
        let mut buffer = cols_iterator
            .next()
            .map(Iterator::collect::<Vec<_>>)
            .ok_or(Vec2dError::EmptyIterator)?;
        let nof_rows = buffer.len();
        let mut old_len = buffer.len();
        let mut nof_cols: usize = 1;

        for col_iterator in cols_iterator {
            buffer.extend(col_iterator);
            let new_len = buffer.len();

            if new_len == old_len.strict_add(nof_rows) {
                old_len = new_len;
                nof_cols = nof_cols.strict_add(1);
            } else {
                return Err(Vec2dError::IteratorLengthsMismatch);
            }
        }

        Ok(Self {
            buffer,
            nof_rows,
            nof_cols,
        })
    }

    /// Constructs a `Vec2d` from a `Vec<Vec<T>>` representing its collumns. Useful wrapper for `from_cols`.
    /// # Errors
    /// If either `cols_vector` is empty or some its elements have diffrerent
    /// lengths, the function returns error.
    pub fn from_cols_vec(cols_vec: Vec<Vec<T>>) -> Result<Self, Vec2dError> {
        Self::from_cols(cols_vec.into_iter().map(IntoIterator::into_iter))
    }

    /// From a `Vec2d<T>` constructs a `Vec<Vec<T>>` representing its collumns. Useful wrapper for `into_cols`.
    #[must_use]
    pub fn into_cols_vec(self) -> Vec<Vec<T>> {
        self.into_cols()
            .map(Iterator::collect::<Vec<_>>)
            .collect::<Vec<_>>()
    }

    /// Constructs a `Vec2d` from an array `[[T; NOF_COLS]; NOF_ROWS]` representing its
    /// collumns.
    /// Useful for defining custom `Vec2d`'s.
    /// # Panics
    /// This function can panic only of `NOF_ROWS * NOF_COLS` exceeds `usize::MAX`.
    pub fn from_cols_arr<const NOF_ROWS: usize, const NOF_COLS: usize>(
        cols_arr: [[T; NOF_ROWS]; NOF_COLS],
    ) -> Self {
        assert!(
            NOF_ROWS.strict_mul(NOF_COLS) < usize::MAX,
            "Max size of a Vec<T> is usize::MAX = {}.",
            usize::MAX
        );
        Self::from_cols(cols_arr.into_iter().map(IntoIterator::into_iter))
            .expect("The arrays have correct bounds.")
    }

    /// Iterator of immutable references to a row of a given index.
    /// # Safety
    /// The caller must guarantee that `row_idx < self.nof_rows()`.
    pub unsafe fn row_unchecked(&self, row_idx: usize) -> impl Iterator<Item = &T> {
        (0..self.row_len())
            .map(move |idx| self.col_len().strict_mul(idx).strict_add(row_idx))
            .map(|idx| self.buffer.get_unchecked(idx))
    }

    /// Iterator of mutable references to a row of a given index.
    /// # Safety
    /// The caller must guarantee that `row_idx < self.nof_rows()`.
    pub unsafe fn row_mut_unchecked(&mut self, row_idx: usize) -> impl Iterator<Item = &mut T> {
        let row_len = self.row_len();
        let col_len = self.col_len();

        let buffer_ptr = self.buffer.as_mut_ptr();

        (0..row_len)
            .map(move |idx| col_len.strict_mul(idx).strict_add(row_idx))
            .map(move |idx| &mut *buffer_ptr.add(idx))
    }

    /// Iterator of immutable references to a row of a given index.
    /// # Errors
    /// If `row_idx < self.nof_rows()`, function returns error.
    pub fn row(&self, row_idx: usize) -> Result<impl Iterator<Item = &T>, Vec2dError> {
        (row_idx < self.nof_rows())
            .then(|| unsafe { self.row_unchecked(row_idx) })
            .ok_or_else(|| Vec2dError::IdxOutOfBounds {
                idx: row_idx,
                bound: self.nof_rows(),
            })
    }

    /// Iterator of mutable references to a row of a given index.
    /// # Errors
    /// If `row_idx < self.nof_rows()`, function returns error.
    pub fn row_mut(&mut self, row_idx: usize) -> Result<impl Iterator<Item = &mut T>, Vec2dError> {
        let nof_rows = self.nof_rows();
        (row_idx < nof_rows)
            .then(|| unsafe { self.row_mut_unchecked(row_idx) })
            .ok_or(Vec2dError::IdxOutOfBounds {
                idx: row_idx,
                bound: nof_rows,
            })
    }

    /// Iterator of iterators to rows of the `Vec2d`, borrowing the `Vec2d`.
    /// # Panics
    /// This function never panics.
    pub fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..self.nof_rows()).map(move |row_idx| unsafe { self.row_unchecked(row_idx) })
    }

    /// Iterator of iterators to rows of the `Vec2d`, taking ownership of the `Vec2d`.
    /// This is intended only for convenience, it is not optimized.
    /// # Panics
    /// This function never panics.
    pub fn into_rows(mut self) -> impl Iterator<Item = impl Iterator<Item = T>> {
        self.transpose();
        self.into_cols()
    }

    /// Constructs a `Vec2d` from an iterator of iterators over `T`
    /// representing its rows.
    /// This is intended only for convenience, it is not optimized.
    /// # Errors
    /// If either `rows_iterator` is empty or some its elements have diffrerent
    /// lengths, the function returns error.
    pub fn from_rows<RowIt: Iterator<Item = T>, It: Iterator<Item = RowIt>>(
        rows_iterator: It,
    ) -> Result<Self, Vec2dError> {
        Self::from_cols(rows_iterator).map(|mut vec2d| {
            vec2d.transpose();
            vec2d
        })
    }

    /// Constructs a `Vec2d` from a `Vec<Vec<T>>` representing its rows.
    /// Useful wrapper for `from_rows`.
    /// This is intended only for convenience, it is not optimized.
    /// # Errors
    /// If either `cols_vector` is empty or some its elements have diffrerent
    /// lengths, the function returns error.
    pub fn from_rows_vec(rows_vec: Vec<Vec<T>>) -> Result<Self, Vec2dError> {
        Self::from_rows(rows_vec.into_iter().map(IntoIterator::into_iter))
    }

    /// From a `Vec2d<T>` constructs a `Vec<Vec<T>>` representing its rows. Useful wrapper for `into_rows`.
    pub fn into_rows_vec(self) -> Vec<Vec<T>> {
        self.into_rows()
            .map(Iterator::collect::<Vec<_>>)
            .collect::<Vec<_>>()
    }

    /// Constructs a `Vec2d` from an array `[[T; NOF_COLS]; NOF_ROWS]` representing its rows.
    /// Useful for defining custom `Vec2d`'s.
    /// # Panics
    /// This function can panic only of `NOF_ROWS * NOF_COLS` exceeds `usize::MAX`.
    pub fn from_rows_arr<const NOF_ROWS: usize, const NOF_COLS: usize>(
        rows_arr: [[T; NOF_COLS]; NOF_ROWS],
    ) -> Self {
        assert!(
            NOF_ROWS.strict_mul(NOF_COLS) < usize::MAX,
            "Max size of a Vec<T> is usize::MAX = {}.",
            usize::MAX
        );
        Self::from_rows(rows_arr.into_iter().map(std::iter::IntoIterator::into_iter))
            .expect("The arrays have correct bounds.")
    }

    /// Create a `Vec2d` from `Vec2d`'s `left` and `right`.
    /// --------------------
    /// | `left` | `right` |
    /// --------------------
    /// # Safety
    /// The caller must ensure that `left` and `right` have the same `col_len`.
    #[must_use]
    pub unsafe fn merge_horizontally_unchecked(left: Self, mut right: Self) -> Self {
        let left_nof_cols = left.nof_cols();
        let mut buffer = left.buffer;
        buffer.append(&mut right.buffer);
        Self {
            buffer,
            nof_cols: left_nof_cols.strict_add(right.nof_cols()),
            nof_rows: right.nof_rows(),
        }
    }

    /// Create a `Vec2d` from `Vec2d`'s `left` and `right`.
    /// --------------------
    /// | `left` | `right` |
    /// --------------------
    /// # Errors
    /// If `left` and `right` have different `col_len`, the function returns error.
    pub fn merge_horizontally(left: Self, right: Self) -> Result<Self, Vec2dError> {
        let left_col_len = left.col_len();
        let right_col_len = right.col_len();
        (left_col_len == right_col_len)
            .then(|| unsafe { Self::merge_horizontally_unchecked(left, right) })
            .ok_or(Vec2dError::DifferentLengths {
                left_len: left_col_len,
                right_len: right_col_len,
            })
    }

    /// Create a `Vec2d` from `Vec2d`'s `top` and `bot`.
    /// ------------
    /// | `top`    |
    /// ------------
    /// | `bottom` |
    /// ------------
    /// # Errors
    /// If `top` and `bot` have different `row_len`, the function returns error.
    pub fn merge_vertically(mut top: Self, mut bot: Self) -> Result<Self, Vec2dError> {
        top.transpose();
        bot.transpose();
        let mut merged = Self::merge_horizontally(top, bot)?;
        merged.transpose();
        Ok(merged)
    }

    /// Creates a `Vec2d` from `Vec2d`'s `left_top`, `right_top`, `left_bot` and `right_bot`.
    /// ----------------------------
    /// | `left_top` | `right_top` |
    /// ----------------------------
    /// | `left_bot` | `right_bot` |
    /// ----------------------------
    /// # Errors
    /// If one of the following is not satisfied:
    /// - `left_top.row_len() == left_bot.row_len()`,
    /// - `right_top.row_len() == right_bot.row_len()`,
    /// - `left_top.col_len() == right_top.col_len()`,
    /// - `left_bot.col_len() == right_bot.col_len()`,
    ///
    /// the function returns error.
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

    /// Given a `col_idx`, splits `Vec2d` into
    /// --------------------
    /// | `left` | `right` |
    /// --------------------
    /// in such a way `left.nof_cols() = col_idx` and `right.nof_cols() = self.nof_cols() - col_idx`.
    /// # Safety
    /// The caller must ensure that `col_idx` in `1..self.nof_cols()`.
    #[must_use]
    pub unsafe fn split_horizontally_unchecked(self, col_idx: usize) -> (Self, Self) {
        let nof_rows = self.nof_rows();
        let nof_cols = self.nof_cols();

        let mut left_buffer = self.buffer;
        let right_buffer = left_buffer.split_off(col_idx.strict_mul(self.nof_rows));
        let left = Self {
            buffer: left_buffer,
            nof_cols: col_idx,
            nof_rows,
        };

        let right = Self {
            buffer: right_buffer,
            nof_cols: nof_cols.strict_sub(col_idx),
            nof_rows,
        };

        (left, right)
    }

    /// Given a `col_idx`, splits `Vec2d` into
    /// --------------------
    /// | `left` | `right` |
    /// --------------------
    /// in such a way `left.nof_cols() = col_idx` and `right.nof_cols() = self.nof_cols() - col_idx`.
    /// # Errors
    ///  If `col_idx` is not in `1..self.nof_cols()`, the function returns error.
    pub fn split_horizontally(self, col_idx: usize) -> Result<(Self, Self), Vec2dError> {
        let col_len = self.col_len();
        (0 < col_idx && col_idx < self.nof_cols())
            .then(|| unsafe { self.split_horizontally_unchecked(col_idx) })
            .ok_or(Vec2dError::IdxOutOfBounds {
                idx: col_idx,
                bound: col_len,
            })
    }

    /// Given a `row_idx`, splits `Vec2d` into
    /// ---------
    /// | `top` |
    /// ---------
    /// | `bot` |
    /// ---------
    /// in such a way `top.nof_rows() = row_idx` and `bot.nof_rows() = self.nof_rows() - row_idx`.
    /// # Errors
    ///  If `row_idx` is not in `1..self.nof_rows()`, the function returns error.
    pub fn split_vertically(mut self, row_idx: usize) -> Result<(Self, Self), Vec2dError> {
        self.transpose();
        let (mut top, mut bot) = self.split_horizontally(row_idx)?;
        top.transpose();
        bot.transpose();
        Ok((top, bot))
    }

    /// Given a `col_idx` and `row_idx`, splits a `Vec2d` into
    ///  --------------------------------
    ///  | `top_left` | `top_right` |
    ///  --------------------------------
    ///  | `bot_left` | `bot_right` |
    ///  --------------------------------
    /// in such a way that `top_left.shape() = (col_idx, row_idx)`.
    /// # Errors
    /// If `col_idx` is not in `1..self.nof_cols()` and `row_idx` is not in `1..self.nof_rows()`,
    /// the function returns error.
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

    /// Swaps collumns of indices `i` and `j`.
    /// # Safety
    /// The caller must ensure that `i`, `j` < `self.nof_cols()`.
    pub unsafe fn swap_cols_unchecked(&mut self, i: usize, j: usize) {
        if i != j {
            let col_len = self.col_len();
            (0..col_len)
                .map(|idx| {
                    (
                        i.strict_mul(col_len).strict_add(idx),
                        j.strict_mul(col_len).strict_add(idx),
                    )
                })
                .for_each(|(col_i_idx, col_j_idx)| {
                    self.buffer.swap_unchecked(col_i_idx, col_j_idx);
                });
        }
    }

    /// Swaps collumns of indices `i` and `j`.
    /// # Errors
    /// If `i`, or `j` is greater or equal than `self.nof_cols()`, function returns error.
    pub fn swap_cols(&mut self, i: usize, j: usize) -> Result<(), Vec2dError> {
        let nof_cols = self.nof_cols();

        if i >= nof_cols {
            Err(Vec2dError::IdxOutOfBounds {
                idx: i,
                bound: nof_cols,
            })
        } else if j >= nof_cols {
            Err(Vec2dError::IdxOutOfBounds {
                idx: j,
                bound: nof_cols,
            })
        } else {
            unsafe {
                self.swap_cols_unchecked(i, j);
            };
            Ok(())
        }
    }

    /// Swaps rows of indices `i` and `j`.
    /// # Safety
    /// The caller must ensure that `i`, `j` < `self.nof_rows()`.
    pub unsafe fn swap_rows_unchecked(&mut self, i: usize, j: usize) {
        let row_len = self.row_len();
        let nof_rows = self.nof_rows();
        if i != j {
            (0..row_len)
                .map(|idx| {
                    (
                        i.strict_add(nof_rows.strict_mul(idx)),
                        j.strict_add(nof_rows.strict_mul(idx)),
                    )
                })
                .for_each(|(row_i_idx, row_j_idx)| {
                    self.buffer.swap_unchecked(row_i_idx, row_j_idx);
                });
        }
    }

    /// Swaps rows of indices `i` and `j`.
    /// # Errors
    /// If `i`, or `j` is greater or equal than `self.nof_rows()`, function returns error.
    pub fn swap_rows(&mut self, i: usize, j: usize) -> Result<(), Vec2dError> {
        let nof_rows = self.nof_rows();

        if i >= nof_rows {
            Err(Vec2dError::IdxOutOfBounds {
                idx: i,
                bound: nof_rows,
            })
        } else if j >= nof_rows {
            Err(Vec2dError::IdxOutOfBounds {
                idx: j,
                bound: nof_rows,
            })
        } else {
            unsafe {
                self.swap_rows_unchecked(i, j);
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
            buffer: self.buffer.into_iter().map(Into::into).collect::<Vec<_>>(),
        }
    }
}

impl<T: Copy> Vec2d<T> {
    /// Given an `ExactSizeIterator<T>`, generate all the `Vec2d`'s with entries being the
    /// elements of this iterator.
    pub fn elements<E: ExactSizeIterator<Item = T>>(
        nof_cols: usize,
        nof_rows: usize,
        elements_iterator: E,
    ) -> impl Iterator<Item = Self> {
        let elements = elements_iterator.collect::<Vec<_>>();

        (0..nof_cols.strict_mul(nof_rows))
            .map(|_| (0..elements.len()))
            .multi_cartesian_product()
            .map(move |buffer_indices: Vec<usize>| Self {
                nof_cols,
                nof_rows,
                buffer: buffer_indices
                    .into_iter()
                    .map(|idx| unsafe { *elements.get_unchecked(idx) })
                    .collect::<Vec<_>>(),
            })
    }
}

impl<T: fmt::Display> fmt::Display for Vec2d<T> {
    /// Displays the 2d vector in a format
    /// (`self.nof_cols()` x `self.nof_rows()`)
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
            .map(String::len)
            .max()
            .unwrap_or(0);

        let stringified = elementwise_stringified
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|entry| {
                        format!("{}{}", " ".repeat(padding.strict_sub(entry.len())), entry)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .map(|row_stringified| format!("[{row_stringified}]"))
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

impl<T: fmt::Display> fmt::Debug for Vec2d<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[cfg(test)]
mod test {
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
        #[allow(clippy::all, reason = "shut up")]
        let vec = vec![vec!['a', 'b', 'c'], vec!['d', 'e', 'f']];
        let vec2d = Vec2d::from_cols_vec(vec).expect("This should be well-defined");
        assert_eq!(vec2d.shape(), (3, 2));
        assert_eq!(vec2d.buffer, vec!['a', 'b', 'c', 'd', 'e', 'f']);
    }

    #[test]
    fn from_cols_empty_iterator() {
        let empty_vec: Vec<Vec<u8>> = vec![];
        let empty_vec2d = Vec2d::from_cols(
            empty_vec
                .into_iter()
                .map(std::iter::IntoIterator::into_iter),
        );
        assert_eq!(
            empty_vec2d.expect_err("This should be empty 2d vector"),
            Vec2dError::EmptyIterator
        );
    }

    #[test]
    fn from_cols_different_col_lens() {
        let different_col_lens_vec = vec![vec![1, 2, 3], vec![4, 5]];
        let different_col_lens_vec2d = Vec2d::from_cols(
            different_col_lens_vec
                .into_iter()
                .map(std::iter::IntoIterator::into_iter),
        );
        assert_eq!(
            different_col_lens_vec2d
                .expect_err("This should be non-empty but have different col lens."),
            Vec2dError::IteratorLengthsMismatch
        );
    }

    #[test]
    fn col() {
        let vec = Vec2d::from_cols_arr([[1, 1], [2, 2], [3, 3], [0, 0]]);
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
            { vec![0, 0] }
        );
        assert_eq!(
            vec.col(7).err().expect("This should fail."),
            Vec2dError::IdxOutOfBounds {
                idx: 7,
                bound: vec.row_len()
            }
        );
    }

    #[test]
    fn col_mut() {
        let mut vec = Vec2d::from_cols_arr([[1, 1], [2, 2], [3, 3], [4, 4]]);
        let mut vec1 = vec.clone();

        vec.col_mut(0)
            .expect("This should be well-defined.")
            .for_each(|x| (*x) *= 10);
        assert_eq!(
            vec.into_cols_vec(),
            vec![vec![10, 10], vec![2, 2], vec![3, 3], vec![4, 4]]
        );

        assert_eq!(
            vec1.col_mut(8).err().expect("This should fail."),
            Vec2dError::IdxOutOfBounds { idx: 8, bound: 4 }
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
        assert_eq!(vec.nof_rows(), vec_t.nof_cols());
        assert_eq!(vec.nof_cols(), vec_t.nof_rows());
        println!("{vec}");
        println!("{vec_t}");

        assert_eq!(vec_t.buffer, vec!['a', 'd', 'b', 'e', 'c', 'f']);
    }

    #[test]
    fn transpose_2() {
        let vec = Vec2d::from_cols_arr([['a', 'd'], ['b', 'e'], ['c', 'f']]);
        let mut vec_t = vec.clone();
        vec_t.transpose();
        assert_eq!(vec.nof_rows(), vec_t.nof_cols());
        assert_eq!(vec.nof_cols(), vec_t.nof_rows());
        assert_eq!(vec_t.buffer, vec!['a', 'b', 'c', 'd', 'e', 'f']);
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
        let empty_vec2d = Vec2d::from_rows(empty_vec.into_iter().map(IntoIterator::into_iter));
        assert_eq!(
            empty_vec2d.expect_err("This should be empty 2d vector"),
            Vec2dError::EmptyIterator
        );
    }

    #[test]
    fn from_rows_different_row_lens() {
        let different_row_lens_vec = vec![vec![1, 2, 3], vec![4, 5]];
        let different_row_lens_vec2d = Vec2d::from_rows(
            different_row_lens_vec
                .into_iter()
                .map(IntoIterator::into_iter),
        );
        assert_eq!(
            different_row_lens_vec2d
                .expect_err("This should be non-empty but have different row lens."),
            Vec2dError::IteratorLengthsMismatch
        );
    }

    #[test]
    fn from_rows_arr() {
        let vec = Vec2d::from_rows_arr([[1, 2, 3, 3], [4, 5, 6, 6], [7, 8, 9, 9]]);
        assert_eq!(vec.buffer, vec![1, 4, 7, 2, 5, 8, 3, 6, 9, 3, 6, 9]);
    }

    #[test]
    fn row() {
        let vec = Vec2d::from_rows_arr([[1, 1], [2, 2], [3, 3], [0, 0]]);
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
            vec.row(7).err().expect("This should fail."),
            Vec2dError::IdxOutOfBounds {
                idx: 7,
                bound: vec.col_len()
            }
        );
    }

    #[test]
    fn row_mut() {
        let mut vec = Vec2d::from_rows_arr([[1, 1], [2, 2], [3, 3], [4, 4]]);
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
            Vec2dError::IdxOutOfBounds { idx: 8, bound: 4 }
        );
    }

    #[test]
    fn merge_horizontally() {
        let vec2d_1 = Vec2d::from_cols_arr([[1, 2, 3], [4, 5, 6]]);
        let vec2d_2 = Vec2d::from_cols_arr([[7, 8, 9]]);
        let vec2d =
            Vec2d::merge_horizontally(vec2d_1, vec2d_2).expect("The merge should be well-defined.");

        assert_eq!(vec2d.shape(), (3, 3));
        assert_eq!(vec2d.buffer, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn merge_horizontally_different_col_lens() {
        let u = Vec2d::from_cols_arr([[1, 2, 3], [4, 5, 6]]);
        let v = Vec2d::from_cols_arr([[7, 8, 9, 10]]);
        let vec = Vec2d::merge_horizontally(u.clone(), v.clone());
        assert_eq!(
            vec.expect_err("The merge should fail."),
            Vec2dError::DifferentLengths {
                left_len: u.col_len(),
                right_len: v.col_len()
            }
        );
    }

    #[test]
    fn merge_vertically() {
        let u = Vec2d::from_rows_arr([['a', 'b', 'c'], ['d', 'e', 'f']]);
        let v = Vec2d::from_rows_arr([['g', 'h', 'i']]);

        let vec = Vec2d::merge_vertically(u, v).expect("The merge should be well-defined.");

        assert_eq!(vec.shape(), (3, 3));
        assert_eq!(
            vec.buffer,
            vec!['a', 'd', 'g', 'b', 'e', 'h', 'c', 'f', 'i']
        );
    }

    #[test]
    fn merge_vertically_different_row_lens() {
        let u = Vec2d::from_rows_arr([['a', 'b', 'c'], ['d', 'e', 'f']]);
        let v = Vec2d::from_rows_arr([['g', 'h', 'i', 'j']]);

        let vec = Vec2d::merge_vertically(u.clone(), v.clone());

        assert_eq!(
            vec.expect_err("The merge should fail."),
            Vec2dError::DifferentLengths {
                left_len: u.row_len(),
                right_len: v.row_len()
            }
        );
    }

    #[test]
    fn merge() {
        let top_left = Vec2d::from_cols_arr([[1, 2, 3]]);
        let top_right = Vec2d::from_cols_arr([[5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19]]);
        let bot_left = Vec2d::from_cols_arr([[4]]);
        let bot_right = Vec2d::from_rows_arr([[8, 12, 16, 20]]);

        let vec = Vec2d::merge(top_left, top_right, bot_left, bot_right)
            .expect("The merge should be well-defined.");
        assert_eq!(vec.buffer, (1..=20).collect::<Vec<_>>());
    }

    #[test]
    fn split_horizontally() {
        let vec = Vec2d::from_rows_arr([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
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
        let vec = Vec2d::from_rows_arr([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

        let split_at_zero = vec.clone().split_horizontally(0);
        assert_eq!(
            split_at_zero.expect_err("The split should fail."),
            Vec2dError::IdxOutOfBounds {
                idx: 0,
                bound: vec.col_len()
            }
        );

        let split_at_col_len = vec.clone().split_horizontally(vec.col_len());
        assert_eq!(
            split_at_col_len.expect_err("The split should fail."),
            Vec2dError::IdxOutOfBounds {
                idx: vec.col_len(),
                bound: vec.col_len()
            }
        );
    }

    #[test]
    fn split_vertically() {
        let vec = Vec2d::from_rows_arr([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

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
        let vec = Vec2d::from_rows_arr([[1, 2, 3], [4, 5, 6]]);

        let split_at_zero = vec.clone().split_vertically(0);
        assert_eq!(
            split_at_zero.expect_err("The split should fail."),
            Vec2dError::IdxOutOfBounds {
                idx: 0,
                bound: vec.row_len()
            }
        );

        let split_at_row_len = vec.clone().split_vertically(vec.row_len());
        assert_eq!(
            split_at_row_len.expect_err("The split should fail."),
            Vec2dError::IdxOutOfBounds {
                idx: vec.row_len(),
                bound: vec.row_len()
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
            Vec2dError::IdxOutOfBounds {
                idx: 5,
                bound: vec.row_len()
            }
        );

        assert_eq!(
            vec.clone()
                .swap_cols(5, 3)
                .expect_err("The swap should fail."),
            Vec2dError::IdxOutOfBounds {
                idx: 5,
                bound: vec.row_len()
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
            Vec2dError::IdxOutOfBounds {
                idx: 5,
                bound: vec.col_len()
            }
        );

        assert_eq!(
            vec.clone()
                .swap_rows(5, 2)
                .expect_err("The swap should fail."),
            Vec2dError::IdxOutOfBounds {
                idx: 5,
                bound: vec.col_len()
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
