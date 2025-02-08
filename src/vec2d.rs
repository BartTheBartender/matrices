use itertools;
use std::fmt;

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

    pub fn cols(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..self.nof_cols())
            .map(|col_idx| col_idx * self.col_len())
            .map(|col_idx| self.buffer[col_idx..col_idx + self.col_len()].iter())
    }

    pub fn from_cols(cols_iterator: impl Iterator<Item = impl Iterator<Item = T>>) -> Option<Self> {
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
                        return None;
                    }
                }
                None => nof_rows = Some(new_len),
            }
            old_len = new_len;
        }

        nof_rows.map(|nof_rows| Self {
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

    pub fn transpose(&mut self) {
        let nof_cols = self.nof_cols();
        let nof_rows = self.nof_rows();

        itertools::iproduct!(0..nof_cols, 0..nof_rows)
            .map(|(i, j)| (nof_rows * i + j, nof_rows * j + i))
            .for_each(|(p, q)| {
                //unsafe{self.buffer.swap_unchecked(p,q)}
                self.buffer.swap(p, q);
            });

        self.nof_rows = nof_cols;
        self.nof_cols = nof_rows;
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
        let vec2d = Vec2d::from_cols(cols).expect("The creation of the vector should be successful");
        println!("{}", vec2d);
        assert_eq!(vec2d.nof_rows(), 3, "The number of rows is incorrect");
        assert_eq!(vec2d.nof_cols(), 4, "The number of cols is incorrect");
    }
}
