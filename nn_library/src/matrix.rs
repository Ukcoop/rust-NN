use nalgebra::DMatrix;
use rayon::prelude::*;
use std::error::Error;

#[derive(Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: DMatrix<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        let data = DMatrix::zeros(rows, cols);
        return Matrix { rows, cols, data };
    }

    pub fn from_vector(vector: &[f64]) -> Matrix {
        let data = DMatrix::from_vec(vector.len(), 1, vector.to_owned());

        return Matrix {
            rows: vector.len(),
            cols: 1,
            data,
        };
    }

    pub fn to_vector(&self) -> Vec<f64> {
        self.data.clone().as_slice().to_vec()
    }

    pub fn transpose(matrix: &Matrix) -> Matrix {
        let data = matrix.data.transpose();

        return Matrix {
            rows: matrix.cols,
            cols: matrix.rows,
            data,
        };
    }

    pub fn randomize(&mut self) {
        self.data
            .as_mut_slice()
            .par_iter_mut()
            .for_each(|x| *x = rand::random());
    }

    pub fn add(&mut self, other: &Matrix) -> Result<(), Box<dyn Error>> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrices must be the same size".into());
        }

        let cols = self.cols;

        self.data
            .as_mut_slice()
            .par_chunks_mut(cols)
            .zip(other.data.as_slice().par_chunks(cols))
            .for_each(|(mine, theirs)| {
                for i in 0..cols {
                    mine[i] += theirs[i];
                }
            });

        return Ok(());
    }

    pub fn add_number(&mut self, number: f64) {
        let cols = self.cols;

        self.data
            .as_mut_slice()
            .par_chunks_mut(cols)
            .for_each(|row| {
                for x in row {
                    *x += number;
                }
            });
    }

    pub fn subtract(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err("Matrices must be the same size".into());
        }
        let mut result = Matrix::new(a.rows, a.cols);
        let cols = a.cols;

        a.data
            .as_slice()
            .par_chunks(cols)
            .zip(b.data.as_slice().par_chunks(cols))
            .zip(result.data.as_mut_slice().par_chunks_mut(cols))
            .for_each(|((ar, br), rr)| {
                for i in 0..cols {
                    rr[i] = ar[i] - br[i];
                }
            });

        return Ok(result);
    }

    pub fn elementwise_multiply(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err("Matrices must be the same size".into());
        }
        let mut result = Matrix::new(a.rows, a.cols);
        let cols = a.cols;

        a.data
            .as_slice()
            .par_chunks(cols)
            .zip(b.data.as_slice().par_chunks(cols))
            .zip(result.data.as_mut_slice().par_chunks_mut(cols))
            .for_each(|((ar, br), rr)| {
                for i in 0..cols {
                    rr[i] = ar[i] * br[i];
                }
            });

        return Ok(result);
    }

    pub fn multiply(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        if a.cols != b.rows {
            return Err("Matrices must be the same size".into());
        }
        let mut result = Matrix::new(a.rows, b.cols);
        let cols = b.cols;
        let inner = a.cols;

        result
            .data
            .as_mut_slice()
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(i, row_out)| {
                for j in 0..cols {
                    let mut sum = 0.0;
                    for k in 0..inner {
                        sum += a.data[(i, k)] * b.data[(k, j)];
                    }
                    row_out[j] = sum;
                }
            });

        return Ok(result);
    }

    pub fn scale(&mut self, scalar: f64) {
        let cols = self.cols;

        self.data
            .as_mut_slice()
            .par_chunks_mut(cols)
            .for_each(|row| {
                for x in row {
                    *x *= scalar;
                }
            });
    }

    pub fn map(&mut self, func: fn(f64) -> f64) {
        let cols = self.cols;

        self.data
            .as_mut_slice()
            .par_chunks_mut(cols)
            .for_each(|row| {
                for x in row {
                    *x = func(*x);
                }
            });
    }

    pub fn static_map(matrix: &Matrix, func: fn(f64) -> f64) -> Matrix {
        let mut result = Matrix::new(matrix.rows, matrix.cols);
        let cols = matrix.cols;

        matrix
            .data
            .as_slice()
            .par_chunks(cols)
            .zip(result.data.as_mut_slice().par_chunks_mut(cols))
            .for_each(|(src_row, dst_row)| {
                for i in 0..cols {
                    dst_row[i] = func(src_row[i]);
                }
            });

        return result;
    }
}
