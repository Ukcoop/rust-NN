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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_zero_matrix() {
        let m = Matrix::new(3, 4);

        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert!(m.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_from_and_to_vector() {
        let v = vec![1.0, 2.0, 3.0];
        let m = Matrix::from_vector(&v);

        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 1);
        assert_eq!(m.to_vector(), v);
    }

    #[test]
    fn test_transpose() {
        let mut m = Matrix::new(2, 3);
        m.data[(0, 0)] = 1.0;
        m.data[(0, 1)] = 2.0;
        m.data[(0, 2)] = 3.0;
        m.data[(1, 0)] = 4.0;
        m.data[(1, 1)] = 5.0;
        m.data[(1, 2)] = 6.0;

        let t = Matrix::transpose(&m);
        assert_eq!((t.rows, t.cols), (3, 2));
        assert_eq!(t.data[(0, 0)], 1.0);
        assert_eq!(t.data[(2, 1)], 6.0);
    }

    #[test]
    fn test_add_and_add_number() {
        let mut m = Matrix::new(2, 3);
        m.data[(0, 0)] = 1.0;
        m.data[(0, 1)] = 2.0;
        m.data[(0, 2)] = 3.0;
        m.data[(1, 0)] = 4.0;
        m.data[(1, 1)] = 5.0;
        m.data[(1, 2)] = 6.0;

        let mut n = Matrix::new(2, 3);
        n.data[(0, 0)] = 1.0;
        n.data[(0, 1)] = 2.0;
        n.data[(0, 2)] = 3.0;
        n.data[(1, 0)] = 4.0;
        n.data[(1, 1)] = 5.0;
        n.data[(1, 2)] = 6.0;

        Matrix::add(&mut m, &n).unwrap();
        assert_eq!(m.to_vector(), vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);

        m.add_number(1.0);
        assert_eq!(m.to_vector(), vec![3.0, 9.0, 5.0, 11.0, 7.0, 13.0]);
    }

    #[test]
    fn test_subtract_and_elementwise_multiply() {
        let a = Matrix::from_vector(&[5.0, 6.0, 7.0, 8.0]);
        let b = Matrix::from_vector(&[1.0, 2.0, 3.0, 4.0]);

        let c = Matrix::subtract(&a, &b).unwrap();
        assert_eq!(c.to_vector(), vec![4.0, 4.0, 4.0, 4.0]);

        let d = Matrix::elementwise_multiply(&a, &b).unwrap();
        assert_eq!(d.to_vector(), vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_matrix_multiply() {
        let mut a = Matrix::new(2, 3);
        let mut b = Matrix::new(3, 2);

        for i in 0..2 {
            for j in 0..3 {
                a.data[(i, j)] = (i * 3 + j + 1) as f64;
            }
        }
        for i in 0..3 {
            for j in 0..2 {
                b.data[(i, j)] = (7 + i * 2 + j) as f64;
            }
        }

        let prod = Matrix::multiply(&a, &b).unwrap();

        assert_eq!(prod.data[(0, 0)], 58.0);
        assert_eq!(prod.data[(1, 1)], 154.0);
    }

    #[test]
    fn test_scale_map_and_static_map() {
        let mut m = Matrix::from_vector(&[1.0, -2.0, 3.0]);
        m.scale(2.0);
        assert_eq!(m.to_vector(), vec![2.0, -4.0, 6.0]);

        m.map(f64::abs);
        assert_eq!(m.to_vector(), vec![2.0, 4.0, 6.0]);

        let sq = Matrix::static_map(&m, |x| x * x);
        assert_eq!(sq.to_vector(), vec![4.0, 16.0, 36.0]);
    }

    #[test]
    fn test_randomize_changes_values() {
        let mut m = Matrix::new(3, 3);
        m.randomize();

        assert!(m.data.iter().any(|&x| x != 0.0));
    }
}
