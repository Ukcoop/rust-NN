use std::error::Error;

#[derive(Debug)]
pub struct Matrix {
    pub rows: u32,
    pub cols: u32,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(rows: u32, cols: u32) -> Matrix {
        let mut data = vec![];
        for _ in 0..rows {
            let mut row = vec![];
            for _ in 0..cols {
                row.push(0.0);
            }
            data.push(row);
        }

        return Matrix { rows, cols, data };
    }

    pub fn from_vector(vector: &Vec<f64>) -> Matrix {
        let mut matrix = Matrix::new(vector.len() as u32, 1);
        for i in 0..vector.len() {
            matrix.data[i][0] = vector[i];
        }
        return matrix;
    }

    pub fn to_vector(&self) -> Vec<f64> {
        let mut vector = vec![];
        for i in 0..self.rows {
            for j in 0..self.cols {
                vector.push(self.data[i as usize][j as usize]);
            }
        }
        return vector;
    }

    pub fn transpose(matrix: &Matrix) -> Matrix {
        let mut result = Matrix::new(matrix.cols, matrix.rows);

        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                result.data[j as usize][i as usize] = matrix.data[i as usize][j as usize];
            }
        }

        return result;
    }

    pub fn randomize(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i as usize][j as usize] = rand::random_range(-1.0..1.0);
            }
        }
    }

    pub fn add(&mut self, other: &Matrix) -> Result<(), Box<dyn Error>> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrices must be the same size".into());
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i as usize][j as usize] += other.data[i as usize][j as usize];
            }
        }

        return Ok(());
    }

    pub fn add_number(&mut self, number: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i as usize][j as usize] += number;
            }
        }
    }

    pub fn subtract(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err("Matrices must be the same size".into());
        }

        let mut result = Matrix::new(a.rows, a.cols);
        for i in 0..a.rows {
            for j in 0..a.cols {
                result.data[i as usize][j as usize] =
                    a.data[i as usize][j as usize] - b.data[i as usize][j as usize];
            }
        }
        return Ok(result);
    }

    pub fn elementwise_multiply(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err("Matrices must be the same size".into());
        }

        let mut result = Matrix::new(a.rows, a.cols);
        for i in 0..a.rows {
            for j in 0..a.cols {
                result.data[i as usize][j as usize] =
                    a.data[i as usize][j as usize] * b.data[i as usize][j as usize];
            }
        }
        return Ok(result);
    }

    pub fn multiply(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        if a.cols != b.rows {
            return Err("Matrices must be the same size".into());
        }
        let mut result = Matrix::new(a.rows, b.cols);
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut sum = 0.0;
                for k in 0..a.cols {
                    sum += a.data[i as usize][k as usize] * b.data[k as usize][j as usize];
                }
                result.data[i as usize][j as usize] = sum;
            }
        }
        return Ok(result);
    }

    pub fn scale(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i as usize][j as usize] *= scalar;
            }
        }
    }

    pub fn map(&mut self, func: fn(f64) -> f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i as usize][j as usize] = func(self.data[i as usize][j as usize]);
            }
        }
    }

    pub fn static_map(matrix: &Matrix, func: fn(f64) -> f64) -> Matrix {
        let mut result = Matrix::new(matrix.rows, matrix.cols);
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                result.data[i as usize][j as usize] = func(matrix.data[i as usize][j as usize]);
            }
        }
        return result;
    }
}
