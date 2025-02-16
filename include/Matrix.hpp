#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>
#include <stdexcept>

template <typename T>
class Matrix {
public:
    Matrix(size_t rows = 0, size_t cols = 0) : data(rows, std::vector<T>(cols)) {}

    // get the number of rows
    size_t rows() const {
        return data.size();
    }

    // get the number of columns
    size_t cols() const {
        return data.empty() ? 0 : data[0].size();
    }

    // get the element at position (i, j)
    virtual T& operator()(size_t i, size_t j) {
        check_bounds(i, j);
        return data[i][j];
    }

    // get the element at position (i, j) (const version)
    virtual const T& operator()(size_t i, size_t j) const {
        check_bounds(i, j);
        return data[i][j];
    }

    // get the row at position i
    std::vector<T>& operator[](size_t i) {
        if (i >= rows()) {
            throw std::out_of_range("Row index out of bounds");
        }
        return data[i];
    }

    // get the row at position i (const version)
    const std::vector<T>& operator[](size_t i) const {
        if (i >= rows()) {
            throw std::out_of_range("Row index out of bounds");
        }
        return data[i];
    }

    // print the matrix
    void print() const {
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    //? ------- Operators ------- ?//

    // addition operator
    Matrix<T> operator+(const Matrix<T>& rhs) const {
        if (rows() != rhs.rows() || cols() != rhs.cols()) {
            throw std::invalid_argument("Matrix dimensions mismatch for addition");
        }
        Matrix<T> result(rows(), cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(i, j) = data[i][j] + rhs(i, j);
            }
        }
        return result;
    }

    // addition-assignment operator
    Matrix<T>& operator+=(const Matrix<T>& rhs) {
        if (rows() != rhs.rows() || cols() != rhs.cols()) {
            throw std::invalid_argument("Matrix dimensions mismatch for addition");
        }
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                data[i][j] += rhs(i, j);
            }
        }
        return *this;
    }
    
    // subtraction operator
    Matrix<T> operator-(const Matrix<T>& rhs) const {
        if (rows() != rhs.rows() || cols() != rhs.cols()) {
            throw std::invalid_argument("Matrix dimensions mismatch for subtraction");
        }
        Matrix<T> result(rows(), cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(i, j) = data[i][j] - rhs(i, j);
            }
        }
        return result;
    }

    // subtraction-assignment operator
    Matrix<T>& operator-=(const Matrix<T>& rhs) {
        if (rows() != rhs.rows() || cols() != rhs.cols()) {
            throw std::invalid_argument("Matrix dimensions mismatch for subtraction");
        }
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                data[i][j] -= rhs(i, j);
            }
        }
        return *this;
    }

    //? virtual methods for multiplication (to be overridden by derived classes)

    // multiplication operator
    virtual Matrix<T> operator*(const Matrix<T>& rhs) const {
        if (cols() != rhs.rows()) {
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
        }
        
        Matrix<T> result(rows(), rhs.cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < rhs.cols(); ++j) {
                for (size_t k = 0; k < cols(); ++k) {
                    result(i, j) += data[i][k] * rhs(k, j);
                }
            }
        }
        return result;
    }

    // scalar multiplication operator
    virtual Matrix<T> operator*(const T& scalar) const {
        Matrix<T> result(rows(), cols());
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                result(i, j) = data[i][j] * scalar;
            }
        }
        return result;
    }

    // multiplication-assignment operator
    virtual Matrix<T>& operator*=(const Matrix<T>& rhs) {
        if (cols() != rhs.rows()) {
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
        }
        
        std::vector<std::vector<T>> result(rows(), std::vector<T>(rhs.cols(), 0));
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < rhs.cols(); ++j) {
                for (size_t k = 0; k < cols(); ++k) {
                    result[i][j] += data[i][k] * rhs(k, j);
                }
            }
        }
        data = result;
        return *this;
    }

    // scalar multiplication-assignment operator
    virtual Matrix<T>& operator*=(const T& scalar) {
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                data[i][j] *= scalar;
            }
        }
        return *this;
    }

private:
    // check if the indices are within bounds
    void check_bounds(size_t i, size_t j) const {
        if (i >= rows() || j >= cols()) {
            throw std::out_of_range("Matrix indices out of bounds");
        }
    }

protected:
    // matrix of data
    std::vector<std::vector<T>> data;
};

#endif // MATRIX_HPP