#ifndef DIAGONAL_MATRIX_HPP
#define DIAGONAL_MATRIX_HPP

#include "Matrix.hpp"
#include <stdexcept>

template <typename T>
class DiagonalMatrix : public SquareMatrix<T> {
public:
    DiagonalMatrix(size_t size = 0) : SquareMatrix<T>(size) {}

    using Matrix<T>::operator[];

    T& operator()(size_t i, size_t j) override {
        if (i == j) {
            return this->data[i][j];
        }
        throw std::logic_error("Cannot modify off-diagonal elements in a diagonal matrix");
    }

// get the element at position (i, j) (const version)
    const T& operator()(size_t i, size_t j) const override {
        if (i == j) {
            return this->data[i][j];
        }
        static const T zero = T();
        return zero;
    }

    // multiplication operator
    virtual Matrix<T> operator*(const Matrix<T>& rhs) const override {
        if (this->cols() != rhs.rows()) {
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
        }
        DiagonalMatrix<T> result(this->rows());
        for (size_t i = 0; i < this->rows(); ++i) {
            result(i, i) = this->data[i][i] * rhs(i, i);
        }
        return result;
    }

    // scalar multiplication operator
    virtual Matrix<T> operator*(const T& scalar) const override {
        DiagonalMatrix<T> result(this->rows());
        for (size_t i = 0; i < this->rows(); ++i) {
            result(i, i) = this->data[i][i] * scalar;
        }
        return result;
    }

    // multiplication-assignment operator
    DiagonalMatrix<T>& operator*=(const Matrix<T>& rhs) override {
        if (this->cols() != rhs.rows()) {
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
        }
        if (auto diag = dynamic_cast<const DiagonalMatrix<T>*>(&rhs)) {
            for (size_t i = 0; i < this->rows(); ++i) {
                this->data[i][i] *= diag->operator()(i, i);
            }
        } else {
            throw std::invalid_argument("DiagonalMatrix can only be multiplied with another DiagonalMatrix");
        }
        return *this;
    }

    // scalar multiplication-assignment operator
    DiagonalMatrix<T>& operator*=(const T& scalar) override {
        for (size_t i = 0; i < this->rows(); ++i) {
            this->data[i][i] *= scalar;
        }
        return *this;
    }

    // Invert this diagonal matrix by taking the reciprocal of each diagonal element
    DiagonalMatrix<T>& invert() override {
        for (size_t i = 0; i < this->rows(); ++i) {
            if (std::abs(this->data[i][i]) < 1e-10) {
                throw std::runtime_error("Matrix is singular and cannot be inverted");
            }
            this->data[i][i] = T(1) / this->data[i][i];  // Use T(1) instead of integer 1
        }
        return *this;
    }

    // Calculate the determinant of this diagonal matrix
    T determinant() const override {
        T det = T(1);
        for (size_t i = 0; i < this->rows(); ++i) {
            det *= this->data[i][i];
        }
        return det;
    }
};

#endif // DIAGONAL_MATRIX_HPP