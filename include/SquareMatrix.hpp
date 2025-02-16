#ifndef SQUARE_MATRIX_HPP
#define SQUARE_MATRIX_HPP

#include "Matrix.hpp"
#include <stdexcept>
#include <algorithm>

template <typename T>
class SquareMatrix : public Matrix<T> {
public:
    SquareMatrix(size_t size = 0) : Matrix<T>(size, size) {}
    // copy constructor
    SquareMatrix(const SquareMatrix<T>& other) : Matrix<T>(other) {}
    // move constructor
    SquareMatrix(SquareMatrix<T>&& other) : Matrix<T>(std::move(other)) {}
    // downcast constructor
    SquareMatrix(const Matrix<T>& other) : Matrix<T>(other) {
        if (other.rows() != other.cols()) {
            throw std::invalid_argument("Matrix is not square");
        }
    }

    using Matrix<T>::operator[];

    // Invert this square matrix using Gauss-Jordan elimination.
    virtual SquareMatrix<T>& invert() {
        size_t n = this->rows();
        // Create an identity matrix of the same size.
        SquareMatrix<T> identity(n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                identity(i, j) = (i == j) ? 1 : 0;
            }
        }

        // Perform Gauss-Jordan elimination.
        for (size_t i = 0; i < n; ++i) {
            // Check for a zero pivot and swap with a lower row if necessary.
            if (std::abs(this->operator()(i, i)) < 1e-10) {
                size_t swapRow = i + 1;
                while (swapRow < n && std::abs(this->operator()(swapRow, i)) < 1e-10) {
                    ++swapRow;
                }
                if (swapRow == n) {
                    throw std::runtime_error("Matrix is singular and cannot be inverted");
                }
                std::swap(this->data[i], this->data[swapRow]);
                std::swap(identity.data[i], identity.data[swapRow]);
            }

            // Normalize the pivot row.
            T pivot = this->operator()(i, i);
            for (size_t j = 0; j < n; ++j) {
                this->operator()(i, j) /= pivot;
                identity(i, j) /= pivot;
            }

            // Eliminate all other entries in the current column.
            for (size_t k = 0; k < n; ++k) {
                if (k == i) continue;
                T factor = this->operator()(k, i);
                for (size_t j = 0; j < n; ++j) {
                    this->operator()(k, j) -= factor * this->operator()(i, j);
                    identity(k, j) -= factor * identity(i, j);
                }
            }
        }

        // Replace the current matrix data with the inverse stored in identity.
        this->data = identity.data;
        return *this;
    }

    virtual T determinant() const {
        size_t n = this->rows();
        if (n == 0) {
            throw std::invalid_argument("Matrix is empty");
        }
        if (n == 1) {
            return this->operator()(0, 0);
        }

        T det = T(0);
        for (size_t i = 0; i < n; ++i) {
            T factor = (i % 2 == 0) ? 1 : -1;
            det += factor * this->operator()(0, i) * this->minor(0, i).determinant();
        }
        return det;
    }

protected:
    // Get the minor matrix obtained by removing the i-th row and j-th column.
    SquareMatrix<T> minor(size_t i, size_t j) const {
        size_t n = this->rows();
        SquareMatrix<T> result(n - 1);
        for (size_t k = 0; k < n; ++k) {
            if (k == i) continue;
            for (size_t l = 0; l < n; ++l) {
                if (l == j) continue;
                result(k < i ? k : k - 1, l < j ? l : l - 1) = this->operator()(k, l);
            }
        }
        return result;
    }

};

#endif // SQUARE_MATRIX_HPP
