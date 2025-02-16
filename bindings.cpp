#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "Matrix.hpp"
#include "SquareMatrix.hpp"
#include "DiagonalMatrix.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

// Funzione helper per dichiarare la classe Matrix per un tipo T
template <typename T>
void declareMatrix(py::module &m, const std::string &name) {
    py::class_<Matrix<T>>(m, name.c_str())
        .def(py::init<size_t, size_t>(), "rows"_a = 0, "cols"_a = 0)
        .def("rows", &Matrix<T>::rows)
        .def("cols", &Matrix<T>::cols)
        .def("print", &Matrix<T>::print)
        // Overload per accedere agli elementi con la sintassi m[i, j]
        .def("__getitem__", [](const Matrix<T> &mat, std::pair<size_t, size_t> index) -> T {
            return mat(index.first, index.second);
        })
        .def("__setitem__", [](Matrix<T> &mat, std::pair<size_t, size_t> index, T value) {
            mat(index.first, index.second) = value;
        })
        // Add new overload for setting entire rows
        .def("__setitem__", [](Matrix<T> &mat, size_t row, const std::vector<T>& values) {
            if (values.size() != mat.cols()) {
                throw std::runtime_error("Vector size must match matrix columns");
            }
            for (size_t j = 0; j < mat.cols(); ++j) {
                mat(row, j) = values[j];
            }
        })
        // Operatore + e +=
        .def("__add__", [](const Matrix<T>& a, const Matrix<T>& b) { return a + b; }, py::is_operator())
        .def("__iadd__", [](Matrix<T>& a, const Matrix<T>& b) -> Matrix<T>& { return a += b; }, py::is_operator())
        // Operatore - e -=
        .def("__sub__", [](const Matrix<T>& a, const Matrix<T>& b) { return a - b; }, py::is_operator())
        .def("__isub__", [](Matrix<T>& a, const Matrix<T>& b) -> Matrix<T>& { return a -= b; }, py::is_operator())
        // Operatore *: versione matrice-matrice e matrice-scalare.
        // Usando py::overload_cast per distinguere le overload.
        .def("__mul__", py::overload_cast<const Matrix<T>&>(&Matrix<T>::operator*, py::const_), py::is_operator())
        .def("__mul__", py::overload_cast<const T&>(&Matrix<T>::operator*, py::const_), py::is_operator())
        // Per __rmul__ per la moltiplicazione scalare da sinistra
        .def("__rmul__", [](const Matrix<T>& a, const T& scalar) { return a * scalar; }, py::is_operator())
        // Operatore *=: versione matrice-matrice e matrice-scalare (usando lambda)
        .def("__imul__", [](Matrix<T>& a, const Matrix<T>& b) -> Matrix<T>& { return a *= b; }, py::is_operator())
        .def("__imul__", [](Matrix<T>& a, const T& scalar) -> Matrix<T>& { return a *= scalar; }, py::is_operator());
}

// Funzione helper per dichiarare la classe SquareMatrix per un tipo T
template <typename T>
void declareSquareMatrix(py::module &m, const std::string &name) {
    py::class_<SquareMatrix<T>, Matrix<T>>(m, name.c_str())
        .def(py::init<size_t>(), "size"_a = 0)
        .def("invert", &SquareMatrix<T>::invert, "Invert the square matrix using Gauss-Jordan elimination")
        .def("determinant", &SquareMatrix<T>::determinant, "Calculate the determinant of the square matrix");
}

// Funzione helper per dichiarare la classe DiagonalMatrix per un tipo T
template <typename T>
void declareDiagonalMatrix(py::module &m, const std::string &name) {
    py::class_<DiagonalMatrix<T>, SquareMatrix<T>>(m, name.c_str())
        .def(py::init<size_t>(), "size"_a = 0)
        .def("invert", &DiagonalMatrix<T>::invert, "Invert the diagonal matrix")
        .def("__setitem__", [](DiagonalMatrix<T> &mat, size_t i, const std::vector<T>& values) {
            if (values.size() != mat.cols()) {
                throw std::runtime_error("Vector size must match matrix size");
            }
            mat(i, i) = values[i];
        })
        .def("__setitem__", [](DiagonalMatrix<T> &mat, std::pair<size_t, size_t> index, T value) {
            if (index.first != index.second && value != 0) {
                throw std::runtime_error("Cannot modify off-diagonal elements in a diagonal matrix");
            }
            if (index.first == index.second) {
                mat(index.first, index.second) = value;
            }
        });
}

PYBIND11_MODULE(matrix, m) {
    m.doc() = "Bindings per le classi Matrix, SquareMatrix e DiagonalMatrix";

    // Dichiarazione delle versioni di Matrix per alcuni tipi
    declareMatrix<int>(m, "MatrixInt");
    declareMatrix<float>(m, "MatrixFloat");
    declareMatrix<double>(m, "MatrixDouble");
    declareMatrix<std::complex<double>>(m, "MatrixComplex");

    // Dichiarazione delle versioni di SquareMatrix e DiagonalMatrix
    // Per operazioni di inversione è consigliabile usare tipi a virgola mobile o complessi.
    declareSquareMatrix<double>(m, "SquareMatrixDouble");
    declareDiagonalMatrix<double>(m, "DiagonalMatrixDouble");
}
