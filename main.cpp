#include <iostream>
#include <vector>
#include <stdexcept>
#include <complex>
#include "Matrix.hpp"
#include "SquareMatrix.hpp"
#include "DiagonalMatrix.hpp"

// Template test for basic Matrix multiplication.
template<typename T>
void testMatrixMultiplication(const std::string& typeName) {
    std::cout << "==== Test: Matrix Multiplication (" << typeName << ") ====\n";
    Matrix<T> m1(3, 3);
    m1[0] = {T(1), T(2), T(3)};
    m1[1] = {T(4), T(5), T(6)};
    m1[2] = {T(7), T(8), T(9)};

    Matrix<T> m2(3, 3);
    m2[0] = {T(9), T(8), T(7)};
    m2[1] = {T(6), T(5), T(4)};
    m2[2] = {T(3), T(2), T(1)};

    std::cout << "Matrix m1:\n";
    m1.print();
    std::cout << "Matrix m2:\n";
    m2.print();

    m1 *= m2;
    std::cout << "Result of m1 *= m2:\n";
    m1.print();
    std::cout << "\n";
}

// Template test for SquareMatrix operations.
template<typename T>
void testSquareMatrixOperations(const std::string& typeName) {
    std::cout << "==== Test: SquareMatrix Operations (" << typeName << ") ====\n";

    // Create a 3x3 square matrix.
    SquareMatrix<T> sm(3);
    sm[0] = {T(2), T(0), T(1)};
    sm[1] = {T(3), T(0), T(0)};
    sm[2] = {T(5), T(1), T(1)};

    std::cout << "Original SquareMatrix sm:\n";
    sm.print();

    // Copy constructor test
    SquareMatrix<T> smCopy(sm);
    std::cout << "SquareMatrix smCopy (copy of sm):\n";
    smCopy.print();

    // Move constructor test
    SquareMatrix<T> smMove(std::move(smCopy));
    std::cout << "SquareMatrix smMove (moved from smCopy):\n";
    smMove.print();

    // Downcast constructor test:
    Matrix<T> generic(3, 3);
    generic[0] = {T(2), T(0), T(1)};
    generic[1] = {T(3), T(0), T(0)};
    generic[2] = {T(5), T(1), T(1)};
    try {
        SquareMatrix<T> smDowncast(generic);
        std::cout << "SquareMatrix smDowncast (constructed from generic Matrix):\n";
        smDowncast.print();
    } catch (std::exception& e) {
        std::cerr << "Downcast constructor error: " << e.what() << "\n";
    }

    // Test multiplication for SquareMatrix.
    SquareMatrix<T> sm2(3);
    sm2[0] = {T(1), T(1), T(1)};
    sm2[1] = {T(0), T(1), T(2)};
    sm2[2] = {T(1), T(0), T(1)};
    std::cout << "SquareMatrix sm2:\n";
    sm2.print();

    sm *= sm2;
    std::cout << "After sm *= sm2:\n";
    sm.print();

    // Test determinant for SquareMatrix.
    std::cout << "Determinant of SquareMatrix sm: " << sm.determinant() << "\n\n";

    // Inversion test (only if T supports division appropriately)
    try {
        SquareMatrix<T> smInv(3);
        smInv[0] = {T(4), T(7), T(2)};
        smInv[1] = {T(3), T(5), T(1)};
        smInv[2] = {T(2), T(1), T(3)};
        std::cout << "Matrix before inversion (smInv):\n";
        smInv.print();
        smInv.invert();
        std::cout << "Inverted matrix (smInv):\n";
        smInv.print();
    } catch (std::exception& e) {
        std::cerr << "Inversion error: " << e.what() << "\n";
    }
    std::cout << "\n";
}

// Template test for DiagonalMatrix operations.
template<typename T>
void testDiagonalMatrixOperations(const std::string& typeName) {
    std::cout << "==== Test: DiagonalMatrix Operations (" << typeName << ") ====\n";
    // Assuming DiagonalMatrix inherits constructors (using SquareMatrix<T>::SquareMatrix)
    DiagonalMatrix<T> dm(3);
    dm[0] = {T(5), T(0), T(0)};
    dm[1] = {T(0), T(3), T(0)};
    dm[2] = {T(0), T(0), T(2)};
    std::cout << "Original DiagonalMatrix dm:\n";
    dm.print();

    DiagonalMatrix<T> dm2(3);
    dm2[0] = {T(1), T(0), T(0)};
    dm2[1] = {T(0), T(2), T(0)};
    dm2[2] = {T(0), T(0), T(3)};
    std::cout << "DiagonalMatrix dm2:\n";
    dm2.print();

    dm *= dm2;
    std::cout << "After dm *= dm2 (diagonal multiplication):\n";
    dm.print();

    // Test scalar multiplication.
    dm *= T(2);
    std::cout << "After scalar multiplication (dm *= 2):\n";
    dm.print();

    // Test determinant for DiagonalMatrix.
    std::cout << "Determinant of DiagonalMatrix dm: " << dm.determinant() << "\n";

    // Test inversion (again, only if T supports proper division)
    try {
        dm.invert();
        std::cout << "After inversion of dm:\n";
        dm.print();
    } catch (std::exception& e) {
        std::cerr << "DiagonalMatrix inversion error: " << e.what() << "\n";
    }
    std::cout << "\n";
}

int main() {
    try {
        // Testing with int (inversion might not be meaningful for integers).
        testMatrixMultiplication<int>("int");
        testSquareMatrixOperations<int>("int");
        testDiagonalMatrixOperations<int>("int");

        // Testing with float.
        testMatrixMultiplication<float>("float");
        testSquareMatrixOperations<float>("float");
        testDiagonalMatrixOperations<float>("float");

        // Testing with double.
        testMatrixMultiplication<double>("double");
        testSquareMatrixOperations<double>("double");
        testDiagonalMatrixOperations<double>("double");

        // Testing with std::complex<double>.
        testMatrixMultiplication<std::complex<double>>("std::complex<double>");
        testSquareMatrixOperations<std::complex<double>>("std::complex<double>");
        testDiagonalMatrixOperations<std::complex<double>>("std::complex<double>");

    } catch (std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    }
    return 0;
}
