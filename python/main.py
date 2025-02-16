import sys
import time
import numpy as np
import matrix

def to_numpy(m):
    """Convert a MatrixDouble (or SquareMatrixDouble) to a numpy array."""
    rows = m.rows()
    cols = m.cols()
    arr = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            arr[i, j] = m[i, j]
    return arr

def test_matrix_int():
    print("=== Testing MatrixInt ===")
    
    # Create a 3x3 matrix and fill it with the sum of indices.
    m1 = matrix.MatrixInt(3, 3)
    for i in range(3):
        for j in range(3):
            m1[i, j] = i + j
    print("Matrix m1:")
    m1.print()
    print()

    # Create another matrix and fill it with explicit rows.
    m2 = matrix.MatrixInt(3, 3)
    m2[0] = [1, 2, 3]
    m2[1] = [4, 5, 6]
    m2[2] = [7, 8, 9]
    print("Matrix m2:")
    m2.print()
    print()

    # Matrix multiplication
    m3 = m1 * m2
    print("Matrix m3 = m1 * m2:")
    m3.print()
    print()

    # Matrix addition
    m4 = m1 + m2
    print("Matrix m4 = m1 + m2:")
    m4.print()
    print()

    # Matrix subtraction
    m5 = m2 - m1
    print("Matrix m5 = m2 - m1:")
    m5.print()
    print()

    # In-place scalar multiplication
    m1 *= 2
    print("Matrix m1 after m1 *= 2:")
    m1.print()
    print()

def test_matrix_double():
    print("=== Testing MatrixDouble ===")
    
    # Create a MatrixDouble 3x3 and fill it with the product of indices.
    m = matrix.MatrixDouble(3, 3)
    for i in range(3):
        for j in range(3):
            m[i, j] = float(i * j)
    print("Matrix m (initial):")
    m.print()
    print()

    # Scalar multiplication
    m2 = m * 3.0
    print("Matrix m2 = m * 3.0:")
    m2.print()
    print()

def test_square_matrix():
    print("=== Testing SquareMatrixDouble ===")
    
    # Create an invertible SquareMatrixDouble 3x3.
    sm = matrix.SquareMatrixDouble(3)
    sm[0] = [4.0, 7.0, 2.0]
    sm[1] = [3.0, 5.0, 1.0]
    sm[2] = [2.0, 1.0, 3.0]
    print("SquareMatrix sm (before inversion):")
    sm.print()
    print()

    # Invert the matrix (in-place) and profile the inversion time.
    start_time = time.perf_counter()
    sm.invert()
    end_time = time.perf_counter()
    print("SquareMatrix sm (after inversion):")
    sm.print()
    print(f"Inversion time: {(end_time - start_time)*1e3:.3f} ms")
    print()

def test_diagonal_matrix():
    print("=== Testing DiagonalMatrixDouble ===")
    
    # Create a DiagonalMatrixDouble 3x3.
    dm = matrix.DiagonalMatrixDouble(3)
    dm[0] = [2.0, 0.0, 0.0]
    dm[1] = [0.0, 3.0, 0.0]
    dm[2] = [0.0, 0.0, 4.0]
    print("DiagonalMatrix dm (original):")
    dm.print()
    print()

    # In-place scalar multiplication.
    dm *= 2.0
    print("DiagonalMatrix dm after dm *= 2.0:")
    dm.print()
    print()

    # Invert the diagonal matrix (in-place) and profile inversion time.
    start_time = time.perf_counter()
    dm.invert()
    end_time = time.perf_counter()
    print("DiagonalMatrix dm after inversion:")
    dm.print()
    print(f"Inversion time: {(end_time - start_time)*1e3:.3f} ms")
    print()

def numpy_comparison():
    print("=== Comparing with NumPy ===")
    
    # Create a 3x3 matrix filled with random numbers.
    np.random.seed(0)
    a = np.random.rand(3, 3)
    print("NumPy matrix a:")
    print(a)
    print()
    
    # Convert the NumPy matrix to SquareMatrixDouble.
    m = matrix.SquareMatrixDouble(3)
    for i in range(3):
        for j in range(3):
            m[i, j] = a[i, j]
    print("SquareMatrixDouble m:")
    m.print()
    print()
    
    # Convert SquareMatrixDouble back to NumPy.
    b = to_numpy(m)
    print("Converted NumPy matrix b:")
    print(b)
    print()
    
    # Check equality.
    print("Are a and b equal?", np.allclose(a, b))
    print()

    # Benchmark matrix multiplication.
    start_np = time.perf_counter()
    c = np.dot(a, a)
    end_np = time.perf_counter()
    
    start_m = time.perf_counter()
    m2 = m * m
    end_m = time.perf_counter()
    
    print("NumPy matrix c = a @ a:")
    print(c)
    print()
    print("SquareMatrixDouble m2 = m * m:")
    m2.print()
    print()
    
    # Convert m2 to NumPy for comparison.
    c_from_m = to_numpy(m2)
    print("Are c and m2 equal?", np.allclose(c, c_from_m))
    print()
    print(f"NumPy multiplication time: {(end_np - start_np)*1e3:.3f} ms")
    print(f"MatrixDouble multiplication time: {(end_m - start_m)*1e3:.3f} ms")
    print()
    
    # Benchmark matrix inversion.
    start_np_inv = time.perf_counter()
    d = np.linalg.inv(a)
    end_np_inv = time.perf_counter()
    
    start_m_inv = time.perf_counter()
    m.invert()  # invert in-place
    end_m_inv = time.perf_counter()
    
    print("NumPy matrix d = inv(a):")
    print(d)
    print()
    print("SquareMatrixDouble m after inversion:")
    m.print()
    print()
    
    # Compare inversion results.
    m_inv_np = to_numpy(m)
    print("Are d and m (after inversion) equal?", np.allclose(d, m_inv_np))
    print()
    print(f"NumPy inversion time: {(end_np_inv - start_np_inv)*1e3:.3f} ms")
    print(f"MatrixDouble inversion time: {(end_m_inv - start_m_inv)*1e3:.3f} ms")
    print()

def test_determinant():
    print("=== Testing Determinant Calculation ===")
    
    # Create a 3x3 matrix with a known determinant.
    m = matrix.SquareMatrixDouble(3)
    m[0] = [1.0, 2.0, 3.0]
    m[1] = [4.0, 5.0, 6.0]
    m[2] = [7.0, 8.0, 9.0]
    
    print("Test matrix m:")
    m.print()
    print()
    
    # Calculate determinant using custom implementation.
    start_m = time.perf_counter()
    det_m = m.determinant()
    end_m = time.perf_counter()
    
    # Convert to NumPy and calculate determinant.
    np_m = to_numpy(m)
    start_np = time.perf_counter()
    det_np = np.linalg.det(np_m)
    end_np = time.perf_counter()
    
    print(f"Custom implementation determinant: {det_m}")
    print(f"NumPy determinant: {det_np}")
    print(f"Are determinants equal? {abs(det_m - det_np) < 1e-10}")
    print()
    print(f"Custom implementation time: {(end_m - start_m)*1e3:.3f} ms")
    print(f"NumPy time: {(end_np - start_np)*1e3:.3f} ms")
    print()

if __name__ == "__main__":
    test_matrix_int()
    test_matrix_double()
    test_square_matrix()
    test_diagonal_matrix()
    numpy_comparison()
    test_determinant()
