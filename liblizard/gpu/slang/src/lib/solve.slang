// Gauss-Jordan elimination with partial pivoting on N in place: R rows, the R columns of the system and then its
// right-hand sides (C columns in all), unrolled so that every index is a constant (no private array is indexed at run
// time: the S26's Adreno returned zeros from one); ok turns false where a pivot is under eps. Row r's solution is then
// N[r][R..] / N[r][r].
void gaussJordan<let R : int, let C : int>(inout float N[R][C], float eps, inout bool ok) {
  [ForceUnroll] for (int col = 0; col < R; col++) {
    int piv = col;
    float pv = abs(N[col][col]);
    [ForceUnroll] for (int r = col + 1; r < R; r++) { float a = abs(N[r][col]); piv = a > pv ? r : piv; pv = max(pv, a); }
    [ForceUnroll] for (int r = col + 1; r < R; r++) {
      bool sw = piv == r;
      [ForceUnroll] for (int j = col; j < C; j++) { float a = N[col][j]; float b = N[r][j]; N[col][j] = sw ? b : a; N[r][j] = sw ? a : b; }
    }
    ok = ok && abs(N[col][col]) >= eps;
    [ForceUnroll] for (int r = 0; r < R; r++) {
      if (r != col) {
        float fc = N[r][col] / N[col][col];
        [ForceUnroll] for (int j = col; j < C; j++) N[r][j] -= fc * N[col][j];
      }
    }
  }
}
