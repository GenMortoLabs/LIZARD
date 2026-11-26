// The transform's FFT of N points (pass 1's columns, pass 2's rows), FFT_LANES transforms side by side, a workgroup of
// T = N / 8 lanes each holding eight points: a radix-8 first stage from the lane's points, then the stages N's factors
// give, read from and written to the pass's workgroup buffers (its N, T, tw, LD and ST, fft_body.slang's lanes). The stages: radix 8 wherever it divides what is left, then 4 or 2, then each 5, then each 3
// (N is 8 x 2^a 3^b 5^c); a power-of-two stage through the lanes' registers, an odd one that is not the last through
// registers too (the lanes' points held in a guarded block where T does not divide its N / R), the last odd one in
// place over the buffer (1152 is 8 8 2 3 3, 1280 8 8 4 5). Every loop over a stage, a lane or a point is unrolled, so
// every index into a lane's points is a constant. The twiddles tw[m] = exp(-2 pi i m / N).
static const float FFT_C5A = 0.30901699437494742, FFT_C5B = 0.80901699437494742, FFT_S5A = 0.95105651629515357, FFT_S5B = 0.58778525229247313;
