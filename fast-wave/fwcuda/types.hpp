// Copyright (c) 2024, ETH Zurich

#ifndef _FAST_WAVE_TYPES_HPP
#define _FAST_WAVE_TYPES_HPP

#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <iostream>

// We differentiate between the complex type used on the GPU vs the one used on the CPU
//
// std::copmlex has the advantage that it overloads the + and * operators, but to access
// the components we need to use .real() and .imag() which are not compiled as `__host__ __device__`
// so we can't use them from CUDA.
//
// The two complex types have the same layout so we can memcopy between them without problems.

/// Complex numbers on the CPU
template <typename S> using Complex = std::complex<S>;

template <typename S> struct DevComplexHelper {
    using type = void;
};
template <> struct DevComplexHelper<double> {
    using type = cuDoubleComplex;
};
template <> struct DevComplexHelper<float> {
    using type = cufftComplex;
};

/// Complex numbers on the GPU
template <typename S> using DevComplex = typename DevComplexHelper<S>::type;

static_assert(sizeof(Complex<float>) == sizeof(DevComplex<float>));
static_assert(sizeof(Complex<double>) == sizeof(DevComplex<double>));

struct SimParams {
    int N;
    double dx;
    double z_detector;
    double detector_size;
    double detector_pixel_size_x;
    double detector_pixel_size_y;
    double wl;
};

/// convert a cpu complex to a gpu complex
template <typename S> [[nodiscard]] inline DevComplex<S> c2dc(Complex<S> c) {
    return DevComplex<S>{c.real(), c.imag()};
}

inline void check_cuda_result(const char *operation, cudaError res) {
    if (res != cudaSuccess) {
        const auto msg = cudaGetErrorString(res);
        std::clog << "cuda error in operation " << operation << ": " << msg << "\n";
        exit(1);
    }
}

inline void check_cufft_result(const char *operation, cufftResult res) {
    if (res != CUFFT_SUCCESS) {
        std::clog << "cuFFT error in operation " << operation << ": " << res << '\n';
        exit(1);
    }
}

#endif // _FAST_WAVE_TYPES_HPP
