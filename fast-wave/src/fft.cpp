// Copyright (c) 2024, ETH Zurich

#include "fft.hpp"
#include <wrappers.hpp>

template <typename S> FFT<S>::FFT(std::size_t N) : N(N) {
    const auto plan_type = std::is_same_v<S, double> ? CUFFT_Z2Z : CUFFT_C2C;
    const auto res = cufftPlan1d(&this->fft_plan, N, plan_type, 1);
    check_cufft_result("cufftPlan1d", res);
}

/// Executes a CUFFT plan with the correct data type. Uses overloading to select
/// the right cufft function.
cufftResult exec_inner(cufftHandle plan, DevComplex<double> *input,
                       DevComplex<double> *output, int direction) {
    return cufftExecZ2Z(plan, input, output, direction);
}
cufftResult exec_inner(cufftHandle plan, DevComplex<float> *input,
                       DevComplex<float> *output, int direction) {
    return cufftExecC2C(plan, input, output, direction);
}

template <typename S> FFT<S>::~FFT() {
    const auto res = cufftDestroy(this->fft_plan);
    check_cufft_result("cufftDestroy", res);
}

template <typename S>
void FFT<S>::forward(DevComplex<S> *d_u, DevComplex<S> *d_U) const {
    const auto res = exec_inner(this->fft_plan, d_u, d_U, CUFFT_FORWARD);
    check_cufft_result("fft forward", res);
}

template <typename S>
void FFT<S>::inverse(DevComplex<S> *d_U, DevComplex<S> *d_u) const {
    const auto res = exec_inner(this->fft_plan, d_U, d_u, CUFFT_INVERSE);
    check_cufft_result("fft inverse", res);

    // todo: the scale step takes as long as one of the fft kernel launches. can
    // we eliminate this? Or at least write a scalar scale function?

    const Complex<S> norm_factor = 1. / this->N;
    scale<S>(d_u, norm_factor, N);
}

// explicit instantiations for float and double:
template class FFT<float>;
template class FFT<double>;
