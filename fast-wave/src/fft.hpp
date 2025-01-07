// Copyright (c) 2024, ETH Zurich

#ifndef _FAST_WAVE_FFT_HPP
#define _FAST_WAVE_FFT_HPP

#include <types.hpp>

template<typename S>
class FFT {
    cufftHandle fft_plan;

    std::size_t N;

    FFT(const FFT &) = delete;
    FFT &operator=(const FFT &) = delete;

  public:
    explicit FFT(std::size_t N);

    ~FFT();

    void forward(DevComplex<S> *d_u, DevComplex<S> *d_U) const;

    void inverse(DevComplex<S> *d_U, DevComplex<S> *d_u) const;
};

#endif // _FAST_WAVE_FFT_HPP
