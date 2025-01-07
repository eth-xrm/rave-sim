// Copyright (c) 2024, ETH Zurich

#ifndef _FAST_WAVE_OPTICAL_ELEMENT_HPP
#define _FAST_WAVE_OPTICAL_ELEMENT_HPP

#include <array>
#include <stdint.h>
#include <vector>

#include <types.hpp>

struct Material {
    std::string name;
    double density;
};

enum class OpticalElementType {
    Grating,
    EnvGrating,
    Sample,
};

struct OpticalElement {
    double z_start;
    std::vector<double> x_positions;
    OpticalElementType type;

    /// Z distance from the start of the element to the end of the element
    virtual double total_thickness() const = 0;

    /// How many history entries will be added for this element?
    virtual std::size_t nr_history_entries() const = 0;
};

struct Grating : public OpticalElement {
    double thickness;
    double pitch;
    std::array<double, 2> dc;
    int nr_steps;
    double substrate_thickness;
    Complex<double> deltabeta_a;
    Complex<double> deltabeta_b;
    Complex<double> deltabeta_substrate;

    double total_thickness() const override { return thickness + substrate_thickness; }

    std::size_t nr_history_entries() const override { return nr_steps; }
};

struct EnvGrating : public OpticalElement {
    double thickness;
    double pitch0;
    double pitch1;
    std::array<double, 2> dc0;
    std::array<double, 2> dc1;
    int nr_steps;
    double substrate_thickness;
    Complex<double> deltabeta_a;
    Complex<double> deltabeta_b;
    Complex<double> deltabeta_substrate;

    double total_thickness() const override { return thickness + substrate_thickness; }

    std::size_t nr_history_entries() const override { return nr_steps; }
};

struct Sample : public OpticalElement {
    std::vector<uint8_t> grid; // linearized: constant z entries are contiguous
    std::size_t x_len;
    std::size_t z_len;
    double pixel_size_x;                     // in metres
    double pixel_size_z;                     // in metres
    std::vector<Complex<double>> deltabetas; // including vacuum as first entry

    double total_thickness() const override { return pixel_size_z * z_len; }
    std::size_t nr_history_entries() const override { return z_len; }
};

#endif // _FAST_WAVE_OPTICAL_ELEMENT_HPP
