// Copyright (c) 2024, ETH Zurich

#include <ProgramOptions.hxx>
#include <algorithm>
#include <array>
#include <ctime>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <vector>

#include <config_parsing.hpp>
#include <optical_element.hpp>
#include <types.hpp>

int main(int argc, char **argv) try {
    po::parser parser;

    std::string sim_dir;
    int source_idx;
    std::optional<double> history_dz = std::nullopt;

    parser[""].bind(sim_dir).description("Main simulation directory").single();
    auto &src = parser["source_idx"]
        .abbreviation('s')
        .bind(source_idx)
        .description("The index of the source point to simulate (required)");
    auto &hist = parser["history_dz"].type(po::f64).description("Z-Spacing for history entries (optional)");

    auto &help = parser["help"].abbreviation('h').description("print this help screen");

    if (!parser(argc, argv)) {
        std::cout << parser << '\n';
        return 1;
    }
    if (help.was_set()) {
        std::cout << parser << '\n';
        return 0;
    }
    if (!src.was_set()) {
        std::cerr << "Source index is required\n\n";
        std::cout << parser << '\n';
        return 1;
    }
    if (hist.was_set()) {
        history_dz = hist.get().f64;
    }

    const Config config = parse_config(sim_dir, source_idx);

    run_simulation(config, get_subdir(sim_dir, source_idx), history_dz);
} catch (std::exception const &e) {
    std::cerr << "uncaught exception in the fast-wave framework: " << e.what() << '\n';
    return 1;
}
