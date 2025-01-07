#include <boost/ut.hpp>

#include <config_parsing.hpp>
#include <fft.hpp>
#include <kernels.hpp>
#include <wrappers.hpp>

#include <source_location>

template <typename S, typename F> void run_gpu_test(std::vector<Complex<S>> &data, F f) {
    std::size_t nr_bytes = sizeof(Complex<S>) * data.size();

    DevComplex<S> *d_data;
    check_cuda_result("malloc", cudaMalloc((void **)&d_data, nr_bytes));

    check_cuda_result("memcpy h2d",
                      cudaMemcpy(d_data, data.data(), nr_bytes, cudaMemcpyHostToDevice));

    f(d_data);

    check_cuda_result("memcpy d2h",
                      cudaMemcpy(data.data(), d_data, nr_bytes, cudaMemcpyDeviceToHost));

    check_cuda_result("sync", cudaDeviceSynchronize());
}

using source_location = boost::ext::ut::v1_1_8::reflection::source_location;

void expect_close(const double &a, const double &b,
                  const source_location &sl = source_location::current()) {
    using namespace boost::ut;
    expect(std::abs(a - b) < 1e-7, sl);
}

void expect_close(const Complex<double> &a, const Complex<double> &b,
                  const source_location &sl = source_location::current()) {
    using namespace boost::ut;
    expect_close(a.real(), b.real(), sl);
    expect_close(a.imag(), b.imag(), sl);
}

void expect_close(const std::vector<double> &a, const std::vector<double> &b,
                  const source_location &sl = source_location::current()) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        expect_close(a[i], b[i], sl);
    }
}

void expect_close(const std::vector<Complex<double>> &a, const std::vector<Complex<double>> &b,
                  const source_location &sl = source_location::current()) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        expect_close(a[i], b[i], sl);
    }
}

int main() {
    using namespace boost::ut;

    "fftfreq"_test = [] {
        constexpr int N = 10;
        std::vector<double> arr(N);
        for (int i = 0; i < N; ++i) {
            arr[i] = fftfreq<double>(i, N, 1.0f);
        }
        const std::vector<double> expected{0.0f,  0.1f,  0.2f,  0.3f,  0.4f,
                                           -0.5f, -0.4f, -0.3f, -0.2f, -0.1f};

        expect_close(arr, expected);
    };

    "grating"_test = [] {
        using C = Complex<double>;
        std::vector<C> data(16, C{1., 0.});

        run_gpu_test(data, [](DevComplex<double> *d_data) {
            const SimParams params{16, 1.0, 1.77, 0., 0., 0., 0.};
            apply_grating_factors_kernel<double>
                <<<1, 32>>>(d_data, params, DevComplex<double>{2., 0.}, DevComplex<double>{0., 0.},
                            6.0, 0.5, 0.0);
        });

        std::vector<C> expected = {C{0., 0.}, C{2., 0.}, C{2., 0.}, C{2., 0.}, C{0., 0.}, C{0., 0.},
                                   C{0., 0.}, C{2., 0.}, C{2., 0.}, C{2., 0.}, C{0., 0.}, C{0., 0.},
                                   C{0., 0.}, C{2., 0.}, C{2., 0.}, C{2., 0.}};

        expect(data == expected);
    };

    "fft"_test = [] {
        // Require the normalization factor to be 1 in the forward direction
        // and 1/n in the inverse direction

        const std::size_t N = 16;
        std::vector<Complex<double>> data(N, Complex<double>{1., 0.});
        const std::vector<Complex<double>> orig = data;

        const FFT<double> fft(N);

        run_gpu_test(data, [&](DevComplex<double> *d_data) { fft.forward(d_data, d_data); });
        check_cuda_result("forward", cudaPeekAtLastError());

        expect(data[0] == Complex<double>{16., 0.});
        for (std::size_t i = 1; i < N; ++i) {
            expect(data[i] == Complex<double>{0., 0.});
        }

        run_gpu_test(data, [&](DevComplex<double> *d_data) { fft.inverse(d_data, d_data); });
        expect_close(data, orig);
    };

    "fft2"_test = [] {
        // compare to np.fft.fft

        const std::size_t N = 8;

        std::vector<Complex<double>> data{{0., 0.}, {0., 0.}, {1., 0.}, {2., 1.},
                                          {3., 2.}, {2., 0.}, {1., 0.}, {0., 0.}};

        std::vector<Complex<double>> expected{
            {9., 3.}, {-5.12132034, -2.70710678}, {0., 2.}, {0.53553391, -1.29289322},
            {1., 1.}, {-0.87867966, -1.29289322}, {2., 2.}, {-6.53553391, -2.70710678}};
        const FFT<double> fft(N);

        run_gpu_test(data, [&](DevComplex<double> *d_data) { fft.forward(d_data, d_data); });

        expect_close(data, expected);
    };

    "config_parsing_grating"_test = [] {
        // We don't care about rounding errors so we use simple values here.
        // Don't use this as an example of a sensible grating.

        const std::string grating_yaml = "type: grating\n"
                                         "pitch: 0.125\n"
                                         "dc: [0.5, 0.75]\n"
                                         "z_start: 0.25\n"
                                         "thickness: 1.25\n"
                                         "nr_steps: 1\n"
                                         "x_positions: [0.0]\n"
                                         "substrate_thickness: 0.0625\n"
                                         "mat_a: [\"Si\", 2.0]\n"
                                         "mat_b: [\"Au\", 3.0]\n"
                                         "mat_substrate: [\"Si\", 4.0]\n";

        const Material si2{"Si", 2.0};
        const Material au3{"Au", 3.0};
        const Material si4{"Si", 4.0};

        const DeltabetaTable db_table{{si2, Complex<double>{11., 0.}},
                                      {au3, Complex<double>{12., 0.}},
                                      {si4, Complex<double>{13., 0.}}};

        const auto grating_node = YAML::Load(grating_yaml);
        const auto grating = parse_grating(grating_node, db_table);

        expect(grating.pitch == 0.125);
        expect(grating.dc[0] == 0.5);
        expect(grating.dc[1] == 0.75);
        expect(grating.z_start == 0.25);
        expect(grating.thickness == 1.25);
        expect(grating.nr_steps == 1);
        expect(grating.x_positions.size() == 1);
        expect(grating.x_positions[0] == 0.0);
        expect(grating.substrate_thickness == 0.0625);
        expect(grating.deltabeta_a == Complex{11., 0.});
        expect(grating.deltabeta_b == Complex{12., 0.});
        expect(grating.deltabeta_substrate == Complex{13., 0.});

        const auto optical_element_ptr =
            parse_optical_element(grating_node, db_table, fs::path("/asdf-config-dir"));
        const auto grating_ptr = static_cast<Grating *>(optical_element_ptr.get());
        expect(grating_ptr->pitch == 0.125);
    };

    "config_parsing_dbtable"_test = [] {
        const YAML::Node db_table_node =
            YAML::Load("- [[\"Si\", 2.34], [2.2888e-7, 2.5847e-10]]\n"
                       "- [[\"Au\", 19.32], [1.4981e-6, 3.6174e-8]]\n");

        const DeltabetaTable db_table = parse_deltabeta_table(db_table_node);

        expect(db_table.size() == 2);
        expect(db_table[0].first.name == "Si");
        expect(db_table[1].first.name == "Au");

        expect_close(db_table[0].first.density, 2.34);
        expect_close(db_table[1].first.density, 19.32);

        expect_close(db_table[0].second, Complex<double>{2.2888e-7, 2.5847e-10});
        expect_close(db_table[1].second, Complex<double>{1.4981e-6, 3.6174e-8});
    };

    "config_parsing_simparams"_test = [] {
        const YAML::Node sim_params_node = YAML::Load("sim_params:\n"
                                                      "    N: 8192\n"
                                                      "    dx: 8e-10\n"
                                                      "    z_detector: 1.77\n"
                                                      "    detector_size: 5e-4\n"
                                                      "    detector_pixel_size_x: 1e-5\n"
                                                      "    detector_pixel_size_y: 1.0\n"
                                                      "    chunk_size: 10469376\n")["sim_params"];

        const SimParams sim_params = parse_sim_params(sim_params_node, 3.0);
        expect(sim_params.N == 8192);
        expect_close(sim_params.dx, 8e-10);
        expect_close(sim_params.z_detector, 1.77);
        expect_close(sim_params.detector_size, 5e-4);
        expect_close(sim_params.detector_pixel_size_x, 1e-5);
        expect_close(sim_params.detector_pixel_size_y, 1.0);
    };
}
