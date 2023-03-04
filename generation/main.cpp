#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "sudoku.h"
#include "generator.h"

// square root of length
// i.e. side length of single block
constexpr int N = 3;


std::string format_ns(int64_t dt) {
    // no overflow till ~580 years
    std::vector<int64_t> lenghts = {1, 1000, 1000, 1000, 60, 60, 24};
    std::vector<std::string> units = {
        "ns", "μs", "ms", "s", "minutes", "hours", "days"
    };
    int idx = 0;
    int64_t prev = 0;
    while (dt > lenghts[idx+1]) {
        ++idx;
        prev = dt%lenghts[idx];
        dt = dt/lenghts[idx];
    }
    std::string out = std::to_string(dt) + units[idx];
    if (0 < prev and dt < 100) {
        out += " " + std::to_string(prev) + units[idx-1];
    }
    return out;
}


int main(int argc, char** argv){
    std::random_device rd;
    std::mt19937 rng(rd());

    std::string filename = "generated.txt";
    int n = 1000;
    // very basic argument parsing
    if (argc > 1) {
        filename = argv[1];
        if (filename == "-h" or filename == "--help") {
            std::cout << "./generate [filename [number of sudokus]]" << std::endl;
            return 0;
        }
    }
    if (argc > 2) {
        n = std::stoi(argv[2]);
    }
    std::cout << "Generating " << n << " sudokus and saving in "
        << filename << std::endl;

    std::ofstream file;
    file.open(filename, std::ofstream::app);

    Sudoku<N> minimal;
    Sudoku<N> filled;
    Sudoku<N>::pprint = false;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int k=0; k < n; ++k) {
        std::tie(minimal, filled) = generate_minimal_sudoku<N>(rng);
        file << minimal << " " << filled << "\n";
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    file << std::flush;
    file.close();

    auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    std::cout << "Generated " << n << "  sudokus in " << format_ns(dt)
        << " i.e. " << format_ns(dt / n) << " for a Sudoku\n" << std::endl;



    Sudoku<N>::pprint = true;
    std::cout << "Last sudoku:\n" << minimal << "\n\nSolution:\n"
        << filled << std::endl;

    std::cout << "Is valid (puzzle): " << minimal.is_valid() << std::endl;
    std::cout << "Is unique and solveable (puzzle): " << minimal.is_unique() << std::endl;
    std::cout << "Is valid (solution): " << filled.is_valid() << std::endl;
    std::cout << "Is solved (solution): " << filled.is_solved() << std::endl;

    return 0;
}
