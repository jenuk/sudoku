#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

#include "sudoku.h"
#include "generator.h"

// square root of length
// i.e. side length of single block
constexpr int N = 3;

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

    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "Generated " << n << "  sudokus in " << dt/1000 << " [ms]"
        << "i.e. " << (dt / n) / 1000 << " [ms/Sudoku]" << std::endl;


    Sudoku<N>::pprint = true;
    std::cout << "Last sudoku:\n" << minimal << "\n\nSolution:\n"
        << filled << std::endl;

    std::cout << "Is valid (puzzle): " << minimal.is_valid() << std::endl;
    std::cout << "Is unique and solveable (puzzle): " << minimal.is_unique() << std::endl;
    std::cout << "Is valid (solution): " << filled.is_valid() << std::endl;
    std::cout << "Is solved (solution): " << filled.is_solved() << std::endl;

    return 0;
}
