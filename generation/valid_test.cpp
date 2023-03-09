#include <iostream>
#include <tuple>
#include <vector>

#include "sudoku.h"
#include "generator.h"

// square root of length
// i.e. side length of single block
constexpr int N = 3;

int main(){
    Sudoku<N>::pprint = true;

    std::random_device rd;
    std::mt19937 rng(rd());

    Sudoku<N> sudoku(random_partial_field<N>(rng));
    std::cout << sudoku << std::endl;

    std::cout << "Is valid: " << sudoku.is_valid() << std::endl;
    std::cout << "Is solveable: " << sudoku.is_solveable() << std::endl;
    std::cout << "Is unique: " << sudoku.is_unique() << std::endl;
    std::vector<Sudoku<N>> solutions = sudoku.solve(all);
    std::cout << "Number of solutions: " << solutions.size() << std::endl;

    for (int k=0; k < 1000; ++k) {
        sudoku = Sudoku<N>(random_partial_field<N>(rng));
        bool valid = sudoku.is_valid();
        bool solveable = sudoku.is_solveable();
        if (valid and solveable) {
            std::cout << ".";
        } else {
            if (valid) {
                std::cout << "v";
            } else if (solveable) {
                std::cout << "s";
            } else {
                std::cout << "#";
            }
        }
    }
    std::cout << std::endl;

    std::cout << "Test minimal generation" << std::endl;
    Sudoku<N> minimal;
    Sudoku<N> filled;
    for (int k=0; k < 1000; ++k) {
        std::tie(minimal, filled) = generate_minimal_sudoku<N>(rng);
        if (minimal.is_minimal()) {
            std::cout << ".";
        } else {
            std::cout << "#";
        }
    }
    std::cout << std::endl;

    return 0;
}
