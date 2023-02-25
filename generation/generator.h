#ifndef SUDOKU_GENERATOR_H
#define SUDOKU_GENERATOR_H

#include <random>
#include <utility>

#include "sudoku.h"


template<int N>
Field<int, N> random_partial_field(std::mt19937& rng) {
    Field<int, N> field = fill_field<int, N>(0);

    std::array<int, N*N> row;
    std::iota(std::begin(row), std::end(row), 1);
    std::shuffle(
        std::begin(row),
        std::end(row),
        rng
    );
    for (int k=0; k < N*N; ++k) {
        field[k][0] = row[k];
    }
    // all further construction seems too slow to justify the further speed up

    return field;
}


template<int N>
Sudoku<N> generate_filled_sudoku(std::mt19937& rng) {
    Sudoku<N> sudoku(random_partial_field<N>(rng));
    std::vector<Sudoku<N>> solutions = sudoku.solve(random_first, rng);
    return solutions[0];
}


template<int N>
Sudoku<N> make_minimal(
        const Sudoku<N>& filled,
        std::mt19937& rng
        ) {
    Sudoku<N> minimal;
    std::array<int, N*N*N*N> clue_order;
    std::iota(std::begin(clue_order), std::end(clue_order), 0);
    std::shuffle(
        std::begin(clue_order),
        std::end(clue_order),
        rng
    );
    // Applying a symmetry to the clue order results in a sudoku
    // with partially that symmetry.

    // **Part 1**
    // remove as many clues as possible without checking every one
    // individually by using binary search
    // Number of solves: O(log(N))
    int lower = 0;
    int upper = clue_order.size();
    int middle = (lower + upper) / 2;
    for (int k=0; k<middle; ++k) {
        int x = clue_order[k]%(N*N);
        int y = clue_order[k]/(N*N);
        minimal.field[x][y] = filled.field[x][y];
    }

    while (upper - lower > 1) {
        if (minimal.is_unique()) {
            upper = middle;
            middle = (lower + upper)/2;
            for (int k=middle; k<upper; ++k) {
                minimal.field[clue_order[k]%(N*N)][clue_order[k]/(N*N)] = 0;
            }
        } else {
            lower = middle;
            middle = (lower + upper)/2;
            for (int k=lower; k<middle; ++k) {
                int x = clue_order[k]%(N*N);
                int y = clue_order[k]/(N*N);
                minimal.field[x][y] = filled.field[x][y];
            }
        }
    }

    for (int k=middle; k<=upper; ++k) {
        int x = clue_order[k]%(N*N);
        int y = clue_order[k]/(N*N);
        minimal.field[x][y] = filled.field[x][y];
    }

    // **Part 2**
    // check for every remaining clue if it is really necessary in context
    // int num_clues = 1;
    // Number of solves: O(upper), in general O(N^4)
    // (N^4 number of fields)
    for (int k=upper-1; k>0; --k) {
        // std::cout << k << std::endl;
        int i = clue_order[k]%(N*N);
        int j = clue_order[k]/(N*N);
        if (minimal.field[i][j] == 0) {
            continue;
        }
        minimal.field[i][j] = 0;
        if (not minimal.is_unique()) {
            minimal.field[i][j] = filled.field[i][j];
            // num_clues++;
        }
    }
    // std::cout << num_clues << std::endl;

    return minimal;
}


template<int N>
std::pair<Sudoku<N>, Sudoku<N>> generate_minimal_sudoku(
        std::mt19937& rng
        ) {
    Sudoku<N> filled = generate_filled_sudoku<N>(rng);
    Sudoku<N> minimal = make_minimal<N>(filled, rng);
    return std::make_pair(minimal, filled);
}


#endif
