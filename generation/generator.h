#ifndef SUDOKU_GENERATOR_H
#define SUDOKU_GENERATOR_H

#include <iostream>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sudoku.h"


template<int N>
Field<int, N> random_partial_field(std::mt19937& rng) {
    Field<int, N> field = fill_field<int, N>(0);

    std::array<int, N*N> row;
    std::iota(std::begin(row), std::end(row), 0);
    std::shuffle(std::begin(row), std::end(row), rng);

    std::array<int, N*N> col;
    std::iota(std::begin(col), std::end(col), 0);
    std::shuffle(std::begin(col), std::end(col), rng);

    for (int k=0; k < N*N; ++k) {
        field[row[k]][col[k]] = k+1;
    }

    return field;
}


template<>
Field<int, 3> random_partial_field<3>(std::mt19937& rng) {
    // specific optimization for 9x9 sudokus
    // each cell has at most 6 possible entries
    // small branching factor allows for (hopefully) unbiased generation
    Field<int, 3> field = fill_field<int, 3>(0);

    std::array<int, 9> nums;
    std::iota(std::begin(nums), std::end(nums), 1);
    std::shuffle(nums.begin(), nums.end(), rng);
    // fill first nums randomly
    for (int k=0; k < 9; ++k) {
        field[k][0] = nums[k];
    }
    // fill first box randomly with remaining numbers
    std::shuffle(nums.begin()+3, nums.end(), rng);
    for (int k=3; k < 9; ++k) {
        field[k%3][k/3] = nums[k];
    }

    // fill second row of second and third box in a valid way
    std::vector<int> bucket1;
    std::vector<int> bucket2;

    std::shuffle(nums.begin(), nums.begin()+3, rng);
    std::unordered_set<int> set1 = {field[3][0], field[4][0], field[5][0]};
    for (int k=0; k < 3; ++k) {
        if (set1.find(field[k][2]) != set1.end()) {
            bucket2.push_back(field[k][2]);
            bucket1.push_back(nums[k]);
        } else {
            bucket1.push_back(field[k][2]);
            bucket2.push_back(nums[k]);
        }
    }
    std::shuffle(bucket1.begin(), bucket1.end(), rng);
    std::shuffle(bucket2.begin(), bucket2.end(), rng);
    for (int k=0; k < 3; ++k) {
        field[k+3][1] = bucket1[k];
        field[k+6][1] = bucket2[k];
    }

    // finish third row
    std::unordered_set<int> set2 = {field[0][2], field[1][2], field[2][2]};
    set1.insert(field[3][1]);
    set1.insert(field[4][1]);
    set1.insert(field[5][1]);
    std::shuffle(nums.begin(), nums.end(), rng);
    bucket1 = {};
    bucket2 = {};
    for (int i=0; i < 9; ++i) {
        int k = nums[i];
        if (set2.find(k) != set2.end()) {
            // already in this nums
        } else if (set1.find(k) != set1.end()) {
            // number is in box 1
            bucket2.push_back(k);
        } else {
            bucket1.push_back(k);
        }
    }
    for (int k=0; k < 3; ++k) {
        field[k+3][2] = bucket1[k];
        field[k+6][2] = bucket2[k];
    }

    // repeat for first three columns
    nums = {
        field[0][0], field[0][1], field[0][2],
        field[1][0], field[1][1], field[1][2],
        field[2][0], field[2][1], field[2][2]
    };
    std::shuffle(nums.begin()+3, nums.end(), rng);
    for (int k=3; k < 9; ++k) {
        field[0][k] = nums[k];
    }

    // fill second column of second and third box in a valid way
    bucket1 = {};
    bucket2 = {};

    std::shuffle(nums.begin(), nums.begin()+3, rng);
    set1 = {field[0][3], field[0][4], field[0][5]};
    for (int k=0; k < 3; ++k) {
        if (set1.find(field[2][k]) != set1.end()) {
            bucket2.push_back(field[2][k]);
            bucket1.push_back(nums[k]);
        } else {
            bucket1.push_back(field[2][k]);
            bucket2.push_back(nums[k]);
        }
    }
    std::shuffle(bucket1.begin(), bucket1.end(), rng);
    std::shuffle(bucket2.begin(), bucket2.end(), rng);
    for (int k=0; k < 3; ++k) {
        field[1][k+3] = bucket1[k];
        field[1][k+6] = bucket2[k];
    }

    // finish third col
    set2 = {field[2][0], field[2][1], field[2][2]};
    set1.insert(field[1][3]);
    set1.insert(field[1][4]);
    set1.insert(field[1][5]);
    std::shuffle(nums.begin(), nums.end(), rng);
    bucket1 = {};
    bucket2 = {};
    for (int i=0; i < 9; ++i) {
        int k = nums[i];
        if (set2.find(k) != set2.end()) {
            // already in this nums
        } else if (set1.find(k) != set1.end()) {
            // number is in box 1
            bucket2.push_back(k);
        } else {
            bucket1.push_back(k);
        }
    }
    for (int k=0; k < 3; ++k) {
        field[2][k+3] = bucket1[k];
        field[2][k+6] = bucket2[k];
    }

    return field;
}


template<int N>
Sudoku<N> generate_filled_sudoku(std::mt19937& rng) {
    Sudoku<N> sudoku(random_partial_field<N>(rng));
    // std::vector<Sudoku<N>> solutions = sudoku.solve(random_first, rng);
    std::vector<Sudoku<N>> solutions = sudoku.solve(all);
    if (solutions.size() == 0) {
        // This shouldn't happen if the solver works
        std::cout << "Failed initial generation for:" << std::endl;
        bool status = Sudoku<N>::pprint;
        Sudoku<N>::pprint = true;
        std::cout << sudoku << std::endl;
        Sudoku<N>::pprint = status;
        return generate_filled_sudoku<N>(rng);
    } else {
        // return solutions[0];
        int idx = std::uniform_int_distribution<>(0, solutions.size()-1)(rng);
        return solutions[idx];
    }
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
    // Applying a symmetry to the clue order (probably, not tested) results in
    // a sudoku with partially that symmetry.

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
    // Number of solves: O(upper), in general O(N^4)
    // (N^4 number of cells in field)
    for (int k=upper-1; k>0; --k) {
        int i = clue_order[k]%(N*N);
        int j = clue_order[k]/(N*N);
        if (minimal.field[i][j] == 0) {
            continue;
        }
        minimal.field[i][j] = 0;
        if (not minimal.is_unique()) {
            minimal.field[i][j] = filled.field[i][j];
        }
    }

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
