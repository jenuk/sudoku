#ifndef SUDOKU_H
#define SUDOKU_H

#include <array>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// N is square root of field length, e.g. 3 for a standard 9x9 sudoku
// -> this way the boxes inside are well defined.
template <class T, int N>
using Field = std::array<std::array<T, N*N>, N*N>;
template <int N>
using Hints = Field<std::array<bool, N*N+1>, N>;

template <class T, int N>
Field<T, N> fill_field(const T& fill_val) {
    std::array<T, N*N> inner;
    inner.fill(fill_val);
    Field<T, N> field;
    field.fill(inner);
    return field;
}

enum SolveMode {
    all,
    two,
    random_first,
};

template <int N>
class Sudoku {
    public:
        Sudoku();
        Sudoku(Field<int, N>);

        Hints<N> gen_hints();
        std::vector<Sudoku> solve();
        std::vector<Sudoku> solve(SolveMode);
        std::vector<Sudoku> solve(SolveMode, std::mt19937&);
        bool is_unique();

        template <int M>
        friend std::ostream& operator<<(std::ostream&, const Sudoku<M>&);
        template <int M>
        friend std::istream& operator>>(std::istream&, Sudoku<M>&);

        template <int M>
        friend Sudoku<M> make_minimal(
            const Sudoku<M>&,
            std::mt19937&
            );

        void flip();

        static bool pprint; // pretty print vs comptact print

    private:
        Field<int, N> field;
        std::array<int, N*N> posibilities;

        bool inner_solve(
            Hints<N>&,
            std::vector<Sudoku<N>>&,
            SolveMode,
            std::mt19937&
            );
        bool random_solve(
            Hints<N>&,
            std::vector<Sudoku<N>>&,
            std::mt19937&
            );
};


template <int N>
bool Sudoku<N>::pprint = true;

template <int N>
void Sudoku<N>::flip() {
    std::reverse(std::begin(this->field), std::end(this->field));
}


template <int N>
Sudoku<N>::Sudoku() {
    this->field = fill_field<int, N>(0);
}

template <int N>
Sudoku<N>::Sudoku(Field<int, N> field) {
    this->field = field;
}

template <int N>
Hints<N> Sudoku<N>::gen_hints() {
    std::array<bool, N*N+1> full_hint;
    full_hint.fill(true);
    full_hint[0] = false;
    Hints<N> unseen = fill_field<std::array<bool, N*N+1>, N>(full_hint);
    
    for (int i=0; i<N*N; ++i) {
        for (int j=0; j<N*N; ++j) {
            if (this->field[i][j] == 0) {
                continue;
            }
            // upper left point of current block
            int bi = N*(i/N);
            int bj = N*(j/N);
            for (int x=0; x<N*N; ++x) {
                unseen[i][x][this->field[i][j]] = false;
                unseen[x][j][this->field[i][j]] = false;
                unseen[bi + x/N][bj + x%N][this->field[i][j]] = false;
            }
        }
    }
    return unseen;
}


template <int N>
std::vector<Sudoku<N>> Sudoku<N>::solve() {
    return this->solve(all);
}


template <int N>
std::vector<Sudoku<N>> Sudoku<N>::solve(SolveMode mode) {
    std::vector<Sudoku> solutions;
    // reset in case of previous random search
    std::random_device rd;
    std::mt19937 rng(rd());
    return this->solve(mode, rng);
}


template <int N>
std::vector<Sudoku<N>> Sudoku<N>::solve(
        SolveMode mode,
        std::mt19937& rng
        ) {
    std::vector<Sudoku> solutions;
    // reset in case of previous random search
    std::iota(std::begin(posibilities), std::end(posibilities), 1);
    Hints<N> hints = this->gen_hints();
    // do not change this sudoku itself:
    Sudoku<N> cp(*this);

    if (mode == random_first) {
        cp.random_solve(hints, solutions, rng);
    } else {
        cp.inner_solve(hints, solutions, mode, rng);
    }
    return solutions;
}


template <int N>
bool Sudoku<N>::is_unique() {
    return this->solve(two).size() == 1;
}


template <int N>
void remove_hint(Hints<N>& hints, int i, int j, int k) {
    for (int x=0; x < N*N; ++x) {
        hints[i][x][k] = false;
        hints[x][j][k] = false;
        hints[N*(i/N) + x/N][N*(j/N) + x%N][k] = false;
    }
}

template <int N>
bool Sudoku<N>::inner_solve(
        Hints<N>& hints,
        std::vector<Sudoku<N>>& solutions,
        SolveMode mode,
        std::mt19937& rng
        ) {
    // Strategy:
    // -- Fill all cells with a single hint in them
    // -- Find cell with fewest hints available
    // -- Try all possible combinations for that cell and call inner_solve

    // TODO: convert recursion to stack and see if that speeds it up

    // find best starting cell
    int best_i = 0;
    int best_j = 0;
    bool updated = false;

    do {
        // this do-while loop doesn't seem to influence speed in praxis
        updated = false;
        int clues = N*N+1;
        bool solved = true; // hope for the best

        for (int i=0; i < N*N; ++i) {
            for (int j=0; j < N*N; ++j) {
                if (this->field[i][j] != 0) {
                    continue;
                }
                int last_k = 0; // for trivial fill only
                int current_clues = 0;
                for (int k=1; k <= N*N; ++k) {
                    if (hints[i][j][k]) {
                        current_clues++;
                        last_k = k;
                    }
                }

                if (current_clues == 1) {
                    // trivial fill
                    this->field[i][j] = last_k;
                    remove_hint<N>(hints, i, j, last_k);
                    updated = true;
                    // -> new possibilities might be earlier in the
                    // grid, so look for trivial fills a second time
                } else if (current_clues == 0) {
                    // no candidate solution left for this cell
                    return false;
                } else if (current_clues < clues) {
                    // non-trivial cell
                    solved = false;
                    best_i = i;
                    best_j = j;
                    clues = current_clues;
                }
            }
        }

        if (solved) {
            // all cells are filled
            solutions.push_back(*this);
            return ((mode == random_first)
                    or (mode == two and solutions.size() > 1));
        }
    } while (updated);

    if (mode == random_first) {
        std::shuffle(
            std::begin(this->posibilities),
            std::end(this->posibilities),
            rng
        );
    }
    // try all posibilities
    // std::vector<Sudoku<N>> solutions;
    for (int kk=0; kk < N*N; ++kk) {
        int k = this->posibilities[kk];
        if (not hints[best_i][best_j][k]) {
            continue;
        }
        // copy
        Sudoku<N> current(*this);
        Hints<N> hints_cp(hints);

        // insert k into copies
        current.field[best_i][best_j] = k;
        remove_hint<N>(hints_cp, best_i, best_j, k);

        if (current.inner_solve(hints_cp, solutions, mode, rng)) {
            return true;
        }
    }

    return false;
}


template <int N>
bool Sudoku<N>::random_solve(
        Hints<N>& hints,
        std::vector<Sudoku<N>>& solutions,
        std::mt19937& rng
        ) {
    // NOTE: this is the same as inner_solve with the difference that a random
    // place is chosen as branching point instead of the first to decrease
    // spatial correlations

    // find best starting cell
    bool updated = false;
    std::vector<std::pair<int, int>> bests;

    do {
        // this do-while loop doesn't seem to influence speed in praxis
        updated = false;
        bests = {};
        int clues = N*N+1;
        bool solved = true; // hope for the best

        for (int i=0; i < N*N; ++i) {
            for (int j=0; j < N*N; ++j) {
                if (this->field[i][j] != 0) {
                    continue;
                }
                int last_k = 0; // for trivial fill only
                int current_clues = 0;
                for (int k=1; k <= N*N; ++k) {
                    if (hints[i][j][k]) {
                        current_clues++;
                        last_k = k;
                    }
                }

                if (current_clues == 1) {
                    // trivial fill
                    this->field[i][j] = last_k;
                    remove_hint<N>(hints, i, j, last_k);
                    updated = true;
                    // -> new possibilities might be earlier in the
                    // grid, so look for trivial fills a second time
                } else if (current_clues == 0) {
                    // no candidate solution left for this cell
                    return false;
                } else if (current_clues < clues) {
                    // non-trivial cell
                    solved = false;
                    clues = current_clues;
                    bests = {std::make_pair(i, j)};
                } else if (current_clues == clues) {
                    bests.push_back(std::make_pair(i, j));
                }
            }
        }
        if (solved) {
            // all cells are filled
            solutions.push_back(*this);
            return true;
        }
    } while (updated);

    int idx = std::uniform_int_distribution<>(0, bests.size()-1)(rng);
    int best_i, best_j;
    std::tie(best_i, best_j) = bests[idx];
    std::shuffle(
        std::begin(this->posibilities),
        std::end(this->posibilities),
        rng
    );
    // try all posibilities
    // std::vector<Sudoku<N>> solutions;
    for (int kk=0; kk < N*N; ++kk) {
        int k = this->posibilities[kk];
        if (not hints[best_i][best_j][k]) {
            continue;
        }
        // copy
        Sudoku<N> current(*this);
        Hints<N> hints_cp(hints);

        // insert k into copies
        current.field[best_i][best_j] = k;
        remove_hint<N>(hints_cp, best_i, best_j, k);

        if (current.random_solve(hints_cp, solutions, rng)) {
            return true;
        }
    }

    return false;
}


template <int N>
std::ostream& operator<<(std::ostream& os, const Sudoku<N>& sudoku) {
    int length = std::to_string(N*N).size();
    if (sudoku.pprint) {
        std::string horizontal_block((length+1)*N, '-');
        std::string row = horizontal_block;
        for (int k=0; k < N-1; ++k) {
            row += "+-" + horizontal_block;
        }
        for (int i=0; i < N*N; ++i) {
            for (int j=0; j < N*N; ++j) {
                std::string next;
                if (sudoku.field[i][j] == 0) {
                    next = ".";
                } else {
                    next = std::to_string(sudoku.field[i][j]);
                }
                os << std::string(length - next.size(), ' ') << next << " ";
                if ((j+1) % N == 0 and j+1 != N*N) {
                    os << "| ";
                }
            }
            os << "\n";
            if ((i+1) % N == 0 and i+1 != N*N) {
                os << row << "\n";
            }
        }
    } else {
        for (int i=0; i < N*N; ++i) {
            for (int j=0; j < N*N; ++j) {
                std::string next = std::to_string(sudoku.field[i][j]);
                os << std::string(length - next.size(), '0') << next;
            }
        }
    }
    return os;
}

template <int N>
std::istream& operator>>(std::istream& is, Sudoku<N>& sudoku) {
    if (sudoku.pprint) {
        char dummy;
        std::string line;
        for (int i=0; i < N*N; ++i) {
            for (int j=0; j < N*N; ++j) {
                is >> sudoku.field[i][j];
                if ((j+1) % N == 0 and j+1 != N*N) {
                    is >> dummy;
                }
            }
            if ((i+1) % N == 0 and i+1 != N*N) {
                is >> line;
            }
        }
    } else {
        int length = std::to_string(N*N).size();
        for (int i=0; i < N*N; ++i) {
            for (int j=0; j < N*N; ++j) {
                char ch;
                int current = 0;
                for (int k=0; k < length; ++k) {
                    is >> ch;
                    // parse char to int myself
                    current = 10*current + (ch - '0');
                }
                sudoku.field[i][j] = current;
            }
        }
    }
    return is;
}

#endif
