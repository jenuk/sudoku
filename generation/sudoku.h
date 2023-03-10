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
    any,
    two,
    random_first,
};

template <int N>
class Sudoku {
    public:
        Sudoku();
        Sudoku(Field<int, N>);

        Hints<N> gen_hints() const;
        std::vector<Sudoku> solve() const;
        std::vector<Sudoku> solve(SolveMode) const;
        std::vector<Sudoku> solve(SolveMode, std::mt19937&) const;

        bool is_unique() const;
        bool is_solveable() const;
        bool is_valid() const;
        bool is_solved() const;
        bool is_minimal() const;
        int num_clues() const;

        void flip();

        // partial ordering
        // is equal or less filled in version of other
        bool operator<=(const Sudoku<N>&) const;
        // is less filled in version of other
        bool operator<(const Sudoku<N>&) const;

        template <int M>
        friend Sudoku<M> make_minimal(
            const Sudoku<M>&,
            std::mt19937&
            );

        template <int M>
        friend std::ostream& operator<<(std::ostream&, const Sudoku<M>&);
        template <int M>
        friend std::istream& operator>>(std::istream&, Sudoku<M>&);

        static bool pprint; // pretty print vs comptact print

    private:
        Field<int, N> field;

        void inner_solve(
            Hints<N>&,
            std::vector<Sudoku<N>>&,
            SolveMode
            ) const;
        void random_solve(
            Hints<N>&,
            std::vector<Sudoku<N>>&,
            std::mt19937&
            ) const;
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
Hints<N> Sudoku<N>::gen_hints() const {
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
std::vector<Sudoku<N>> Sudoku<N>::solve() const {
    return this->solve(all);
}


template <int N>
std::vector<Sudoku<N>> Sudoku<N>::solve(SolveMode mode) const {
    std::vector<Sudoku> solutions;
    Hints<N> hints = this->gen_hints();

    if (mode == random_first) {
        // do not change this sudoku itself
        Sudoku<N> cp(*this);
        // this should be avoided if possible
        std::random_device rd;
        std::mt19937 rng(rd());
        cp.random_solve(hints, solutions, rng);
    } else {
        this->inner_solve(hints, solutions, mode);
    }
    return solutions;
}


template <int N>
std::vector<Sudoku<N>> Sudoku<N>::solve(
        SolveMode mode,
        std::mt19937& rng
        ) const {
    std::vector<Sudoku> solutions;
    Hints<N> hints = this->gen_hints();

    if (mode == random_first) {
        this->random_solve(hints, solutions, rng);
    } else {
        this->inner_solve(hints, solutions, mode);
    }
    return solutions;
}


template <int N>
bool Sudoku<N>::is_unique() const {
    return this->solve(two).size() == 1;
}


template <int N>
bool Sudoku<N>::is_solveable() const {
    return this->solve(any).size() > 0;
}


template <int N>
bool Sudoku<N>::is_valid() const {
    for (int i=0; i < N*N; ++i) {
        for (int j=0; j < N*N; ++j) {
            if (this->field[i][j] == 0) {
                continue;
            }

            int x = this->field[i][j];
            int bi = N*(i/N);
            int bj = N*(j/N);
            for (int k=0; k < N*N; ++k) {
                if (j!=k and this->field[i][k] == x) {
                    return false;
                } else if (i!=k and this->field[k][j] == x) {
                    return false;
                } else if (bi+k/N!=i and bj+k%N!=j
                        and this->field[bi+k/N][bj+k%N] == x) {
                    return false;
                }
            }
        }
    }

    return true;
}


template <int N>
bool Sudoku<N>::is_solved() const {
    for (int i=0; i < N*N; ++i) {
        for (int j=0; j < N*N; ++j) {
            if (this->field[i][j] == 0) {
                return false;
            }
        }
    }

    return true;
}


template <int N>
bool Sudoku<N>::is_minimal() const {
    Sudoku<N> cp(*this);
    for (int i=0; i < N*N; ++i) {
        for (int j=0; j < N*N; ++j) {
            if (this->field[i][j] == 0) {
                continue;
            }
            cp.field[i][j] = 0;
            if (cp.is_unique()) {
                return false;
            }
            cp.field[i][j] = this->field[i][j];
        }
    }

    return true;
}


template <int N>
int Sudoku<N>::num_clues() const {
    int num = 0;
    for (int i=0; i < N*N; ++i) {
        for (int j=0; j < N*N; ++j) {
            if (this->field[i][j] != 0) {
                ++num;
            }
        }
    }

    return num;
}


template <int N>
bool Sudoku<N>::operator<=(const Sudoku<N>& other) const {
    for (int i=0; i < N*N; ++i) {
        for (int j=0; j < N*N; ++j) {
            if (this->field[i][j] != 0
                    and this->field[i][j] != other.field[i][j]) {
                return false;
            }
        }
    }

    return true;
}

template <int N>
bool Sudoku<N>::operator<(const Sudoku<N>& other) const {
    bool is_less = false;
    for (int i=0; i < N*N; ++i) {
        for (int j=0; j < N*N; ++j) {
            if (this->field[i][j] == 0) {
                is_less = is_less or other.field[i][j] != 0;
            } else if (this->field[i][j] != other.field[i][j]) {
                return false;
            }
        }
    }

    return true;
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
void Sudoku<N>::inner_solve(
        Hints<N>& hints,
        std::vector<Sudoku<N>>& solutions,
        SolveMode mode
        ) const {
    // Strategy:
    // -- Fill all cells with a single hint in them
    // -- Find last cell with fewest hints available
    // -- Try all possible combinations for that cell

    std::vector<std::pair<Sudoku<N>, Hints<N>>> stack = {std::make_pair(*this, hints)};
    // max length at most O(N^6), maybe a lower bound exists

    while (stack.size() > 0) {
        Sudoku<N> current;
        std::tie(current, hints) = stack.back();
        stack.pop_back();

        // find best starting cell
        int best_i = 0;
        int best_j = 0;
        int clues = N*N+1; // max number of clues == N*N

        for (int i=0; i < N*N; ++i) {
            for (int j=0; j < N*N; ++j) {
                if (current.field[i][j] != 0) {
                    continue;
                }
                int last_k = 0; // for trivial fill only
                int current_clues = 0;
                for (int k=1; k <= N*N; ++k) {
                    if (hints[i][j][k]) {
                        ++current_clues;
                        last_k = k;
                    }
                }

                if (current_clues == 1) {
                    // trivial fill
                    current.field[i][j] = last_k;
                    remove_hint<N>(hints, i, j, last_k);
                    // revisiting previous cells doesn't seem to speed it up
                } else if (current_clues == 0) {
                    // no candidate solution left for this cell
                    // there is no way to solve this sudoku with the current
                    // hints.
                    // goto goes to next element in stack, i.e. a continue for
                    // the outer while loop, sorry
                    goto next;
                } else if (current_clues <= clues) {
                    best_i = i;
                    best_j = j;
                    clues = current_clues;
                }
            }
        }
        // no location found to fill in
        if (clues == N*N+1) {
            // all cells are filled
            solutions.push_back(current);
            if ((mode == any) or (mode == two and solutions.size() > 1)) {
                // suffieciently many solutions found to stop now
                return;
            } else {
                continue;
            }
        }

        // try all posibilities
        for (int k=1; k <= N*N; ++k) {
            if (not hints[best_i][best_j][k]) {
                continue;
            }
            // stack creates a copy
            stack.push_back(std::make_pair(current, hints));
            // insert k into copies
            remove_hint<N>(stack.back().second, best_i, best_j, k);
            stack.back().first.field[best_i][best_j] = k;
        }
        next:;
    }
}


template <int N>
void Sudoku<N>::random_solve(
        Hints<N>& hints,
        std::vector<Sudoku<N>>& solutions,
        std::mt19937& rng
        ) const {
    // NOTE: this is the same as inner_solve with the difference that a random
    // place is chosen as branching point instead of the first to decrease
    // spatial correlations
    // probably too slow for N > 3

    std::vector<std::pair<Sudoku<N>, Hints<N>>> stack = {std::make_pair(*this, hints)};
    std::array<int, N*N> posibilities;
    std::iota(std::begin(posibilities), std::end(posibilities), 1);
    // max length at most O(N^6), maybe a lower bound exists

    while (stack.size() > 0) {
        Sudoku<N> current;
        std::tie(current, hints) = stack.back();
        stack.pop_back();

        // find best starting cell
        std::vector<std::pair<int, int>> positions;
        int p_i, p_j; // final positions, will be bound later
        int clues = N*N+1; // max number of clues == N*N

        for (int i=0; i < N*N; ++i) {
            for (int j=0; j < N*N; ++j) {
                if (current.field[i][j] != 0) {
                    continue;
                }
                int last_k = 0; // for trivial fill only
                int current_clues = 0;
                for (int k=1; k <= N*N; ++k) {
                    if (hints[i][j][k]) {
                        ++current_clues;
                        last_k = k;
                    }
                }

                if (current_clues == 1) {
                    // trivial fill
                    current.field[i][j] = last_k;
                    remove_hint<N>(hints, i, j, last_k);
                    // revisiting previous cells doesn't seem to speed it up
                } else if (current_clues == 0) {
                    // no candidate solution left for this cell
                    // there is no way to solve this sudoku with the current
                    // hints.
                    // goto goes to next element in stack, i.e. a continue for
                    // the outer while loop, sorry
                    goto next;
                } else if (current_clues <= clues) {
                    positions.push_back(std::make_pair(i, j));
                    clues = current_clues;
                }
            }
        }
        // no location found to fill in
        if (clues == N*N+1) {
            // all cells are filled
            solutions.push_back(current);
            return;
        }

        std::tie(p_i, p_j) = positions[std::uniform_int_distribution<>(0, positions.size()-1)(rng)];
        std::shuffle(std::begin(posibilities), std::end(posibilities), rng);
        // try all posibilities
        for (int kk=0; kk < N*N; ++kk) {
            int k = posibilities[kk];
            if (not hints[p_i][p_j][k]) {
                continue;
            }
            // stack creates a copy
            stack.push_back(std::make_pair(current, hints));
            // insert k into copies
            remove_hint<N>(stack.back().second, p_i, p_j, k);
            stack.back().first.field[p_i][p_j] = k;
        }
        next:;
    }
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
