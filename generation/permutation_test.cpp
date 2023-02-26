// generating a lot of permutations to see if the permutation gnerator is
// biased in a meaningful way, since permutating a long list can cause
// issues in some cases

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

// square root of length
// i.e. side length of single block
constexpr int N = 3;

int main(int argc, char** argv){
    std::random_device rd;
    std::mt19937 rng(rd());

    std::string filename = "permutations.txt";
    int n = 1000;
    // very basic argument parsing
    if (argc > 1) {
        n = std::stoi(argv[1]);
    }

    std::ofstream file;
    file.open(filename);

    std::array<int, N*N*N*N> order;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int k=0; k < n; ++k) {
        std::iota(std::begin(order), std::end(order), 0);
        std::shuffle(
            std::begin(order),
            std::end(order),
            rng
        );
        for (int i=0; i < order.size(); ++i) {
            file << order[i] << " ";
        }
        file << "\n";
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    file << std::flush;
    file.close();

    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "Generated " << n << "  permutations in " << dt/1000 << " [ms]"
        << "i.e. " << (dt / n) / 1000 << " [ms/Sudoku]" << std::endl;

    return 0;
}
