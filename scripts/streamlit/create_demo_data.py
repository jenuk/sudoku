"""Script to create some data subsets with specific biases"""

N = 100_000

def save(data: list[str], fn: str):
    if len(data) < N:
        print(f"Warning: Less than {N:,} lines for {fn}")
    with open(f"demo/{fn}.txt", "w") as file:
        file.write("\n".join(data[:N]))
        file.write("\n") # trailing line break


def duplicate_puzzle(data):
    res = []
    for line in data:
        res.append(line)
        res.append(line)
        if len(res) >= N:
            break
    return res


def duplicate_solutions(data):
    res = []
    for line in data:
        res.append(line)

        line2 = list(line)
        idx = line2.index("0")
        line2[idx] = line2[82 + idx]
        res.append("".join(line2))
        if len(res) >= N:
            break
    return res


def few_hints(data):
    res = []
    for line in data:
        if line.count("0") > 59:
            res.append(line)
        if len(res) >= N:
            break
    for line in data:
        if line.count("0") == 59:
            res.append(line)
        if len(res) >= N:
            break
    return res


def fix_point(data, idx, val):
    res = []
    for line in data:
        if line[idx] == str(val):
            res.append(line)
        if len(res) >= N:
            break
    return res


def linked(data):
    res = []
    for line in data:
        if line[0] == line[80] and line[82] == line[-1]:
        # if line[82] == line[-1]:
            res.append(line)
        if len(res) >= N:
            break
    return res

if __name__ == "__main__":
    with open("data/v09.txt") as file:
        source = [line.strip() for line in file]

    save(duplicate_puzzle(source), "duplicate_puzzle")
    save(duplicate_solutions(source), "duplicate_solutions")
    save(few_hints(source), "few_hints")
    save(fix_point(source, 41, 0), "single_bias_0")
    save(fix_point(source, 41, 3), "single_bias_3")
    save(linked(source), "linked")
