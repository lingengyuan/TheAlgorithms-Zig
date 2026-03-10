# Special Topics / 专题算法

- Source of truth: the detailed catalog sections from the pre-split root README.
- 数据来源：拆分前根 README 的详细目录条目。

### Geodesy (2)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Haversine Distance | [`geodesy/haversine_distance.zig`](geodesy/haversine_distance.zig) | O(1) |
| Lambert's Ellipsoidal Distance | [`geodesy/lamberts_ellipsoidal_distance.zig`](geodesy/lamberts_ellipsoidal_distance.zig) | O(1) |

### 测地学 (2)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| Haversine 距离 | [`geodesy/haversine_distance.zig`](geodesy/haversine_distance.zig) | O(1) |
| Lambert 椭球距离 | [`geodesy/lamberts_ellipsoidal_distance.zig`](geodesy/lamberts_ellipsoidal_distance.zig) | O(1) |

### Geometry (1)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Geometry Primitives and Measurements | [`geometry/geometry.zig`](geometry/geometry.zig) | O(1) to O(n) by operation |

### 几何 (1)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 几何基本对象与测量 | [`geometry/geometry.zig`](geometry/geometry.zig) | 视操作而定，O(1) 到 O(n) |

### Cellular Automata (6)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Conway's Game of Life (Next Generation) | [`cellular_automata/conways_game_of_life.zig`](cellular_automata/conways_game_of_life.zig) | O(h · w) |
| Langton's Ant | [`cellular_automata/langtons_ant.zig`](cellular_automata/langtons_ant.zig) | O(steps) |
| Game of Life (Numpy-Slice Variant) | [`cellular_automata/game_of_life.zig`](cellular_automata/game_of_life.zig) | O(size²) |
| One-Dimensional Cellular Automata | [`cellular_automata/one_dimensional.zig`](cellular_automata/one_dimensional.zig) | O(n) |
| Nagel-Schreckenberg Traffic Model | [`cellular_automata/nagel_schrekenberg.zig`](cellular_automata/nagel_schrekenberg.zig) | O(updates · cells²) |
| Wa-Tor Simulation | [`cellular_automata/wa_tor.zig`](cellular_automata/wa_tor.zig) | O(iterations · entities) |

### 元胞自动机 (6)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| Conway 生命游戏（下一代） | [`cellular_automata/conways_game_of_life.zig`](cellular_automata/conways_game_of_life.zig) | O(h · w) |
| Langton 蚂蚁 | [`cellular_automata/langtons_ant.zig`](cellular_automata/langtons_ant.zig) | O(steps) |
| 生命游戏（Numpy 切片语义） | [`cellular_automata/game_of_life.zig`](cellular_automata/game_of_life.zig) | O(size²) |
| 一维元胞自动机 | [`cellular_automata/one_dimensional.zig`](cellular_automata/one_dimensional.zig) | O(n) |
| Nagel-Schreckenberg 交通模型 | [`cellular_automata/nagel_schrekenberg.zig`](cellular_automata/nagel_schrekenberg.zig) | O(updates · cells²) |
| Wa-Tor 仿真 | [`cellular_automata/wa_tor.zig`](cellular_automata/wa_tor.zig) | O(iterations · entities) |

### Fractals (5)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Mandelbrot Set Utilities | [`fractals/mandelbrot.zig`](fractals/mandelbrot.zig) | O(max_step) per point |
| Koch Snowflake Utilities | [`fractals/koch_snowflake.zig`](fractals/koch_snowflake.zig) | O(segments · 4^steps) |
| Vicsek Fractal Utilities | [`fractals/vicsek.zig`](fractals/vicsek.zig) | O(5^depth) |
| Sierpinski Triangle Utilities | [`fractals/sierpinski_triangle.zig`](fractals/sierpinski_triangle.zig) | O(3^depth) |
| Julia Sets Utilities | [`fractals/julia_sets.zig`](fractals/julia_sets.zig) | O(iterations · pixels²) |

### 分形 (5)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| Mandelbrot 集工具 | [`fractals/mandelbrot.zig`](fractals/mandelbrot.zig) | 每点 O(max_step) |
| Koch 雪花工具 | [`fractals/koch_snowflake.zig`](fractals/koch_snowflake.zig) | O(segments · 4^steps) |
| Vicsek 分形工具 | [`fractals/vicsek.zig`](fractals/vicsek.zig) | O(5^depth) |
| Sierpinski 三角形工具 | [`fractals/sierpinski_triangle.zig`](fractals/sierpinski_triangle.zig) | O(3^depth) |
| Julia 集工具 | [`fractals/julia_sets.zig`](fractals/julia_sets.zig) | O(iterations · pixels²) |

### Project Euler (110)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Problem 001: Multiples of 3 and 5 | [`project_euler/problem_001.zig`](project_euler/problem_001.zig) | O(1) |
| Problem 002: Even Fibonacci Numbers | [`project_euler/problem_002.zig`](project_euler/problem_002.zig) | O(k) |
| Problem 003: Largest Prime Factor | [`project_euler/problem_003.zig`](project_euler/problem_003.zig) | O(sqrt(n)) |
| Problem 004: Largest Palindrome Product | [`project_euler/problem_004.zig`](project_euler/problem_004.zig) | O(900^2) |
| Problem 005: Smallest Multiple | [`project_euler/problem_005.zig`](project_euler/problem_005.zig) | O(n log n) |
| Problem 006: Sum Square Difference | [`project_euler/problem_006.zig`](project_euler/problem_006.zig) | O(1) |
| Problem 007: 10001st Prime | [`project_euler/problem_007.zig`](project_euler/problem_007.zig) | O(n * sqrt(p_n)) |
| Problem 008: Largest Product in a Series | [`project_euler/problem_008.zig`](project_euler/problem_008.zig) | O(13n) |
| Problem 009: Special Pythagorean Triplet | [`project_euler/problem_009.zig`](project_euler/problem_009.zig) | O(target^2) fast |
| Problem 010: Summation of Primes | [`project_euler/problem_010.zig`](project_euler/problem_010.zig) | O(n * sqrt(n)) |
| Problem 011: Largest Product in a Grid | [`project_euler/problem_011.zig`](project_euler/problem_011.zig) | O(r * c) |
| Problem 012: Highly Divisible Triangular Number | [`project_euler/problem_012.zig`](project_euler/problem_012.zig) | O(k * sqrt(t_k)) |
| Problem 013: Large Sum (First Ten Digits) | [`project_euler/problem_013.zig`](project_euler/problem_013.zig) | O(m * n) |
| Problem 014: Longest Collatz Sequence | [`project_euler/problem_014.zig`](project_euler/problem_014.zig) | O(limit * avg_chain) |
| Problem 015: Lattice Paths | [`project_euler/problem_015.zig`](project_euler/problem_015.zig) | O(n * log n) |
| Problem 016: Power Digit Sum | [`project_euler/problem_016.zig`](project_euler/problem_016.zig) | O(power * digits) |
| Problem 017: Number Letter Counts | [`project_euler/problem_017.zig`](project_euler/problem_017.zig) | O(n) |
| Problem 018: Maximum Path Sum I | [`project_euler/problem_018.zig`](project_euler/problem_018.zig) | O(r^2) |
| Problem 019: Counting Sundays | [`project_euler/problem_019.zig`](project_euler/problem_019.zig) | O(years * 12) |
| Problem 020: Factorial Digit Sum | [`project_euler/problem_020.zig`](project_euler/problem_020.zig) | O(n * digits(n!)) |
| Problem 021: Amicable Numbers | [`project_euler/problem_021.zig`](project_euler/problem_021.zig) | O(n * sqrt(n)) |
| Problem 022: Names Scores | [`project_euler/problem_022.zig`](project_euler/problem_022.zig) | O(m log m + chars) |
| Problem 023: Non-Abundant Sums | [`project_euler/problem_023.zig`](project_euler/problem_023.zig) | O(limit^2) worst |
| Problem 024: Lexicographic Permutations | [`project_euler/problem_024.zig`](project_euler/problem_024.zig) | O(n^2) |
| Problem 025: 1000-digit Fibonacci Number | [`project_euler/problem_025.zig`](project_euler/problem_025.zig) | O(index * digits) |
| Problem 026: Reciprocal Cycles | [`project_euler/problem_026.zig`](project_euler/problem_026.zig) | O((d-n+1) * d^2) |
| Problem 027: Quadratic Primes | [`project_euler/problem_027.zig`](project_euler/problem_027.zig) | O(a*b*run*sqrt(v)) |
| Problem 028: Number Spiral Diagonals | [`project_euler/problem_028.zig`](project_euler/problem_028.zig) | O(n) |
| Problem 029: Distinct Powers | [`project_euler/problem_029.zig`](project_euler/problem_029.zig) | O(n^2 * log n) |
| Problem 030: Digit Fifth Powers | [`project_euler/problem_030.zig`](project_euler/problem_030.zig) | O(1e6 * digits) |
| Problem 031: Coin Sums | [`project_euler/problem_031.zig`](project_euler/problem_031.zig) | O(n * coins) |
| Problem 032: Pandigital Products | [`project_euler/problem_032.zig`](project_euler/problem_032.zig) | O(search_space) |
| Problem 033: Digit Cancelling Fractions | [`project_euler/problem_033.zig`](project_euler/problem_033.zig) | O(1) |
| Problem 034: Digit Factorials | [`project_euler/problem_034.zig`](project_euler/problem_034.zig) | O(limit * digits) |
| Problem 035: Circular Primes | [`project_euler/problem_035.zig`](project_euler/problem_035.zig) | O(limit log log limit + candidates * rotations) |
| Problem 036: Double-Base Palindromes | [`project_euler/problem_036.zig`](project_euler/problem_036.zig) | O(n log n) |
| Problem 037: Truncatable Primes | [`project_euler/problem_037.zig`](project_euler/problem_037.zig) | O(search_horizon · digits · sqrt(n)) |
| Problem 038: Pandigital Multiples | [`project_euler/problem_038.zig`](project_euler/problem_038.zig) | O(1) bounded search |
| Problem 039: Integer Right Triangles | [`project_euler/problem_039.zig`](project_euler/problem_039.zig) | O(p²) |
| Problem 040: Champernowne's Constant | [`project_euler/problem_040.zig`](project_euler/problem_040.zig) | O(n) |
| Problem 041: Pandigital Prime | [`project_euler/problem_041.zig`](project_euler/problem_041.zig) | O(n! · sqrt(10ⁿ)) |
| Problem 042: Coded Triangle Numbers | [`project_euler/problem_042.zig`](project_euler/problem_042.zig) | O(len(data)) |
| Problem 043: Sub-string Divisibility | [`project_euler/problem_043.zig`](project_euler/problem_043.zig) | O(10!) worst-case with pruning |
| Problem 044: Pentagon Numbers | [`project_euler/problem_044.zig`](project_euler/problem_044.zig) | O(limit²) |
| Problem 045: Triangular, Pentagonal, and Hexagonal | [`project_euler/problem_045.zig`](project_euler/problem_045.zig) | O(search span) |
| Problem 046: Goldbach's Other Conjecture | [`project_euler/problem_046.zig`](project_euler/problem_046.zig) | O(search_horizon · sqrt(n)) |
| Problem 047: Distinct Prime Factors | [`project_euler/problem_047.zig`](project_euler/problem_047.zig) | roughly O(answer · span · sqrt(answer)) |
| Problem 048: Self Powers | [`project_euler/problem_048.zig`](project_euler/problem_048.zig) | O(limit log limit) |
| Problem 049: Prime Permutations | [`project_euler/problem_049.zig`](project_euler/problem_049.zig) | O(p² log p) |
| Problem 050: Consecutive Prime Sum | [`project_euler/problem_050.zig`](project_euler/problem_050.zig) | O(p²) |
| Problem 051: Prime Digit Replacements | [`project_euler/problem_051.zig`](project_euler/problem_051.zig) | roughly O(limit log log limit) |
| Problem 052: Permuted Multiples | [`project_euler/problem_052.zig`](project_euler/problem_052.zig) | unbounded search; practical runtime is small |
| Problem 053: Combinatoric Selections | [`project_euler/problem_053.zig`](project_euler/problem_053.zig) | O(max_n²) |
| Problem 054: Poker Hands | [`project_euler/problem_054.zig`](project_euler/problem_054.zig) | O(lines) |
| Problem 055: Lychrel Numbers | [`project_euler/problem_055.zig`](project_euler/problem_055.zig) | O(limit · iterations · digits) |
| Problem 056: Powerful Digit Sum | [`project_euler/problem_056.zig`](project_euler/problem_056.zig) | O(a · b · digits) |
| Problem 057: Square Root Convergents | [`project_euler/problem_057.zig`](project_euler/problem_057.zig) | O(n · digits) |
| Problem 058: Spiral Primes | [`project_euler/problem_058.zig`](project_euler/problem_058.zig) | roughly O(side · sqrt(n)) |
| Problem 059: XOR Decryption | [`project_euler/problem_059.zig`](project_euler/problem_059.zig) | O(26^3 · n · common_words) |
| Problem 062: Cubic Permutations | [`project_euler/problem_062.zig`](project_euler/problem_062.zig) | unbounded search; practical runtime is small |
| Problem 063: Powerful Digit Counts | [`project_euler/problem_063.zig`](project_euler/problem_063.zig) | O(max_base · max_power²) |
| Problem 064: Odd Period Square Roots | [`project_euler/problem_064.zig`](project_euler/problem_064.zig) | O(n · average_period) |
| Problem 065: Convergents of e | [`project_euler/problem_065.zig`](project_euler/problem_065.zig) | O(max_n · bigint_digits) |
| Problem 067: Maximum Path Sum II | [`project_euler/problem_067.zig`](project_euler/problem_067.zig) | O(values) |
| Problem 069: Totient Maximum | [`project_euler/problem_069.zig`](project_euler/problem_069.zig) | O(limit log log limit) |
| Problem 071: Ordered Fractions | [`project_euler/problem_071.zig`](project_euler/problem_071.zig) | O(limit) |
| Problem 072: Counting Fractions | [`project_euler/problem_072.zig`](project_euler/problem_072.zig) | O(limit log log limit) |
| Problem 073: Counting Fractions in a Range | [`project_euler/problem_073.zig`](project_euler/problem_073.zig) | O(max_d² log max_d) |
| Problem 074: Digit Factorial Chains | [`project_euler/problem_074.zig`](project_euler/problem_074.zig) | amortized O(max_start · average_chain) |
| Problem 075: Singular Integer Right Triangles | [`project_euler/problem_075.zig`](project_euler/problem_075.zig) | roughly O(limit log limit) |
| Problem 076: Counting Summations | [`project_euler/problem_076.zig`](project_euler/problem_076.zig) | O(m²) |
| Problem 077: Prime Summations | [`project_euler/problem_077.zig`](project_euler/problem_077.zig) | roughly O(answer² / log answer) |
| Problem 079: Passcode Derivation | [`project_euler/problem_079.zig`](project_euler/problem_079.zig) | O(logins + digits²) |
| Problem 081: Path Sum Two Ways | [`project_euler/problem_081.zig`](project_euler/problem_081.zig) | O(rows · cols) |
| Problem 082: Path Sum Three Ways | [`project_euler/problem_082.zig`](project_euler/problem_082.zig) | O(rows · cols) |
| Problem 085: Counting Rectangles | [`project_euler/problem_085.zig`](project_euler/problem_085.zig) | O(sqrt(target)) |
| Problem 087: Prime Power Triples | [`project_euler/problem_087.zig`](project_euler/problem_087.zig) | roughly O(pi(n^1/2) · pi(n^1/3) · pi(n^1/4)) |
| Problem 089: Roman Numerals | [`project_euler/problem_089.zig`](project_euler/problem_089.zig) | O(total_chars) |
| Problem 091: Right Triangles with Integer Coordinates | [`project_euler/problem_091.zig`](project_euler/problem_091.zig) | O(limit^4) |
| Problem 092: Square Digit Chains | [`project_euler/problem_092.zig`](project_euler/problem_092.zig) | O(number · digits) |
| Problem 094: Almost Equilateral Triangles | [`project_euler/problem_094.zig`](project_euler/problem_094.zig) | O(number of valid triangles) |
| Problem 095: Amicable Chains | [`project_euler/problem_095.zig`](project_euler/problem_095.zig) | roughly O(max_num log max_num) |
| Problem 097: Large Non-Mersenne Prime | [`project_euler/problem_097.zig`](project_euler/problem_097.zig) | O(log exponent) |
| Problem 099: Largest Exponential | [`project_euler/problem_099.zig`](project_euler/problem_099.zig) | O(lines) |
| Problem 100: Arranged Probability | [`project_euler/problem_100.zig`](project_euler/problem_100.zig) | O(recurrence steps) |
| Problem 102: Triangle Containment | [`project_euler/problem_102.zig`](project_euler/problem_102.zig) | O(lines) |
| Problem 107: Minimal Network | [`project_euler/problem_107.zig`](project_euler/problem_107.zig) | O(n^3) |
| Problem 109: Darts | [`project_euler/problem_109.zig`](project_euler/problem_109.zig) | O(doubles · throws^2) |
| Problem 112: Bouncy Numbers | [`project_euler/problem_112.zig`](project_euler/problem_112.zig) | O(answer · digits) |
| Problem 113: Non-Bouncy Numbers | [`project_euler/problem_113.zig`](project_euler/problem_113.zig) | O(n^2) |
| Problem 114: Counting Block Combinations I | [`project_euler/problem_114.zig`](project_euler/problem_114.zig) | O(length^3) |
| Problem 115: Counting Block Combinations II | [`project_euler/problem_115.zig`](project_euler/problem_115.zig) | O(answer^3) in the direct recurrence |
| Problem 116: Red, Green or Blue Tiles | [`project_euler/problem_116.zig`](project_euler/problem_116.zig) | O(length^2) |
| Problem 117: Red, Green, and Blue Tiles | [`project_euler/problem_117.zig`](project_euler/problem_117.zig) | O(length^2) |
| Problem 120: Square Remainders | [`project_euler/problem_120.zig`](project_euler/problem_120.zig) | O(n) |
| Problem 121: Disc Game Prize Fund | [`project_euler/problem_121.zig`](project_euler/problem_121.zig) | O(num_turns^2) |
| Problem 123: Prime Square Remainders | [`project_euler/problem_123.zig`](project_euler/problem_123.zig) | roughly O(answer · pi(sqrt(p_n))) |
| Problem 125: Palindromic Sums | [`project_euler/problem_125.zig`](project_euler/problem_125.zig) | roughly O(sqrt(limit)^2) |
| Problem 131: Prime Cube Partnership | [`project_euler/problem_131.zig`](project_euler/problem_131.zig) | roughly O(number of candidates · sqrt(max_prime)) |
| Problem 145: Reversible Numbers | [`project_euler/problem_145.zig`](project_euler/problem_145.zig) | fast recursive digit-pair enumeration |
| Problem 173: Square Laminae Count | [`project_euler/problem_173.zig`](project_euler/problem_173.zig) | O(limit) |
| Problem 174: Laminae Type Count | [`project_euler/problem_174.zig`](project_euler/problem_174.zig) | roughly O(number of laminae) |
| Problem 188: Hyperexponentiation | [`project_euler/problem_188.zig`](project_euler/problem_188.zig) | O(height · log exponent) |
| Problem 190: Maximising a Weighted Product | [`project_euler/problem_190.zig`](project_euler/problem_190.zig) | O(n^2) |
| Problem 191: Prize Strings | [`project_euler/problem_191.zig`](project_euler/problem_191.zig) | O(days) |
| Problem 301: Nim | [`project_euler/problem_301.zig`](project_euler/problem_301.zig) | O(exponent) |
| Problem 203: Squarefree Binomial Coefficients | [`project_euler/problem_203.zig`](project_euler/problem_203.zig) | acceptable direct translation for n <= 51 |
| Problem 205: Dice Game | [`project_euler/problem_205.zig`](project_euler/problem_205.zig) | O(dice_number · sides_number · max_total) |
| Problem 206: Concealed Square | [`project_euler/problem_206.zig`](project_euler/problem_206.zig) | narrow brute-force over candidates ending in 3 or 7 |
| Problem 207: Integer Partition Equations | [`project_euler/problem_207.zig`](project_euler/problem_207.zig) | O(sqrt(answer)) |

### Project Euler (110)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 第 001 题：3 和 5 的倍数和 | [`project_euler/problem_001.zig`](project_euler/problem_001.zig) | O(1) |
| 第 002 题：偶数 Fibonacci 和 | [`project_euler/problem_002.zig`](project_euler/problem_002.zig) | O(k) |
| 第 003 题：最大质因子 | [`project_euler/problem_003.zig`](project_euler/problem_003.zig) | O(sqrt(n)) |
| 第 004 题：最大回文乘积 | [`project_euler/problem_004.zig`](project_euler/problem_004.zig) | O(900^2) |
| 第 005 题：最小公倍整除数 | [`project_euler/problem_005.zig`](project_euler/problem_005.zig) | O(n log n) |
| 第 006 题：平方和差值 | [`project_euler/problem_006.zig`](project_euler/problem_006.zig) | O(1) |
| 第 007 题：第 n 个质数 | [`project_euler/problem_007.zig`](project_euler/problem_007.zig) | O(n * sqrt(p_n)) |
| 第 008 题：序列中最大相邻乘积 | [`project_euler/problem_008.zig`](project_euler/problem_008.zig) | O(13n) |
| 第 009 题：特殊勾股三元组 | [`project_euler/problem_009.zig`](project_euler/problem_009.zig) | 快速版 O(target^2) |
| 第 010 题：质数求和 | [`project_euler/problem_010.zig`](project_euler/problem_010.zig) | O(n * sqrt(n)) |
| 第 011 题：网格中的最大乘积 | [`project_euler/problem_011.zig`](project_euler/problem_011.zig) | O(r * c) |
| 第 012 题：高因数三角数 | [`project_euler/problem_012.zig`](project_euler/problem_012.zig) | O(k * sqrt(t_k)) |
| 第 013 题：大数求和前十位 | [`project_euler/problem_013.zig`](project_euler/problem_013.zig) | O(m * n) |
| 第 014 题：最长 Collatz 序列 | [`project_euler/problem_014.zig`](project_euler/problem_014.zig) | O(limit * avg_chain) |
| 第 015 题：网格路径计数 | [`project_euler/problem_015.zig`](project_euler/problem_015.zig) | O(n * log n) |
| 第 016 题：2^n 数位和 | [`project_euler/problem_016.zig`](project_euler/problem_016.zig) | O(power * digits) |
| 第 017 题：数字英文计数 | [`project_euler/problem_017.zig`](project_euler/problem_017.zig) | O(n) |
| 第 018 题：最大路径和 I | [`project_euler/problem_018.zig`](project_euler/problem_018.zig) | O(r^2) |
| 第 019 题：周日计数 | [`project_euler/problem_019.zig`](project_euler/problem_019.zig) | O(years * 12) |
| 第 020 题：阶乘数位和 | [`project_euler/problem_020.zig`](project_euler/problem_020.zig) | O(n * digits(n!)) |
| 第 021 题：亲和数 | [`project_euler/problem_021.zig`](project_euler/problem_021.zig) | O(n * sqrt(n)) |
| 第 022 题：姓名分数 | [`project_euler/problem_022.zig`](project_euler/problem_022.zig) | O(m log m + chars) |
| 第 023 题：非盈数和 | [`project_euler/problem_023.zig`](project_euler/problem_023.zig) | 最坏 O(limit^2) |
| 第 024 题：字典序排列 | [`project_euler/problem_024.zig`](project_euler/problem_024.zig) | O(n^2) |
| 第 025 题：首个 1000 位 Fibonacci 下标 | [`project_euler/problem_025.zig`](project_euler/problem_025.zig) | O(index * digits) |
| 第 026 题：循环小数周期 | [`project_euler/problem_026.zig`](project_euler/problem_026.zig) | O((d-n+1) * d^2) |
| 第 027 题：二次多项式与质数 | [`project_euler/problem_027.zig`](project_euler/problem_027.zig) | O(a*b*run*sqrt(v)) |
| 第 028 题：数字螺旋对角线和 | [`project_euler/problem_028.zig`](project_euler/problem_028.zig) | O(n) |
| 第 029 题：不同幂的个数 | [`project_euler/problem_029.zig`](project_euler/problem_029.zig) | O(n^2 * log n) |
| 第 030 题：数位五次幂 | [`project_euler/problem_030.zig`](project_euler/problem_030.zig) | O(1e6 * digits) |
| 第 031 题：硬币组合数 | [`project_euler/problem_031.zig`](project_euler/problem_031.zig) | O(n * coins) |
| 第 032 题：全数字乘积 | [`project_euler/problem_032.zig`](project_euler/problem_032.zig) | O(search_space) |
| 第 033 题：消位分数 | [`project_euler/problem_033.zig`](project_euler/problem_033.zig) | O(1) |
| 第 034 题：各位阶乘和 | [`project_euler/problem_034.zig`](project_euler/problem_034.zig) | O(limit * digits) |
| 第 035 题：循环质数 | [`project_euler/problem_035.zig`](project_euler/problem_035.zig) | O(limit log log limit + 候选旋转检查) |
| 第 036 题：双进制回文数 | [`project_euler/problem_036.zig`](project_euler/problem_036.zig) | O(n log n) |
| 第 037 题：可截断质数 | [`project_euler/problem_037.zig`](project_euler/problem_037.zig) | O(search_horizon · digits · sqrt(n)) |
| 第 038 题：Pandigital 倍数 | [`project_euler/problem_038.zig`](project_euler/problem_038.zig) | 有界搜索 O(1) |
| 第 039 题：整数直角三角形 | [`project_euler/problem_039.zig`](project_euler/problem_039.zig) | O(p²) |
| 第 040 题：Champernowne 常数 | [`project_euler/problem_040.zig`](project_euler/problem_040.zig) | O(n) |
| 第 041 题：Pandigital 质数 | [`project_euler/problem_041.zig`](project_euler/problem_041.zig) | O(n! · sqrt(10ⁿ)) |
| 第 042 题：编码三角词 | [`project_euler/problem_042.zig`](project_euler/problem_042.zig) | O(len(data)) |
| 第 043 题：子串整除性 | [`project_euler/problem_043.zig`](project_euler/problem_043.zig) | 最坏 O(10!)，含剪枝 |
| 第 044 题：五边形数 | [`project_euler/problem_044.zig`](project_euler/problem_044.zig) | O(limit²) |
| 第 045 题：三角/五边形/六边形数 | [`project_euler/problem_045.zig`](project_euler/problem_045.zig) | O(search span) |
| 第 046 题：Goldbach 另一猜想 | [`project_euler/problem_046.zig`](project_euler/problem_046.zig) | O(search_horizon · sqrt(n)) |
| 第 047 题：不同质因子 | [`project_euler/problem_047.zig`](project_euler/problem_047.zig) | 约 O(answer · span · sqrt(answer)) |
| 第 048 题：自幂和末十位 | [`project_euler/problem_048.zig`](project_euler/problem_048.zig) | O(limit log limit) |
| 第 049 题：质数排列序列 | [`project_euler/problem_049.zig`](project_euler/problem_049.zig) | O(p² log p) |
| 第 050 题：连续质数和 | [`project_euler/problem_050.zig`](project_euler/problem_050.zig) | O(p²) |
| 第 051 题：质数数位替换族 | [`project_euler/problem_051.zig`](project_euler/problem_051.zig) | 约 O(limit log log limit) |
| 第 052 题：排列倍数 | [`project_euler/problem_052.zig`](project_euler/problem_052.zig) | 无界搜索；实际运行很小 |
| 第 053 题：组合数筛选 | [`project_euler/problem_053.zig`](project_euler/problem_053.zig) | O(max_n²) |
| 第 054 题：扑克牌比较 | [`project_euler/problem_054.zig`](project_euler/problem_054.zig) | O(lines) |
| 第 055 题：Lychrel 数 | [`project_euler/problem_055.zig`](project_euler/problem_055.zig) | O(limit · iterations · digits) |
| 第 056 题：最大数位和 | [`project_euler/problem_056.zig`](project_euler/problem_056.zig) | O(a · b · digits) |
| 第 057 题：平方根连分数展开 | [`project_euler/problem_057.zig`](project_euler/problem_057.zig) | O(n · digits) |
| 第 058 题：螺旋素数比例 | [`project_euler/problem_058.zig`](project_euler/problem_058.zig) | 约 O(side · sqrt(n)) |
| 第 059 题：XOR 解密 | [`project_euler/problem_059.zig`](project_euler/problem_059.zig) | O(26^3 · n · common_words) |
| 第 062 题：立方排列 | [`project_euler/problem_062.zig`](project_euler/problem_062.zig) | 无界搜索；实际运行很小 |
| 第 063 题：n 次幂的 n 位数计数 | [`project_euler/problem_063.zig`](project_euler/problem_063.zig) | O(max_base · max_power²) |
| 第 064 题：奇周期平方根连分数 | [`project_euler/problem_064.zig`](project_euler/problem_064.zig) | O(n · average_period) |
| 第 065 题：e 的连分数收敛项 | [`project_euler/problem_065.zig`](project_euler/problem_065.zig) | O(max_n · bigint_digits) |
| 第 067 题：最大路径和 II | [`project_euler/problem_067.zig`](project_euler/problem_067.zig) | O(values) |
| 第 069 题：欧拉函数最大比值 | [`project_euler/problem_069.zig`](project_euler/problem_069.zig) | O(limit log log limit) |
| 第 071 题：有序分数 | [`project_euler/problem_071.zig`](project_euler/problem_071.zig) | O(limit) |
| 第 072 题：计数化简分数 | [`project_euler/problem_072.zig`](project_euler/problem_072.zig) | O(limit log log limit) |
| 第 073 题：区间内分数计数 | [`project_euler/problem_073.zig`](project_euler/problem_073.zig) | O(max_d² log max_d) |
| 第 074 题：数位阶乘链 | [`project_euler/problem_074.zig`](project_euler/problem_074.zig) | 均摊 O(max_start · average_chain) |
| 第 075 题：恰有一种勾股三角形的周长 | [`project_euler/problem_075.zig`](project_euler/problem_075.zig) | 约 O(limit log limit) |
| 第 076 题：整数拆分计数 | [`project_euler/problem_076.zig`](project_euler/problem_076.zig) | O(m²) |
| 第 077 题：质数拆分和 | [`project_euler/problem_077.zig`](project_euler/problem_077.zig) | 约 O(answer² / log answer) |
| 第 079 题：口令推导 | [`project_euler/problem_079.zig`](project_euler/problem_079.zig) | O(logins + digits²) |
| 第 081 题：两向路径最小和 | [`project_euler/problem_081.zig`](project_euler/problem_081.zig) | O(rows · cols) |
| 第 082 题：三向路径最小和 | [`project_euler/problem_082.zig`](project_euler/problem_082.zig) | O(rows · cols) |
| 第 085 题：矩形计数最接近目标 | [`project_euler/problem_085.zig`](project_euler/problem_085.zig) | O(sqrt(target)) |
| 第 087 题：质数幂和三元组 | [`project_euler/problem_087.zig`](project_euler/problem_087.zig) | 约 O(pi(n^1/2) · pi(n^1/3) · pi(n^1/4)) |
| 第 089 题：罗马数字最简写法 | [`project_euler/problem_089.zig`](project_euler/problem_089.zig) | O(total_chars) |
| 第 091 题：整点直角三角形 | [`project_euler/problem_091.zig`](project_euler/problem_091.zig) | O(limit^4) |
| 第 092 题：平方数位链 | [`project_euler/problem_092.zig`](project_euler/problem_092.zig) | O(number · digits) |
| 第 094 题：近等边三角形 | [`project_euler/problem_094.zig`](project_euler/problem_094.zig) | O(number of valid triangles) |
| 第 095 题：亲和链 | [`project_euler/problem_095.zig`](project_euler/problem_095.zig) | 约 O(max_num log max_num) |
| 第 097 题：大非梅森素数末位 | [`project_euler/problem_097.zig`](project_euler/problem_097.zig) | O(log exponent) |
| 第 099 题：最大指数形式 | [`project_euler/problem_099.zig`](project_euler/problem_099.zig) | O(lines) |
| 第 100 题：蓝红球概率排列 | [`project_euler/problem_100.zig`](project_euler/problem_100.zig) | O(recurrence steps) |
| 第 102 题：原点是否在三角形内 | [`project_euler/problem_102.zig`](project_euler/problem_102.zig) | O(lines) |
| 第 107 题：最小网络 | [`project_euler/problem_107.zig`](project_euler/problem_107.zig) | O(n^3) |
| 第 109 题：飞镖结账组合 | [`project_euler/problem_109.zig`](project_euler/problem_109.zig) | O(doubles · throws^2) |
| 第 112 题：弹跳数 | [`project_euler/problem_112.zig`](project_euler/problem_112.zig) | O(answer · digits) |
| 第 113 题：非弹跳数 | [`project_euler/problem_113.zig`](project_euler/problem_113.zig) | O(n^2) |
| 第 114 题：块填充计数 I | [`project_euler/problem_114.zig`](project_euler/problem_114.zig) | O(length^3) |
| 第 115 题：块填充计数 II | [`project_euler/problem_115.zig`](project_euler/problem_115.zig) | 直接递推下为 O(answer^3) |
| 第 116 题：单色彩砖替换 | [`project_euler/problem_116.zig`](project_euler/problem_116.zig) | O(length^2) |
| 第 117 题：多色彩砖铺法 | [`project_euler/problem_117.zig`](project_euler/problem_117.zig) | O(length^2) |
| 第 120 题：平方余数和 | [`project_euler/problem_120.zig`](project_euler/problem_120.zig) | O(n) |
| 第 121 题：抽盘游戏最大奖金 | [`project_euler/problem_121.zig`](project_euler/problem_121.zig) | O(num_turns^2) |
| 第 123 题：质数平方余数 | [`project_euler/problem_123.zig`](project_euler/problem_123.zig) | 约 O(answer · pi(sqrt(p_n))) |
| 第 125 题：回文连续平方和 | [`project_euler/problem_125.zig`](project_euler/problem_125.zig) | 约 O(sqrt(limit)^2) |
| 第 131 题：立方关系素数 | [`project_euler/problem_131.zig`](project_euler/problem_131.zig) | 约 O(number of candidates · sqrt(max_prime)) |
| 第 145 题：可逆数统计 | [`project_euler/problem_145.zig`](project_euler/problem_145.zig) | 快速递归枚举数位对 |
| 第 173 题：方环数量统计 | [`project_euler/problem_173.zig`](project_euler/problem_173.zig) | O(limit) |
| 第 174 题：方环类型统计 | [`project_euler/problem_174.zig`](project_euler/problem_174.zig) | 约 O(number of laminae) |
| 第 188 题：超指数幂末位 | [`project_euler/problem_188.zig`](project_euler/problem_188.zig) | O(height · log exponent) |
| 第 190 题：加权乘积最大化 | [`project_euler/problem_190.zig`](project_euler/problem_190.zig) | O(n^2) |
| 第 191 题：奖励出勤字符串 | [`project_euler/problem_191.zig`](project_euler/problem_191.zig) | O(days) |
| 第 301 题：Nim 必胜败局计数 | [`project_euler/problem_301.zig`](project_euler/problem_301.zig) | O(exponent) |
| 第 203 题：无平方因子二项式系数和 | [`project_euler/problem_203.zig`](project_euler/problem_203.zig) | `n <= 51` 下可接受的直接移植 |
| 第 205 题：骰子博弈胜率 | [`project_euler/problem_205.zig`](project_euler/problem_205.zig) | O(dice_number · sides_number · max_total) |
| 第 206 题：隐藏平方数 | [`project_euler/problem_206.zig`](project_euler/problem_206.zig) | 只搜索末位为 3/7 的窄区间候选 |
| 第 207 题：整数划分方程 | [`project_euler/problem_207.zig`](project_euler/problem_207.zig) | O(sqrt(answer)) |
