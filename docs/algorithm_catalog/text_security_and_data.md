# Text Security And Data / 文本、安全与数据

- Source of truth: the detailed catalog sections from the pre-split root README.
- 数据来源：拆分前根 README 的详细目录条目。

### Strings (59)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Palindrome Check | [`strings/palindrome.zig`](strings/palindrome.zig) | O(n) |
| Reverse Words | [`strings/reverse_words.zig`](strings/reverse_words.zig) | O(n) |
| Anagram Check | [`strings/anagram.zig`](strings/anagram.zig) | O(n) |
| Anagrams (Dictionary-Driven) | [`strings/anagrams.zig`](strings/anagrams.zig) | O(d · n log k) |
| Check Anagrams (Compatibility Wrapper) | [`strings/check_anagrams.zig`](strings/check_anagrams.zig) | O(n) |
| Hamming Distance | [`strings/hamming_distance.zig`](strings/hamming_distance.zig) | O(n) |
| Naive String Search | [`strings/naive_string_search.zig`](strings/naive_string_search.zig) | O(n·m) |
| Knuth-Morris-Pratt | [`strings/knuth_morris_pratt.zig`](strings/knuth_morris_pratt.zig) | O(n + m) |
| Rabin-Karp | [`strings/rabin_karp.zig`](strings/rabin_karp.zig) | O(n + m) avg |
| Z-Function | [`strings/z_function.zig`](strings/z_function.zig) | O(n + m) |
| Levenshtein Distance | [`strings/levenshtein_distance.zig`](strings/levenshtein_distance.zig) | O(m × n) |
| Edit Distance (Compatibility Wrapper) | [`strings/edit_distance.zig`](strings/edit_distance.zig) | O(m × n) |
| Damerau-Levenshtein Distance | [`strings/damerau_levenshtein_distance.zig`](strings/damerau_levenshtein_distance.zig) | O(m × n) |
| Frequency Finder | [`strings/frequency_finder.zig`](strings/frequency_finder.zig) | O(n + 26 log 26) |
| Is Pangram | [`strings/is_pangram.zig`](strings/is_pangram.zig) | O(n) |
| Polish National ID Validator | [`strings/is_polish_national_id.zig`](strings/is_polish_national_id.zig) | O(n) |
| Spain National ID Validator | [`strings/is_spain_national_id.zig`](strings/is_spain_national_id.zig) | O(n) |
| Aho-Corasick | [`strings/aho_corasick.zig`](strings/aho_corasick.zig) | O(text + matches) query |
| Autocomplete Using Trie | [`strings/autocomplete_using_trie.zig`](strings/autocomplete_using_trie.zig) | O(prefix + results) |
| Suffix Array | [`strings/suffix_array.zig`](strings/suffix_array.zig) | O(n log² n) build |
| Run-Length Encoding | [`strings/run_length_encoding.zig`](strings/run_length_encoding.zig) | O(n) encode/decode |
| Barcode Validator | [`strings/barcode_validator.zig`](strings/barcode_validator.zig) | O(d) |
| Camel Case to Snake Case | [`strings/camel_case_to_snake_case.zig`](strings/camel_case_to_snake_case.zig) | O(n) |
| Palindrome Rearrangement Check | [`strings/can_string_be_rearranged_as_palindrome.zig`](strings/can_string_be_rearranged_as_palindrome.zig) | O(n) |
| Capitalize | [`strings/capitalize.zig`](strings/capitalize.zig) | O(n) |
| Count Vowels | [`strings/count_vowels.zig`](strings/count_vowels.zig) | O(n) |
| Credit Card Validator | [`strings/credit_card_validator.zig`](strings/credit_card_validator.zig) | O(n) |
| Detecting English Programmatically | [`strings/detecting_english_programmatically.zig`](strings/detecting_english_programmatically.zig) | O(n + d) |
| DNA Complement | [`strings/dna.zig`](strings/dna.zig) | O(n) |
| Indian Phone Validator | [`strings/indian_phone_validator.zig`](strings/indian_phone_validator.zig) | O(n) |
| Contains Unique Characters | [`strings/is_contains_unique_chars.zig`](strings/is_contains_unique_chars.zig) | O(n) |
| Is Isogram | [`strings/is_isogram.zig`](strings/is_isogram.zig) | O(n) |
| Sri Lankan Phone Validator | [`strings/is_srilankan_phone_number.zig`](strings/is_srilankan_phone_number.zig) | O(n) |
| Email Address Validator | [`strings/is_valid_email_address.zig`](strings/is_valid_email_address.zig) | O(n) |
| Jaro-Winkler Similarity | [`strings/jaro_winkler.zig`](strings/jaro_winkler.zig) | O(n²) |
| Join Strings | [`strings/join.zig`](strings/join.zig) | O(total_len) |
| Lowercase ASCII | [`strings/lower.zig`](strings/lower.zig) | O(n) |
| N-Gram | [`strings/ngram.zig`](strings/ngram.zig) | O(n · k) |
| Split String | [`strings/split.zig`](strings/split.zig) | O(n) |
| String Switch Case | [`strings/string_switch_case.zig`](strings/string_switch_case.zig) | O(n) |
| Text Justification | [`strings/text_justification.zig`](strings/text_justification.zig) | O(m · n) |
| Uppercase ASCII | [`strings/upper.zig`](strings/upper.zig) | O(n) |
| Alternative String Arrange | [`strings/alternative_string_arrange.zig`](strings/alternative_string_arrange.zig) | O(n + m) |
| Boyer-Moore Search | [`strings/boyer_moore_search.zig`](strings/boyer_moore_search.zig) | O(n·m) worst |
| Bitap String Match | [`strings/bitap_string_match.zig`](strings/bitap_string_match.zig) | O(n) core |
| Prefix Function | [`strings/prefix_function.zig`](strings/prefix_function.zig) | O(n) |
| Remove Duplicate Words | [`strings/remove_duplicate.zig`](strings/remove_duplicate.zig) | O(k log k + n) |
| Reverse Letters | [`strings/reverse_letters.zig`](strings/reverse_letters.zig) | O(n) |
| Snake Case to Camel/Pascal Case | [`strings/snake_case_to_camel_pascal_case.zig`](strings/snake_case_to_camel_pascal_case.zig) | O(n) |
| Strip | [`strings/strip.zig`](strings/strip.zig) | O(n) |
| Title Case | [`strings/title.zig`](strings/title.zig) | O(n) |
| Word Occurrence | [`strings/word_occurrence.zig`](strings/word_occurrence.zig) | O(n) |
| Pig Latin | [`strings/pig_latin.zig`](strings/pig_latin.zig) | O(n) |
| Wildcard Pattern Matching | [`strings/wildcard_pattern_matching.zig`](strings/wildcard_pattern_matching.zig) | O(n × m) |
| Wave String | [`strings/wave_string.zig`](strings/wave_string.zig) | O(n²) |
| Top K Frequent Words | [`strings/top_k_frequent_words.zig`](strings/top_k_frequent_words.zig) | O(n + u log u) |
| Manacher | [`strings/manacher.zig`](strings/manacher.zig) | O(n) |
| Min Cost String Conversion | [`strings/min_cost_string_conversion.zig`](strings/min_cost_string_conversion.zig) | O(m × n) |
| Word Patterns | [`strings/word_patterns.zig`](strings/word_patterns.zig) | O(n) |

## Quick Start

```bash
# Install Zig 0.15.2 (no root required)
# See https://ziglang.org/download/

# Run all tests
zig build test

# Run a single algorithm's tests
zig test sorts/bubble_sort.zig
```

## Project Structure

```
TheAlgorithms-Zig/
├── build.zig                # Build script — registers all test files
├── build.zig.zon            # Package manifest
├── sorts/                   # 50 sorting algorithms
├── searches/                # 16 search algorithms
├── maths/                   # 144 math algorithms
├── data_structures/         # 101 data structure implementations
├── dynamic_programming/     # 54 dynamic programming algorithms
├── graphs/                  # 46 graph algorithms
├── bit_manipulation/        # 27 bit manipulation algorithms
├── conversions/             # 27 conversion algorithms
├── boolean_algebra/         # 12 boolean algebra algorithms
├── divide_and_conquer/      # 11 divide-and-conquer algorithms
├── linear_algebra/          # 11 linear algebra algorithms
├── physics/                 # 29 physics algorithms
├── electronics/             # 19 electronics algorithms
├── audio_filters/           # 2 audio filter algorithms
├── financial/               # 7 financial algorithms
├── scheduling/              # 8 scheduling algorithms
├── ciphers/                 # 47 cipher algorithms
├── hashing/                 # 12 hashing algorithms
├── data_compression/        # 8 data compression algorithms
├── cellular_automata/       # 6 cellular automata algorithms
├── fractals/                # 5 fractal algorithms
├── project_euler/           # 54 project euler algorithms
├── strings/                 # 59 string algorithms
├── greedy_methods/          # 8 greedy algorithms
├── matrix/                  # 20 matrix algorithms
├── geodesy/                 # 2 geodesy algorithms
├── geometry/                # 1 geometry algorithm
├── knapsack/                # 3 knapsack algorithms
└── backtracking/            # 21 backtracking algorithms
```

## Development

**Requirements:** Zig ≥ 0.15.2

Each algorithm file is self-contained: implementation + tests in one file. To add a new algorithm:

1. Create `<category>/<algorithm_name>.zig`
2. Implement the algorithm as a `pub fn` with comptime generics where appropriate
3. Add `test` blocks at the bottom of the file
4. Register the file in `build.zig`'s `test_files` array
5. Run `zig build test` to verify

## Vibe Coding Experiment

This project doubles as a research experiment on AI-assisted development. Every algorithm records:

- How many AI attempts were needed to produce compilable code
- What error categories appeared (type inference, allocator, comptime, etc.)
- How many lines of manual human fixes were required

Results will be published in `EXPERIMENT_LOG.md` as the project progresses.

## Contributing

Contributions welcome! Please ensure:

- [ ] `zig build test` passes
- [ ] Each file includes a reference comment linking to the Python source
- [ ] Complexity is documented in the doc comment

## License

MIT

---

# TheAlgorithms - Zig（简体中文）

经典算法的 Zig 实现，每个算法内置单元测试。灵感来自 [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python)。

本项目同时是一个 **vibe coding 实验**：用 AI 将 Python 算法翻译为 Zig——一门作者此前零基础的语言——并记录 AI 的成功率、报错模式和人工干预成本。

Phase 6 统计说明（2026-03-10）：`graphs` 已完成真实缺项清零，`maths` 已推进到数值分析批次，`project_euler` 现已在上一批基础上继续扩展到第 `145`、`188`、`190` 题。当前 [`build.zig`](/root/projects/TheAlgorithms-Zig/build.zig) 已注册 `902` 个算法；按修正后的分类上限口径，其中 `894` 个计入目标，剩余计划缺口为 `22`。

---

## 功能特性

- 每个算法独立为一个 `.zig` 文件，实现与测试写在同一文件中
- 使用 comptime 泛型，大多数算法支持 `i32`、`f64` 等数值类型
- 零外部依赖，仅使用 Zig 标准库
- 统一测试入口：`zig build test` 一键运行所有算法测试
- 基于 **Zig 0.15.2** 测试通过

## 算法目录

### 字符串 (59)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 回文检查 | [`strings/palindrome.zig`](strings/palindrome.zig) | O(n) |
| 单词反转 | [`strings/reverse_words.zig`](strings/reverse_words.zig) | O(n) |
| 异位词检查 | [`strings/anagram.zig`](strings/anagram.zig) | O(n) |
| 字典异位词检索 | [`strings/anagrams.zig`](strings/anagrams.zig) | O(d · n log k) |
| 异位词检查（兼容包装） | [`strings/check_anagrams.zig`](strings/check_anagrams.zig) | O(n) |
| 汉明距离 | [`strings/hamming_distance.zig`](strings/hamming_distance.zig) | O(n) |
| 朴素字符串搜索 | [`strings/naive_string_search.zig`](strings/naive_string_search.zig) | O(n·m) |
| KMP 字符串搜索 | [`strings/knuth_morris_pratt.zig`](strings/knuth_morris_pratt.zig) | O(n + m) |
| Rabin-Karp | [`strings/rabin_karp.zig`](strings/rabin_karp.zig) | O(n + m) 平均 |
| Z 函数 | [`strings/z_function.zig`](strings/z_function.zig) | O(n + m) |
| Levenshtein 距离 | [`strings/levenshtein_distance.zig`](strings/levenshtein_distance.zig) | O(m × n) |
| 编辑距离（兼容包装） | [`strings/edit_distance.zig`](strings/edit_distance.zig) | O(m × n) |
| Damerau-Levenshtein 距离 | [`strings/damerau_levenshtein_distance.zig`](strings/damerau_levenshtein_distance.zig) | O(m × n) |
| 字母频率分析器 | [`strings/frequency_finder.zig`](strings/frequency_finder.zig) | O(n + 26 log 26) |
| 全字母句检查 | [`strings/is_pangram.zig`](strings/is_pangram.zig) | O(n) |
| 波兰 PESEL 校验器 | [`strings/is_polish_national_id.zig`](strings/is_polish_national_id.zig) | O(n) |
| 西班牙 DNI 校验器 | [`strings/is_spain_national_id.zig`](strings/is_spain_national_id.zig) | O(n) |
| Aho-Corasick 多模式匹配 | [`strings/aho_corasick.zig`](strings/aho_corasick.zig) | 查询 O(text + matches) |
| Trie 自动补全 | [`strings/autocomplete_using_trie.zig`](strings/autocomplete_using_trie.zig) | O(prefix + results) |
| 后缀数组 | [`strings/suffix_array.zig`](strings/suffix_array.zig) | 构建 O(n log² n) |
| 游程编码（RLE） | [`strings/run_length_encoding.zig`](strings/run_length_encoding.zig) | 编码/解码 O(n) |
| 条形码校验器 | [`strings/barcode_validator.zig`](strings/barcode_validator.zig) | O(d) |
| 驼峰转下划线 | [`strings/camel_case_to_snake_case.zig`](strings/camel_case_to_snake_case.zig) | O(n) |
| 回文重排可行性检查 | [`strings/can_string_be_rearranged_as_palindrome.zig`](strings/can_string_be_rearranged_as_palindrome.zig) | O(n) |
| 首字母大写 | [`strings/capitalize.zig`](strings/capitalize.zig) | O(n) |
| 统计元音数量 | [`strings/count_vowels.zig`](strings/count_vowels.zig) | O(n) |
| 信用卡校验器 | [`strings/credit_card_validator.zig`](strings/credit_card_validator.zig) | O(n) |
| 英文文本检测 | [`strings/detecting_english_programmatically.zig`](strings/detecting_english_programmatically.zig) | O(n + d) |
| DNA 互补链 | [`strings/dna.zig`](strings/dna.zig) | O(n) |
| 印度手机号校验器 | [`strings/indian_phone_validator.zig`](strings/indian_phone_validator.zig) | O(n) |
| 唯一字符检查 | [`strings/is_contains_unique_chars.zig`](strings/is_contains_unique_chars.zig) | O(n) |
| 同构词（Isogram）检查 | [`strings/is_isogram.zig`](strings/is_isogram.zig) | O(n) |
| 斯里兰卡手机号校验器 | [`strings/is_srilankan_phone_number.zig`](strings/is_srilankan_phone_number.zig) | O(n) |
| 电子邮件地址校验器 | [`strings/is_valid_email_address.zig`](strings/is_valid_email_address.zig) | O(n) |
| Jaro-Winkler 相似度 | [`strings/jaro_winkler.zig`](strings/jaro_winkler.zig) | O(n²) |
| 字符串连接 | [`strings/join.zig`](strings/join.zig) | O(total_len) |
| 转小写（ASCII） | [`strings/lower.zig`](strings/lower.zig) | O(n) |
| N-Gram 生成 | [`strings/ngram.zig`](strings/ngram.zig) | O(n · k) |
| 字符串分割 | [`strings/split.zig`](strings/split.zig) | O(n) |
| 字符串命名风格转换 | [`strings/string_switch_case.zig`](strings/string_switch_case.zig) | O(n) |
| 文本对齐 | [`strings/text_justification.zig`](strings/text_justification.zig) | O(m · n) |
| 转大写（ASCII） | [`strings/upper.zig`](strings/upper.zig) | O(n) |
| 交替合并字符串 | [`strings/alternative_string_arrange.zig`](strings/alternative_string_arrange.zig) | O(n + m) |
| Boyer-Moore 搜索 | [`strings/boyer_moore_search.zig`](strings/boyer_moore_search.zig) | 最坏 O(n·m) |
| Bitap 字符串匹配 | [`strings/bitap_string_match.zig`](strings/bitap_string_match.zig) | 核心 O(n) |
| 前缀函数 | [`strings/prefix_function.zig`](strings/prefix_function.zig) | O(n) |
| 去重单词 | [`strings/remove_duplicate.zig`](strings/remove_duplicate.zig) | O(k log k + n) |
| 反转长单词字母 | [`strings/reverse_letters.zig`](strings/reverse_letters.zig) | O(n) |
| 下划线转驼峰/帕斯卡 | [`strings/snake_case_to_camel_pascal_case.zig`](strings/snake_case_to_camel_pascal_case.zig) | O(n) |
| 裁剪首尾字符 | [`strings/strip.zig`](strings/strip.zig) | O(n) |
| 标题大小写转换 | [`strings/title.zig`](strings/title.zig) | O(n) |
| 单词出现次数统计 | [`strings/word_occurrence.zig`](strings/word_occurrence.zig) | O(n) |
| 猪拉丁文转换 | [`strings/pig_latin.zig`](strings/pig_latin.zig) | O(n) |
| 通配模式匹配（. 和 *） | [`strings/wildcard_pattern_matching.zig`](strings/wildcard_pattern_matching.zig) | O(n × m) |
| 波浪字符串 | [`strings/wave_string.zig`](strings/wave_string.zig) | O(n²) |
| Top K 高频词 | [`strings/top_k_frequent_words.zig`](strings/top_k_frequent_words.zig) | O(n + u log u) |
| Manacher 最长回文子串 | [`strings/manacher.zig`](strings/manacher.zig) | O(n) |
| 最小代价字符串转换 | [`strings/min_cost_string_conversion.zig`](strings/min_cost_string_conversion.zig) | O(m × n) |
| 单词模式编码 | [`strings/word_patterns.zig`](strings/word_patterns.zig) | O(n) |

## 快速开始

```bash
# 安装 Zig 0.15.2（无需 root 权限）
# 参见 https://ziglang.org/download/

# 运行所有测试
zig build test

# 运行单个算法的测试
zig test sorts/bubble_sort.zig
```

## 项目结构

```
TheAlgorithms-Zig/
├── build.zig                # 构建脚本 — 注册所有测试文件
├── build.zig.zon            # 包清单
├── sorts/                   # 50 种排序算法
├── searches/                # 16 种查找算法
├── maths/                   # 144 种数学算法
├── data_structures/         # 101 种数据结构实现
├── dynamic_programming/     # 54 个动态规划算法
├── graphs/                  # 46 个图算法
├── bit_manipulation/        # 27 个位运算算法
├── conversions/             # 27 个进制转换算法
├── boolean_algebra/         # 12 个布尔代数算法
├── divide_and_conquer/      # 11 个分治算法
├── linear_algebra/          # 11 个线性代数算法
├── physics/                 # 29 个物理算法
├── electronics/             # 19 个电子学算法
├── audio_filters/           # 2 个音频滤波算法
├── financial/               # 7 个金融算法
├── scheduling/              # 8 个调度算法
├── ciphers/                 # 47 个密码学算法
├── hashing/                 # 12 个哈希算法
├── data_compression/        # 8 个数据压缩算法
├── cellular_automata/       # 6 个元胞自动机算法
├── fractals/                # 5 个分形算法
├── project_euler/           # 54 个 Project Euler 算法
├── strings/                 # 59 个字符串算法
├── greedy_methods/          # 8 个贪心算法
├── matrix/                  # 20 个矩阵算法
├── geodesy/                 # 2 个测地学算法
├── geometry/                # 1 个几何算法
├── knapsack/                # 3 个背包算法
└── backtracking/            # 21 个回溯算法
```

## 开发指南

**环境要求：** Zig ≥ 0.15.2

每个算法文件自包含：实现 + 测试写在同一文件。添加新算法的步骤：

1. 创建 `<分类>/<算法名>.zig`
2. 用 `pub fn` 实现算法，适当使用 comptime 泛型
3. 在文件末尾添加 `test` 块
4. 在 `build.zig` 的 `test_files` 数组中注册该文件
5. 运行 `zig build test` 验证

## Vibe Coding 实验

本项目同时是一项 AI 辅助开发的研究实验。每个算法会记录：

- AI 需要几次尝试才能生成可编译的代码
- 出现了哪些报错类别（类型推断、内存分配、comptime 语法等）
- 需要人工手动修改多少行

实验结果将在 `EXPERIMENT_LOG.md` 中持续更新。

## 贡献指南

欢迎贡献！请确保：

- [ ] `zig build test` 通过
- [ ] 每个文件包含指向 Python 源代码的参考注释
- [ ] 文档注释中标注了时间/空间复杂度

## 许可证

MIT

### Ciphers (47)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Caesar Cipher | [`ciphers/caesar_cipher.zig`](ciphers/caesar_cipher.zig) | O(n · m) |
| ROT13 / Caesar Shift | [`ciphers/rot13.zig`](ciphers/rot13.zig) | O(n) |
| Atbash Cipher | [`ciphers/atbash.zig`](ciphers/atbash.zig) | O(n) |
| Vigenere Cipher | [`ciphers/vigenere_cipher.zig`](ciphers/vigenere_cipher.zig) | O(n) |
| Rail Fence Cipher | [`ciphers/rail_fence_cipher.zig`](ciphers/rail_fence_cipher.zig) | O(n) |
| XOR Cipher | [`ciphers/xor_cipher.zig`](ciphers/xor_cipher.zig) | O(n) |
| Base64 Cipher | [`ciphers/base64_cipher.zig`](ciphers/base64_cipher.zig) | O(n) |
| Transposition Cipher (Route) | [`ciphers/transposition_cipher.zig`](ciphers/transposition_cipher.zig) | O(n) |
| A1Z26 Letter-Number Cipher | [`ciphers/a1z26.zig`](ciphers/a1z26.zig) | O(n) |
| Affine Cipher | [`ciphers/affine_cipher.zig`](ciphers/affine_cipher.zig) | O(n · m) |
| Baconian Cipher | [`ciphers/baconian_cipher.zig`](ciphers/baconian_cipher.zig) | O(n) |
| Base16 Encoding/Decoding | [`ciphers/base16.zig`](ciphers/base16.zig) | O(n) |
| Base32 Encoding/Decoding | [`ciphers/base32.zig`](ciphers/base32.zig) | O(n) |
| Base85 Encoding/Decoding | [`ciphers/base85.zig`](ciphers/base85.zig) | O(n) |
| Morse Code Cipher | [`ciphers/morse_code.zig`](ciphers/morse_code.zig) | O(n · table) |
| Polybius Square Cipher | [`ciphers/polybius.zig`](ciphers/polybius.zig) | O(n) |
| Autokey Cipher | [`ciphers/autokey.zig`](ciphers/autokey.zig) | O(n) |
| Beaufort Cipher | [`ciphers/beaufort_cipher.zig`](ciphers/beaufort_cipher.zig) | O(n) |
| Gronsfeld Cipher | [`ciphers/gronsfeld_cipher.zig`](ciphers/gronsfeld_cipher.zig) | O(n) |
| Vernam Cipher | [`ciphers/vernam_cipher.zig`](ciphers/vernam_cipher.zig) | O(n) |
| Running Key Cipher | [`ciphers/running_key_cipher.zig`](ciphers/running_key_cipher.zig) | O(n) |
| Onepad Cipher | [`ciphers/onepad_cipher.zig`](ciphers/onepad_cipher.zig) | O(n) |
| Permutation Cipher | [`ciphers/permutation_cipher.zig`](ciphers/permutation_cipher.zig) | O(n) |
| Mono Alphabetic Cipher | [`ciphers/mono_alphabetic_ciphers.zig`](ciphers/mono_alphabetic_ciphers.zig) | O(n) |
| Brute Force Caesar Cipher | [`ciphers/brute_force_caesar_cipher.zig`](ciphers/brute_force_caesar_cipher.zig) | O(26 · n) |
| Cryptomath Modular Inverse | [`ciphers/cryptomath_module.zig`](ciphers/cryptomath_module.zig) | O(log m) |
| Diffie Primitive Root Search | [`ciphers/diffie.zig`](ciphers/diffie.zig) | O(m² log m) |
| Deterministic Miller-Rabin | [`ciphers/deterministic_miller_rabin.zig`](ciphers/deterministic_miller_rabin.zig) | O(k · log³ n) |
| RSA Factorization (d,e,n) | [`ciphers/rsa_factorization.zig`](ciphers/rsa_factorization.zig) | randomized |
| Porta Cipher | [`ciphers/porta_cipher.zig`](ciphers/porta_cipher.zig) | O(n) |
| Mixed Keyword Cipher | [`ciphers/mixed_keyword_cypher.zig`](ciphers/mixed_keyword_cypher.zig) | O(n) |
| Simple Keyword Cipher | [`ciphers/simple_keyword_cypher.zig`](ciphers/simple_keyword_cypher.zig) | O(n) |
| Simple Substitution Cipher | [`ciphers/simple_substitution_cipher.zig`](ciphers/simple_substitution_cipher.zig) | O(n) |
| Rabin-Miller Primality Test | [`ciphers/rabin_miller.zig`](ciphers/rabin_miller.zig) | O(k · log³ n) |
| RSA Key Generator | [`ciphers/rsa_key_generator.zig`](ciphers/rsa_key_generator.zig) | probabilistic |
| RSA Cipher | [`ciphers/rsa_cipher.zig`](ciphers/rsa_cipher.zig) | O(blocks · log exp) |
| ElGamal Key Generator | [`ciphers/elgamal_key_generator.zig`](ciphers/elgamal_key_generator.zig) | probabilistic |
| Transposition File Wrapper | [`ciphers/transposition_cipher_encrypt_decrypt_file.zig`](ciphers/transposition_cipher_encrypt_decrypt_file.zig) | O(n) |
| Bifid Cipher | [`ciphers/bifid.zig`](ciphers/bifid.zig) | O(n) |
| Playfair Cipher | [`ciphers/playfair_cipher.zig`](ciphers/playfair_cipher.zig) | O(n) |
| Caesar Chi-Squared Decryption | [`ciphers/decrypt_caesar_with_chi_squared.zig`](ciphers/decrypt_caesar_with_chi_squared.zig) | O(26 · n²) |
| Fractionated Morse Cipher | [`ciphers/fractionated_morse_cipher.zig`](ciphers/fractionated_morse_cipher.zig) | O(n) |
| Hill Cipher | [`ciphers/hill_cipher.zig`](ciphers/hill_cipher.zig) | O(n) |
| Shuffled Shift Cipher | [`ciphers/shuffled_shift_cipher.zig`](ciphers/shuffled_shift_cipher.zig) | O(n · m) |
| Trifid Cipher | [`ciphers/trifid_cipher.zig`](ciphers/trifid_cipher.zig) | O(n) |
| Enigma Machine 2 | [`ciphers/enigma_machine2.zig`](ciphers/enigma_machine2.zig) | O(n · 26) |
| Diffie-Hellman Key Exchange | [`ciphers/diffie_hellman.zig`](ciphers/diffie_hellman.zig) | O(log exp) per pow |

Note:
- `ciphers/diffie_hellman.zig` currently uses toy safe-prime groups (keeping group-id API shape) instead of RFC3526 huge primes, because this repository phase focuses on algorithm behavior validation under Zig `u128` without adding a big-integer dependency.

### 密码学 (47)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 凯撒密码 | [`ciphers/caesar_cipher.zig`](ciphers/caesar_cipher.zig) | O(n · m) |
| ROT13 / 凯撒移位 | [`ciphers/rot13.zig`](ciphers/rot13.zig) | O(n) |
| Atbash 密码 | [`ciphers/atbash.zig`](ciphers/atbash.zig) | O(n) |
| 维吉尼亚密码 | [`ciphers/vigenere_cipher.zig`](ciphers/vigenere_cipher.zig) | O(n) |
| 栅栏密码 | [`ciphers/rail_fence_cipher.zig`](ciphers/rail_fence_cipher.zig) | O(n) |
| XOR 密码 | [`ciphers/xor_cipher.zig`](ciphers/xor_cipher.zig) | O(n) |
| Base64 编解码 | [`ciphers/base64_cipher.zig`](ciphers/base64_cipher.zig) | O(n) |
| 列换位密码（Route） | [`ciphers/transposition_cipher.zig`](ciphers/transposition_cipher.zig) | O(n) |
| A1Z26 字母数字密码 | [`ciphers/a1z26.zig`](ciphers/a1z26.zig) | O(n) |
| 仿射密码 | [`ciphers/affine_cipher.zig`](ciphers/affine_cipher.zig) | O(n · m) |
| Baconian 密码 | [`ciphers/baconian_cipher.zig`](ciphers/baconian_cipher.zig) | O(n) |
| Base16 编解码 | [`ciphers/base16.zig`](ciphers/base16.zig) | O(n) |
| Base32 编解码 | [`ciphers/base32.zig`](ciphers/base32.zig) | O(n) |
| Base85 编解码 | [`ciphers/base85.zig`](ciphers/base85.zig) | O(n) |
| 摩尔斯密码 | [`ciphers/morse_code.zig`](ciphers/morse_code.zig) | O(n · table) |
| 波利比奥斯方阵密码 | [`ciphers/polybius.zig`](ciphers/polybius.zig) | O(n) |
| 自动密钥密码 | [`ciphers/autokey.zig`](ciphers/autokey.zig) | O(n) |
| Beaufort 密码 | [`ciphers/beaufort_cipher.zig`](ciphers/beaufort_cipher.zig) | O(n) |
| Gronsfeld 密码 | [`ciphers/gronsfeld_cipher.zig`](ciphers/gronsfeld_cipher.zig) | O(n) |
| Vernam 密码 | [`ciphers/vernam_cipher.zig`](ciphers/vernam_cipher.zig) | O(n) |
| Running Key 密码 | [`ciphers/running_key_cipher.zig`](ciphers/running_key_cipher.zig) | O(n) |
| Onepad 密码 | [`ciphers/onepad_cipher.zig`](ciphers/onepad_cipher.zig) | O(n) |
| 置换密码 | [`ciphers/permutation_cipher.zig`](ciphers/permutation_cipher.zig) | O(n) |
| 单表代换密码 | [`ciphers/mono_alphabetic_ciphers.zig`](ciphers/mono_alphabetic_ciphers.zig) | O(n) |
| 凯撒暴力解密 | [`ciphers/brute_force_caesar_cipher.zig`](ciphers/brute_force_caesar_cipher.zig) | O(26 · n) |
| Cryptomath 模逆计算 | [`ciphers/cryptomath_module.zig`](ciphers/cryptomath_module.zig) | O(log m) |
| Diffie 原根搜索 | [`ciphers/diffie.zig`](ciphers/diffie.zig) | O(m² log m) |
| 确定性 Miller-Rabin | [`ciphers/deterministic_miller_rabin.zig`](ciphers/deterministic_miller_rabin.zig) | O(k · log³ n) |
| RSA 因子分解（d,e,n） | [`ciphers/rsa_factorization.zig`](ciphers/rsa_factorization.zig) | 随机化 |
| Porta 密码 | [`ciphers/porta_cipher.zig`](ciphers/porta_cipher.zig) | O(n) |
| 混合关键词密码 | [`ciphers/mixed_keyword_cypher.zig`](ciphers/mixed_keyword_cypher.zig) | O(n) |
| 简单关键词密码 | [`ciphers/simple_keyword_cypher.zig`](ciphers/simple_keyword_cypher.zig) | O(n) |
| 简单替换密码 | [`ciphers/simple_substitution_cipher.zig`](ciphers/simple_substitution_cipher.zig) | O(n) |
| Rabin-Miller 素性测试 | [`ciphers/rabin_miller.zig`](ciphers/rabin_miller.zig) | O(k · log³ n) |
| RSA 密钥生成 | [`ciphers/rsa_key_generator.zig`](ciphers/rsa_key_generator.zig) | 概率型 |
| RSA 加解密 | [`ciphers/rsa_cipher.zig`](ciphers/rsa_cipher.zig) | O(块数 · log 指数) |
| ElGamal 密钥生成 | [`ciphers/elgamal_key_generator.zig`](ciphers/elgamal_key_generator.zig) | 概率型 |
| 置换密码文件封装 | [`ciphers/transposition_cipher_encrypt_decrypt_file.zig`](ciphers/transposition_cipher_encrypt_decrypt_file.zig) | O(n) |
| Bifid 密码 | [`ciphers/bifid.zig`](ciphers/bifid.zig) | O(n) |
| Playfair 密码 | [`ciphers/playfair_cipher.zig`](ciphers/playfair_cipher.zig) | O(n) |
| 凯撒卡方解密 | [`ciphers/decrypt_caesar_with_chi_squared.zig`](ciphers/decrypt_caesar_with_chi_squared.zig) | O(26 · n²) |
| 分式摩尔斯密码 | [`ciphers/fractionated_morse_cipher.zig`](ciphers/fractionated_morse_cipher.zig) | O(n) |
| Hill 密码 | [`ciphers/hill_cipher.zig`](ciphers/hill_cipher.zig) | O(n) |
| Shuffled Shift 密码 | [`ciphers/shuffled_shift_cipher.zig`](ciphers/shuffled_shift_cipher.zig) | O(n · m) |
| Trifid 密码 | [`ciphers/trifid_cipher.zig`](ciphers/trifid_cipher.zig) | O(n) |
| Enigma 机器 2 | [`ciphers/enigma_machine2.zig`](ciphers/enigma_machine2.zig) | O(n · 26) |
| Diffie-Hellman 密钥交换 | [`ciphers/diffie_hellman.zig`](ciphers/diffie_hellman.zig) | 每次幂运算 O(log exp) |

说明：
- `ciphers/diffie_hellman.zig` 当前采用 toy-safe 质数组（保留 group id 入口形态），未直接落 RFC3526 超大素数，原因是本阶段优先在不引入大整数依赖的前提下完成算法行为验证。

### Hashing (12)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| SHA-256 | [`hashing/sha256.zig`](hashing/sha256.zig) | O(n) |
| Adler-32 | [`hashing/adler32.zig`](hashing/adler32.zig) | O(n) |
| DJB2 | [`hashing/djb2.zig`](hashing/djb2.zig) | O(n) |
| ELF Hash | [`hashing/elf.zig`](hashing/elf.zig) | O(n) |
| Fletcher-16 | [`hashing/fletcher16.zig`](hashing/fletcher16.zig) | O(n) |
| Luhn Check | [`hashing/luhn.zig`](hashing/luhn.zig) | O(n) |
| SHA-1 | [`hashing/sha1.zig`](hashing/sha1.zig) | O(n) |
| MD5 | [`hashing/md5.zig`](hashing/md5.zig) | O(n) |
| Chaos Machine PRNG | [`hashing/chaos_machine.zig`](hashing/chaos_machine.zig) | O(1) per push/pull |
| SDBM Hash | [`hashing/sdbm.zig`](hashing/sdbm.zig) | O(n * k) |
| Enigma Machine (ASCII 32-125) | [`hashing/enigma_machine.zig`](hashing/enigma_machine.zig) | O(n * 94) |
| Hamming Code Utilities | [`hashing/hamming_code.zig`](hashing/hamming_code.zig) | O((n + p) * p) |

### 哈希 (12)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| SHA-256 | [`hashing/sha256.zig`](hashing/sha256.zig) | O(n) |
| Adler-32 | [`hashing/adler32.zig`](hashing/adler32.zig) | O(n) |
| DJB2 | [`hashing/djb2.zig`](hashing/djb2.zig) | O(n) |
| ELF 哈希 | [`hashing/elf.zig`](hashing/elf.zig) | O(n) |
| Fletcher-16 | [`hashing/fletcher16.zig`](hashing/fletcher16.zig) | O(n) |
| Luhn 校验 | [`hashing/luhn.zig`](hashing/luhn.zig) | O(n) |
| SHA-1 | [`hashing/sha1.zig`](hashing/sha1.zig) | O(n) |
| MD5 | [`hashing/md5.zig`](hashing/md5.zig) | O(n) |
| Chaos Machine 伪随机发生器 | [`hashing/chaos_machine.zig`](hashing/chaos_machine.zig) | 每次 push/pull 为 O(1) |
| SDBM 哈希 | [`hashing/sdbm.zig`](hashing/sdbm.zig) | O(n * k) |
| Enigma 机器（ASCII 32-125） | [`hashing/enigma_machine.zig`](hashing/enigma_machine.zig) | O(n * 94) |
| Hamming 码工具 | [`hashing/hamming_code.zig`](hashing/hamming_code.zig) | O((n + p) * p) |

### Data Compression (8)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Run-Length Encoding | [`data_compression/run_length_encoding.zig`](data_compression/run_length_encoding.zig) | O(n) |
| Burrows-Wheeler Transform | [`data_compression/burrows_wheeler.zig`](data_compression/burrows_wheeler.zig) | O(n² log n) |
| Coordinate Compression | [`data_compression/coordinate_compression.zig`](data_compression/coordinate_compression.zig) | O(n log n) build |
| LZ77 Compression | [`data_compression/lz77.zig`](data_compression/lz77.zig) | O(n * window_size) |
| Peak Signal-to-Noise Ratio | [`data_compression/peak_signal_to_noise_ratio.zig`](data_compression/peak_signal_to_noise_ratio.zig) | O(n) |
| Lempel-Ziv Bitstring Compression | [`data_compression/lempel_ziv.zig`](data_compression/lempel_ziv.zig) | O(n²) |
| Lempel-Ziv Bitstring Decompression | [`data_compression/lempel_ziv_decompress.zig`](data_compression/lempel_ziv_decompress.zig) | O(n²) |
| Huffman Coding | [`data_compression/huffman.zig`](data_compression/huffman.zig) | O(n + k²) |

### 数据压缩 (8)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 游程编码（RLE） | [`data_compression/run_length_encoding.zig`](data_compression/run_length_encoding.zig) | O(n) |
| Burrows-Wheeler 变换 | [`data_compression/burrows_wheeler.zig`](data_compression/burrows_wheeler.zig) | O(n² log n) |
| 坐标压缩 | [`data_compression/coordinate_compression.zig`](data_compression/coordinate_compression.zig) | 构建 O(n log n) |
| LZ77 压缩 | [`data_compression/lz77.zig`](data_compression/lz77.zig) | O(n * window_size) |
| 峰值信噪比（PSNR） | [`data_compression/peak_signal_to_noise_ratio.zig`](data_compression/peak_signal_to_noise_ratio.zig) | O(n) |
| Lempel-Ziv 位串压缩 | [`data_compression/lempel_ziv.zig`](data_compression/lempel_ziv.zig) | O(n²) |
| Lempel-Ziv 位串解压 | [`data_compression/lempel_ziv_decompress.zig`](data_compression/lempel_ziv_decompress.zig) | O(n²) |
| Huffman 编码 | [`data_compression/huffman.zig`](data_compression/huffman.zig) | O(n + k²) |
