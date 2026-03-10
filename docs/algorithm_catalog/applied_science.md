# Applied Science / 应用科学

- Source of truth: the detailed catalog sections from the pre-split root README.
- 数据来源：拆分前根 README 的详细目录条目。

### Boolean Algebra (12)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| AND Gate | [`boolean_algebra/and_gate.zig`](boolean_algebra/and_gate.zig) | O(1) / O(n) |
| OR Gate | [`boolean_algebra/or_gate.zig`](boolean_algebra/or_gate.zig) | O(1) |
| XOR Gate | [`boolean_algebra/xor_gate.zig`](boolean_algebra/xor_gate.zig) | O(1) |
| NAND Gate | [`boolean_algebra/nand_gate.zig`](boolean_algebra/nand_gate.zig) | O(1) |
| NOR Gate | [`boolean_algebra/nor_gate.zig`](boolean_algebra/nor_gate.zig) | O(1) |
| NOT Gate | [`boolean_algebra/not_gate.zig`](boolean_algebra/not_gate.zig) | O(1) |
| XNOR Gate | [`boolean_algebra/xnor_gate.zig`](boolean_algebra/xnor_gate.zig) | O(1) |
| IMPLY Gate | [`boolean_algebra/imply_gate.zig`](boolean_algebra/imply_gate.zig) | O(1) / O(n) |
| NIMPLY Gate | [`boolean_algebra/nimply_gate.zig`](boolean_algebra/nimply_gate.zig) | O(1) |
| 2-to-1 Multiplexer | [`boolean_algebra/multiplexer.zig`](boolean_algebra/multiplexer.zig) | O(1) |
| Karnaugh Map Simplification | [`boolean_algebra/karnaugh_map_simplification.zig`](boolean_algebra/karnaugh_map_simplification.zig) | O(r · c) |
| Quine-McCluskey Simplification | [`boolean_algebra/quine_mc_cluskey.zig`](boolean_algebra/quine_mc_cluskey.zig) | O(n² · m) reference-compatible |

### 布尔代数 (12)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| AND 门 | [`boolean_algebra/and_gate.zig`](boolean_algebra/and_gate.zig) | O(1) / O(n) |
| OR 门 | [`boolean_algebra/or_gate.zig`](boolean_algebra/or_gate.zig) | O(1) |
| XOR 门 | [`boolean_algebra/xor_gate.zig`](boolean_algebra/xor_gate.zig) | O(1) |
| NAND 门 | [`boolean_algebra/nand_gate.zig`](boolean_algebra/nand_gate.zig) | O(1) |
| NOR 门 | [`boolean_algebra/nor_gate.zig`](boolean_algebra/nor_gate.zig) | O(1) |
| NOT 门 | [`boolean_algebra/not_gate.zig`](boolean_algebra/not_gate.zig) | O(1) |
| XNOR 门 | [`boolean_algebra/xnor_gate.zig`](boolean_algebra/xnor_gate.zig) | O(1) |
| IMPLY 门 | [`boolean_algebra/imply_gate.zig`](boolean_algebra/imply_gate.zig) | O(1) / O(n) |
| NIMPLY 门 | [`boolean_algebra/nimply_gate.zig`](boolean_algebra/nimply_gate.zig) | O(1) |
| 2选1 多路复用器 | [`boolean_algebra/multiplexer.zig`](boolean_algebra/multiplexer.zig) | O(1) |
| 卡诺图化简 | [`boolean_algebra/karnaugh_map_simplification.zig`](boolean_algebra/karnaugh_map_simplification.zig) | O(r · c) |
| Quine-McCluskey 化简 | [`boolean_algebra/quine_mc_cluskey.zig`](boolean_algebra/quine_mc_cluskey.zig) | O(n² · m)（参考实现兼容） |

### Divide and Conquer (11)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Maximum Subarray (Divide and Conquer) | [`divide_and_conquer/max_subarray.zig`](divide_and_conquer/max_subarray.zig) | O(n log n) |
| Peak of Unimodal Array | [`divide_and_conquer/peak.zig`](divide_and_conquer/peak.zig) | O(log n) |
| Fast Power | [`divide_and_conquer/power.zig`](divide_and_conquer/power.zig) | O(log |b|) |
| Kth Order Statistic | [`divide_and_conquer/kth_order_statistic.zig`](divide_and_conquer/kth_order_statistic.zig) | O(n) average |
| Inversion Count | [`divide_and_conquer/inversions.zig`](divide_and_conquer/inversions.zig) | O(n log n) |
| Max Difference Pair | [`divide_and_conquer/max_difference_pair.zig`](divide_and_conquer/max_difference_pair.zig) | O(n log n) |
| Merge Sort (Divide and Conquer) | [`divide_and_conquer/mergesort.zig`](divide_and_conquer/mergesort.zig) | O(n log n) |
| Heap's Algorithm (Permutations) | [`divide_and_conquer/heaps_algorithm.zig`](divide_and_conquer/heaps_algorithm.zig) | O(n · n!) |
| Heap's Algorithm (Iterative) | [`divide_and_conquer/heaps_algorithm_iterative.zig`](divide_and_conquer/heaps_algorithm_iterative.zig) | O(n · n!) |
| Closest Pair of Points | [`divide_and_conquer/closest_pair_of_points.zig`](divide_and_conquer/closest_pair_of_points.zig) | O(n log n) |
| Convex Hull | [`divide_and_conquer/convex_hull.zig`](divide_and_conquer/convex_hull.zig) | O(n log n) |

### 分治 (11)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 最大子数组（分治） | [`divide_and_conquer/max_subarray.zig`](divide_and_conquer/max_subarray.zig) | O(n log n) |
| 单峰数组峰值 | [`divide_and_conquer/peak.zig`](divide_and_conquer/peak.zig) | O(log n) |
| 快速幂（分治） | [`divide_and_conquer/power.zig`](divide_and_conquer/power.zig) | O(log |b|) |
| 第 k 小元素（分治） | [`divide_and_conquer/kth_order_statistic.zig`](divide_and_conquer/kth_order_statistic.zig) | 平均 O(n) |
| 逆序对计数 | [`divide_and_conquer/inversions.zig`](divide_and_conquer/inversions.zig) | O(n log n) |
| 最大差值对（分治） | [`divide_and_conquer/max_difference_pair.zig`](divide_and_conquer/max_difference_pair.zig) | O(n log n) |
| 归并排序（分治） | [`divide_and_conquer/mergesort.zig`](divide_and_conquer/mergesort.zig) | O(n log n) |
| Heap 排列算法（分治） | [`divide_and_conquer/heaps_algorithm.zig`](divide_and_conquer/heaps_algorithm.zig) | O(n · n!) |
| Heap 排列算法（迭代） | [`divide_and_conquer/heaps_algorithm_iterative.zig`](divide_and_conquer/heaps_algorithm_iterative.zig) | O(n · n!) |
| 最近点对（分治） | [`divide_and_conquer/closest_pair_of_points.zig`](divide_and_conquer/closest_pair_of_points.zig) | O(n log n) |
| 凸包（分治） | [`divide_and_conquer/convex_hull.zig`](divide_and_conquer/convex_hull.zig) | O(n log n) |

### Linear Algebra (11)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Gaussian Elimination | [`linear_algebra/gaussian_elimination.zig`](linear_algebra/gaussian_elimination.zig) | O(n³) |
| LU Decomposition | [`linear_algebra/lu_decomposition.zig`](linear_algebra/lu_decomposition.zig) | O(n³) |
| Jacobi Iteration Method | [`linear_algebra/jacobi_iteration_method.zig`](linear_algebra/jacobi_iteration_method.zig) | O(iterations · n²) |
| Matrix Inversion | [`linear_algebra/matrix_inversion.zig`](linear_algebra/matrix_inversion.zig) | O(n³) |
| Rank of Matrix | [`linear_algebra/rank_of_matrix.zig`](linear_algebra/rank_of_matrix.zig) | O(min(r,c)·r·c) |
| Rayleigh Quotient | [`linear_algebra/rayleigh_quotient.zig`](linear_algebra/rayleigh_quotient.zig) | O(n²) |
| Power Iteration | [`linear_algebra/power_iteration.zig`](linear_algebra/power_iteration.zig) | O(iterations · n²) |
| Schur Complement | [`linear_algebra/schur_complement.zig`](linear_algebra/schur_complement.zig) | O(n³ + n²m + nm²) |
| 2D Transformations | [`linear_algebra/transformations_2d.zig`](linear_algebra/transformations_2d.zig) | O(1) |
| Gaussian Elimination (Pivoting) | [`linear_algebra/gaussian_elimination_pivoting.zig`](linear_algebra/gaussian_elimination_pivoting.zig) | O(n³) |
| Conjugate Gradient Method | [`linear_algebra/conjugate_gradient.zig`](linear_algebra/conjugate_gradient.zig) | O(iterations · n²) |

### 线性代数 (11)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 高斯消元 | [`linear_algebra/gaussian_elimination.zig`](linear_algebra/gaussian_elimination.zig) | O(n³) |
| LU 分解 | [`linear_algebra/lu_decomposition.zig`](linear_algebra/lu_decomposition.zig) | O(n³) |
| Jacobi 迭代法 | [`linear_algebra/jacobi_iteration_method.zig`](linear_algebra/jacobi_iteration_method.zig) | O(iterations · n²) |
| 矩阵求逆 | [`linear_algebra/matrix_inversion.zig`](linear_algebra/matrix_inversion.zig) | O(n³) |
| 矩阵秩 | [`linear_algebra/rank_of_matrix.zig`](linear_algebra/rank_of_matrix.zig) | O(min(r,c)·r·c) |
| 瑞利商 | [`linear_algebra/rayleigh_quotient.zig`](linear_algebra/rayleigh_quotient.zig) | O(n²) |
| 幂迭代法 | [`linear_algebra/power_iteration.zig`](linear_algebra/power_iteration.zig) | O(iterations · n²) |
| Schur 补 | [`linear_algebra/schur_complement.zig`](linear_algebra/schur_complement.zig) | O(n³ + n²m + nm²) |
| 二维变换矩阵 | [`linear_algebra/transformations_2d.zig`](linear_algebra/transformations_2d.zig) | O(1) |
| 高斯消元（部分主元） | [`linear_algebra/gaussian_elimination_pivoting.zig`](linear_algebra/gaussian_elimination_pivoting.zig) | O(n³) |
| 共轭梯度法 | [`linear_algebra/conjugate_gradient.zig`](linear_algebra/conjugate_gradient.zig) | O(iterations · n²) |

### Physics (29)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Kinetic Energy | [`physics/kinetic_energy.zig`](physics/kinetic_energy.zig) | O(1) |
| Potential Energy | [`physics/potential_energy.zig`](physics/potential_energy.zig) | O(1) |
| Newton's Second Law of Motion | [`physics/newtons_second_law_of_motion.zig`](physics/newtons_second_law_of_motion.zig) | O(1) |
| Escape Velocity | [`physics/escape_velocity.zig`](physics/escape_velocity.zig) | O(1) |
| Centripetal Force | [`physics/centripetal_force.zig`](physics/centripetal_force.zig) | O(1) |
| Newton's Law of Gravitation | [`physics/newtons_law_of_gravitation.zig`](physics/newtons_law_of_gravitation.zig) | O(1) |
| Period of Pendulum | [`physics/period_of_pendulum.zig`](physics/period_of_pendulum.zig) | O(1) |
| Speed of Sound in a Fluid | [`physics/speed_of_sound.zig`](physics/speed_of_sound.zig) | O(1) |
| Mass-Energy Equivalence | [`physics/mass_energy_equivalence.zig`](physics/mass_energy_equivalence.zig) | O(1) |
| Ideal Gas Law Utilities | [`physics/ideal_gas_law.zig`](physics/ideal_gas_law.zig) | O(1) |
| Terminal Velocity | [`physics/terminal_velocity.zig`](physics/terminal_velocity.zig) | O(1) |
| RMS Speed of Molecule | [`physics/rms_speed_of_molecule.zig`](physics/rms_speed_of_molecule.zig) | O(1) |
| Reynolds Number | [`physics/reynolds_number.zig`](physics/reynolds_number.zig) | O(1) |
| Shear Stress | [`physics/shear_stress.zig`](physics/shear_stress.zig) | O(1) |
| Archimedes Principle of Buoyant Force | [`physics/archimedes_principle_of_buoyant_force.zig`](physics/archimedes_principle_of_buoyant_force.zig) | O(1) |
| Doppler Frequency Shift | [`physics/doppler_frequency.zig`](physics/doppler_frequency.zig) | O(1) |
| Hubble Parameter | [`physics/hubble_parameter.zig`](physics/hubble_parameter.zig) | O(1) |
| Malus Law | [`physics/malus_law.zig`](physics/malus_law.zig) | O(1) |
| Photoelectric Effect | [`physics/photoelectric_effect.zig`](physics/photoelectric_effect.zig) | O(1) |
| Lens Formulae | [`physics/lens_formulae.zig`](physics/lens_formulae.zig) | O(1) |
| Altitude from Pressure | [`physics/altitude_pressure.zig`](physics/altitude_pressure.zig) | O(1) |
| Basic Orbital Capture | [`physics/basic_orbital_capture.zig`](physics/basic_orbital_capture.zig) | O(1) |
| Casimir Effect Solver | [`physics/casimir_effect.zig`](physics/casimir_effect.zig) | O(1) |
| Center of Mass (3D) | [`physics/center_of_mass.zig`](physics/center_of_mass.zig) | O(n) |
| Horizontal Projectile Motion | [`physics/horizontal_projectile_motion.zig`](physics/horizontal_projectile_motion.zig) | O(1) |
| Mirror Formulae | [`physics/mirror_formulae.zig`](physics/mirror_formulae.zig) | O(1) |
| Orbital Transfer Work | [`physics/orbital_transfer_work.zig`](physics/orbital_transfer_work.zig) | O(1) |
| Speeds of Gas Molecules | [`physics/speeds_of_gas_molecules.zig`](physics/speeds_of_gas_molecules.zig) | O(1) |
| Static Equilibrium Check | [`physics/in_static_equilibrium.zig`](physics/in_static_equilibrium.zig) | O(n) |

### 物理 (29)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 动能 | [`physics/kinetic_energy.zig`](physics/kinetic_energy.zig) | O(1) |
| 势能 | [`physics/potential_energy.zig`](physics/potential_energy.zig) | O(1) |
| 牛顿第二定律 | [`physics/newtons_second_law_of_motion.zig`](physics/newtons_second_law_of_motion.zig) | O(1) |
| 逃逸速度 | [`physics/escape_velocity.zig`](physics/escape_velocity.zig) | O(1) |
| 向心力 | [`physics/centripetal_force.zig`](physics/centripetal_force.zig) | O(1) |
| 万有引力定律 | [`physics/newtons_law_of_gravitation.zig`](physics/newtons_law_of_gravitation.zig) | O(1) |
| 单摆周期 | [`physics/period_of_pendulum.zig`](physics/period_of_pendulum.zig) | O(1) |
| 流体中的声速 | [`physics/speed_of_sound.zig`](physics/speed_of_sound.zig) | O(1) |
| 质能等价 | [`physics/mass_energy_equivalence.zig`](physics/mass_energy_equivalence.zig) | O(1) |
| 理想气体方程工具 | [`physics/ideal_gas_law.zig`](physics/ideal_gas_law.zig) | O(1) |
| 终端速度 | [`physics/terminal_velocity.zig`](physics/terminal_velocity.zig) | O(1) |
| 分子均方根速率 | [`physics/rms_speed_of_molecule.zig`](physics/rms_speed_of_molecule.zig) | O(1) |
| 雷诺数 | [`physics/reynolds_number.zig`](physics/reynolds_number.zig) | O(1) |
| 剪应力 | [`physics/shear_stress.zig`](physics/shear_stress.zig) | O(1) |
| 阿基米德浮力原理 | [`physics/archimedes_principle_of_buoyant_force.zig`](physics/archimedes_principle_of_buoyant_force.zig) | O(1) |
| 多普勒频移 | [`physics/doppler_frequency.zig`](physics/doppler_frequency.zig) | O(1) |
| 哈勃参数 | [`physics/hubble_parameter.zig`](physics/hubble_parameter.zig) | O(1) |
| 马吕斯定律 | [`physics/malus_law.zig`](physics/malus_law.zig) | O(1) |
| 光电效应 | [`physics/photoelectric_effect.zig`](physics/photoelectric_effect.zig) | O(1) |
| 透镜公式 | [`physics/lens_formulae.zig`](physics/lens_formulae.zig) | O(1) |
| 气压估算高度 | [`physics/altitude_pressure.zig`](physics/altitude_pressure.zig) | O(1) |
| 基础轨道捕获 | [`physics/basic_orbital_capture.zig`](physics/basic_orbital_capture.zig) | O(1) |
| 卡西米尔效应求解 | [`physics/casimir_effect.zig`](physics/casimir_effect.zig) | O(1) |
| 质心计算（3D） | [`physics/center_of_mass.zig`](physics/center_of_mass.zig) | O(n) |
| 水平抛体运动 | [`physics/horizontal_projectile_motion.zig`](physics/horizontal_projectile_motion.zig) | O(1) |
| 球面镜公式 | [`physics/mirror_formulae.zig`](physics/mirror_formulae.zig) | O(1) |
| 轨道转移做功 | [`physics/orbital_transfer_work.zig`](physics/orbital_transfer_work.zig) | O(1) |
| 气体分子速度 | [`physics/speeds_of_gas_molecules.zig`](physics/speeds_of_gas_molecules.zig) | O(1) |
| 静力平衡判定 | [`physics/in_static_equilibrium.zig`](physics/in_static_equilibrium.zig) | O(n) |

### Electronics (19)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Ohm's Law | [`electronics/ohms_law.zig`](electronics/ohms_law.zig) | O(1) |
| Electric Power | [`electronics/electric_power.zig`](electronics/electric_power.zig) | O(1) |
| Resistor Equivalence (Series/Parallel) | [`electronics/resistor_equivalence.zig`](electronics/resistor_equivalence.zig) | O(n) |
| Capacitor Equivalence (Parallel/Series) | [`electronics/capacitor_equivalence.zig`](electronics/capacitor_equivalence.zig) | O(n) |
| Electrical Impedance | [`electronics/electrical_impedance.zig`](electronics/electrical_impedance.zig) | O(1) |
| Inductive Reactance | [`electronics/ind_reactance.zig`](electronics/ind_reactance.zig) | O(1) |
| Resonant Frequency (LC Circuit) | [`electronics/resonant_frequency.zig`](electronics/resonant_frequency.zig) | O(1) |
| Electric Conductivity | [`electronics/electric_conductivity.zig`](electronics/electric_conductivity.zig) | O(1) |
| Charging Capacitor (RC) | [`electronics/charging_capacitor.zig`](electronics/charging_capacitor.zig) | O(1) |
| Charging Inductor (RL) | [`electronics/charging_inductor.zig`](electronics/charging_inductor.zig) | O(1) |
| Apparent Power (AC Phasors) | [`electronics/apparent_power.zig`](electronics/apparent_power.zig) | O(1) |
| Real and Reactive Power | [`electronics/real_and_reactive_power.zig`](electronics/real_and_reactive_power.zig) | O(1) |
| Wheatstone Bridge Solver | [`electronics/wheatstone_bridge.zig`](electronics/wheatstone_bridge.zig) | O(1) |
| Builtin Voltage (PN Junction) | [`electronics/builtin_voltage.zig`](electronics/builtin_voltage.zig) | O(1) |
| Carrier Concentration Solver | [`electronics/carrier_concentration.zig`](electronics/carrier_concentration.zig) | O(1) |
| Circular Convolution | [`electronics/circular_convolution.zig`](electronics/circular_convolution.zig) | O(n²) |
| Coulomb's Law Solver | [`electronics/coulombs_law.zig`](electronics/coulombs_law.zig) | O(1) |
| IC 555 Timer (Astable) | [`electronics/ic_555_timer.zig`](electronics/ic_555_timer.zig) | O(1) |
| Resistor Color Code Calculator | [`electronics/resistor_color_code.zig`](electronics/resistor_color_code.zig) | O(bands) |

### 电子学 (19)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 欧姆定律 | [`electronics/ohms_law.zig`](electronics/ohms_law.zig) | O(1) |
| 电功率计算 | [`electronics/electric_power.zig`](electronics/electric_power.zig) | O(1) |
| 等效电阻（串联/并联） | [`electronics/resistor_equivalence.zig`](electronics/resistor_equivalence.zig) | O(n) |
| 等效电容（并联/串联） | [`electronics/capacitor_equivalence.zig`](electronics/capacitor_equivalence.zig) | O(n) |
| 电阻抗计算 | [`electronics/electrical_impedance.zig`](electronics/electrical_impedance.zig) | O(1) |
| 感抗计算 | [`electronics/ind_reactance.zig`](electronics/ind_reactance.zig) | O(1) |
| 谐振频率（LC 电路） | [`electronics/resonant_frequency.zig`](electronics/resonant_frequency.zig) | O(1) |
| 电导率计算 | [`electronics/electric_conductivity.zig`](electronics/electric_conductivity.zig) | O(1) |
| 电容充电（RC） | [`electronics/charging_capacitor.zig`](electronics/charging_capacitor.zig) | O(1) |
| 电感充电（RL） | [`electronics/charging_inductor.zig`](electronics/charging_inductor.zig) | O(1) |
| 视在功率（交流相量） | [`electronics/apparent_power.zig`](electronics/apparent_power.zig) | O(1) |
| 有功与无功功率 | [`electronics/real_and_reactive_power.zig`](electronics/real_and_reactive_power.zig) | O(1) |
| 惠斯通电桥求解 | [`electronics/wheatstone_bridge.zig`](electronics/wheatstone_bridge.zig) | O(1) |
| 内建电势（PN 结） | [`electronics/builtin_voltage.zig`](electronics/builtin_voltage.zig) | O(1) |
| 载流子浓度求解 | [`electronics/carrier_concentration.zig`](electronics/carrier_concentration.zig) | O(1) |
| 循环卷积 | [`electronics/circular_convolution.zig`](electronics/circular_convolution.zig) | O(n²) |
| 库仑定律求解 | [`electronics/coulombs_law.zig`](electronics/coulombs_law.zig) | O(1) |
| 555 定时器（无稳态） | [`electronics/ic_555_timer.zig`](electronics/ic_555_timer.zig) | O(1) |
| 电阻色环编码计算 | [`electronics/resistor_color_code.zig`](electronics/resistor_color_code.zig) | O(bands) |

### Audio Filters (2)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| IIR Filter | [`audio_filters/iir_filter.zig`](audio_filters/iir_filter.zig) | O(order) per sample |
| Butterworth Filter Design | [`audio_filters/butterworth_filter.zig`](audio_filters/butterworth_filter.zig) | O(1) |

### 音频滤波 (2)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| IIR 滤波器 | [`audio_filters/iir_filter.zig`](audio_filters/iir_filter.zig) | 每个采样点 O(order) |
| Butterworth 滤波器设计 | [`audio_filters/butterworth_filter.zig`](audio_filters/butterworth_filter.zig) | O(1) |

### Financial (7)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Interest Calculations | [`financial/interest.zig`](financial/interest.zig) | O(1) |
| Present Value | [`financial/present_value.zig`](financial/present_value.zig) | O(n) |
| Price Plus Tax | [`financial/price_plus_tax.zig`](financial/price_plus_tax.zig) | O(1) |
| Simple Moving Average | [`financial/simple_moving_average.zig`](financial/simple_moving_average.zig) | O(n · window_size) |
| Equated Monthly Installments | [`financial/equated_monthly_installments.zig`](financial/equated_monthly_installments.zig) | O(1) |
| Straight-Line Depreciation | [`financial/straight_line_depreciation.zig`](financial/straight_line_depreciation.zig) | O(years) |
| Time and Half Pay | [`financial/time_and_half_pay.zig`](financial/time_and_half_pay.zig) | O(1) |

### 金融 (7)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 利息计算 | [`financial/interest.zig`](financial/interest.zig) | O(1) |
| 现值计算 | [`financial/present_value.zig`](financial/present_value.zig) | O(n) |
| 含税价格计算 | [`financial/price_plus_tax.zig`](financial/price_plus_tax.zig) | O(1) |
| 简单移动平均（SMA） | [`financial/simple_moving_average.zig`](financial/simple_moving_average.zig) | O(n · window_size) |
| 等额月供（EMI） | [`financial/equated_monthly_installments.zig`](financial/equated_monthly_installments.zig) | O(1) |
| 直线折旧法 | [`financial/straight_line_depreciation.zig`](financial/straight_line_depreciation.zig) | O(years) |
| 一倍半工资计算 | [`financial/time_and_half_pay.zig`](financial/time_and_half_pay.zig) | O(1) |

### Scheduling (8)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Round Robin Scheduling | [`scheduling/round_robin.zig`](scheduling/round_robin.zig) | O(total_quanta · n) |
| Shortest Job First (Preemptive) | [`scheduling/shortest_job_first.zig`](scheduling/shortest_job_first.zig) | O(T · n) |
| First Come First Served | [`scheduling/first_come_first_served.zig`](scheduling/first_come_first_served.zig) | O(n) |
| Highest Response Ratio Next | [`scheduling/highest_response_ratio_next.zig`](scheduling/highest_response_ratio_next.zig) | O(n²) |
| Non-Preemptive Shortest Job First | [`scheduling/non_preemptive_shortest_job_first.zig`](scheduling/non_preemptive_shortest_job_first.zig) | O(n²) |
| Job Sequence With Deadline | [`scheduling/job_sequence_with_deadline.zig`](scheduling/job_sequence_with_deadline.zig) | O(n²) |
| Job Sequencing With Deadlines (Profit Slots) | [`scheduling/job_sequencing_with_deadline.zig`](scheduling/job_sequencing_with_deadline.zig) | O(n log n + n·d) |
| Multi Level Feedback Queue | [`scheduling/multi_level_feedback_queue.zig`](scheduling/multi_level_feedback_queue.zig) | O(q · n²) educational |

### 调度 (8)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 轮转调度 | [`scheduling/round_robin.zig`](scheduling/round_robin.zig) | O(total_quanta · n) |
| 最短作业优先（可抢占） | [`scheduling/shortest_job_first.zig`](scheduling/shortest_job_first.zig) | O(T · n) |
| 先来先服务调度 | [`scheduling/first_come_first_served.zig`](scheduling/first_come_first_served.zig) | O(n) |
| 最高响应比优先调度 | [`scheduling/highest_response_ratio_next.zig`](scheduling/highest_response_ratio_next.zig) | O(n²) |
| 非抢占式最短作业优先 | [`scheduling/non_preemptive_shortest_job_first.zig`](scheduling/non_preemptive_shortest_job_first.zig) | O(n²) |
| 截止期任务排序 | [`scheduling/job_sequence_with_deadline.zig`](scheduling/job_sequence_with_deadline.zig) | O(n²) |
| 截止期作业排序（利润槽位） | [`scheduling/job_sequencing_with_deadline.zig`](scheduling/job_sequencing_with_deadline.zig) | O(n log n + n·d) |
| 多级反馈队列 | [`scheduling/multi_level_feedback_queue.zig`](scheduling/multi_level_feedback_queue.zig) | O(q · n²)（教学实现） |
