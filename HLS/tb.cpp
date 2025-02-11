#include <iostream>
#include <iomanip>
#include "ap_int.h"

typedef ap_int<8> ap_int8_t;
typedef ap_int<32> ap_int32_t;

// 声明外部函数
void PE_chain3(
    ap_int8_t input_data_reg[7],
    ap_int8_t weight_reg[7][7],
    ap_int32_t &psum_out3,
    ap_int8_t output_data[7],
    bool reset
);

void print_state(int cycle, ap_int32_t result, const ap_int8_t output_data[7]) {
    std::cout << "Cycle " << std::setw(2) << cycle << ": ";
    std::cout << "Conv result = " << std::setw(4) << result;
    std::cout << " | Output data: ";
    for(int i = 0; i < 7; i++) {
        std::cout << std::setw(2) << (int)output_data[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    // 初始化测试数据
    ap_int8_t input_data[7] = {2, 1, 1, 0, 0, 0, 0};  // 全1输入
    ap_int8_t weight_reg[7][7] = {0};  // 初始化为0
    ap_int8_t output_data[7] = {0};
    ap_int32_t psum_out3 = 0;
    
    // 设置3x3卷积核权重
    // 第一行：1,2,3
    weight_reg[0][0] = 1;
    weight_reg[0][1] = 2;
    weight_reg[0][2] = 3;
    // 第二行：4,5,6
    weight_reg[1][0] = 4;
    weight_reg[1][1] = 5;
    weight_reg[1][2] = 6;
    // 第三行：7,8,9
    weight_reg[2][0] = 7;
    weight_reg[2][1] = 8;
    weight_reg[2][2] = 9;
    
    std::cout << "Testing PE_chain3 with 3x3 kernel [1,2,3; 4,5,6; 7,8,9] and all-ones input\n";
    std::cout << "Expected result: (1+2+3+4+5+6+7+8+9) = 45\n\n";
    
    // 复位
    PE_chain3(input_data, weight_reg, psum_out3, output_data, true);
    print_state(0, psum_out3, output_data);
    
    // 运行10个周期
    for(int cycle = 1; cycle <= 10; cycle++) {

        PE_chain3(input_data, weight_reg, psum_out3, output_data, false);
        print_state(cycle, psum_out3, output_data);

    }
    
    return 0;
}
