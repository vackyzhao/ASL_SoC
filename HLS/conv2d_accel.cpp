#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>

typedef ap_int<8> ap_int8_t;     // 8-bit 输入数据 & 权重
typedef ap_int<32> ap_int32_t;   // 32-bit 部分和 & 输出

// 7 MAC Processing Element (PE) - 直接使用 7 个独立的权重端口
void PE(
    ap_int8_t input_data,
    ap_int8_t weight_0,
    ap_int8_t weight_1,
    ap_int8_t weight_2,
    ap_int8_t weight_3,
    ap_int8_t weight_4,
    ap_int8_t weight_5,
    ap_int8_t weight_6,
    ap_int32_t psum_in,
    ap_int32_t &psum_out,
    ap_int8_t &output_data,
    bool reset
) {
	#pragma HLS PIPELINE II=1
    // 1️⃣ 7 级独立移位寄存器
    static ap_int8_t shift_reg_0 = 0;
    static ap_int8_t shift_reg_1 = 0;
    static ap_int8_t shift_reg_2 = 0;
    static ap_int8_t shift_reg_3 = 0;
    static ap_int8_t shift_reg_4 = 0;
    static ap_int8_t shift_reg_5 = 0;
    static ap_int8_t shift_reg_6 = 0;
    static ap_int8_t prev_input = 0;  // **用于延迟 `input_data`**
    static ap_int32_t psum_reg = 0;   // **用于累加 `psum_out`**




    // 2️⃣ 复位逻辑
    if (reset) {
        shift_reg_0 = 0;
        shift_reg_1 = 0;
        shift_reg_2 = 0;
        shift_reg_3 = 0;
        shift_reg_4 = 0;
        shift_reg_5 = 0;
        shift_reg_6 = 0;
        prev_input = 0;
        psum_reg = 0;
        psum_out = 0;

    } else {
        // 3️⃣ 并行执行移位寄存器（完全避免 RAM 访问）
        shift_reg_6 = shift_reg_5;
        shift_reg_5 = shift_reg_4;
        shift_reg_4 = shift_reg_3;
        shift_reg_3 = shift_reg_2;
        shift_reg_2 = shift_reg_1;
        shift_reg_1 = shift_reg_0;
        shift_reg_0 = input_data;

        // 4️⃣ 计算 7 MAC 乘加
        ap_int32_t mac_result = 0;
        mac_result += shift_reg_0 * weight_0;
        mac_result += shift_reg_1 * weight_1;
        mac_result += shift_reg_2 * weight_2;
        mac_result += shift_reg_3 * weight_3;
        mac_result += shift_reg_4 * weight_4;
        mac_result += shift_reg_5 * weight_5;
        mac_result += shift_reg_6 * weight_6;

        // 5️⃣ 计算部分和
        psum_reg = mac_result + psum_in;

        std::cout << "  Shift registers: ["
                      << (int)shift_reg_0 << " "
                      << (int)shift_reg_1 << " "
                      << (int)shift_reg_2 << " "
                      << (int)shift_reg_3 << " "
                      << (int)shift_reg_4 << " "
                      << (int)shift_reg_5 << " "
                      << (int)shift_reg_6 << " "
					  << (int)psum_reg    << "]\n";

    }

    // 6️⃣ 输出数据
    psum_out = psum_reg;
    output_data = prev_input;
    prev_input = input_data;

}



void PE_chain7(
    ap_int8_t input_data_reg[7],
    ap_int8_t weight_reg[7][7],  // 每个 PE 的权重都是独立的
    ap_int32_t &psum_out3,
	ap_int32_t &psum_out5,
	ap_int32_t &psum_out7,
    ap_int8_t output_data[7],   // 输出数据改为数组

    bool reset
) {
    //#pragma HLS DATAFLOW
    //#pragma HLS PIPELINE II=7
    // 定义每个 PE 独立的 psum_in 和 psum_out
    ap_int32_t psum_in_temp[7]  = {0}, psum_out_temp[7] = {0},psum_mid_temp[7] = {0};


    // 使用 #pragma HLS ARRAY_PARTITION 展开数组，确保每个元素都作为寄存器
    #pragma HLS ARRAY_PARTITION variable=input_data_reg complete
    #pragma HLS ARRAY_PARTITION variable=weight_reg complete
    #pragma HLS ARRAY_PARTITION variable=output_data complete
	#pragma HLS ARRAY_PARTITION variable=psum_in_temp complete
	#pragma HLS ARRAY_PARTITION variable=psum_out_temp complete
	//#pragma HLS ARRAY_PARTITION variable=psum_mid_temp complete


    // PE链处理
        for (int i = 0; i < 7; i++) {
            #pragma HLS UNROLL
            if (i == 0) {
                psum_in_temp[0] = 0;
            } else {
                psum_in_temp[i] = psum_out_temp[i-1];
            }

            PE(input_data_reg[i],
               weight_reg[i][0], weight_reg[i][1], weight_reg[i][2],
               weight_reg[i][3], weight_reg[i][4], weight_reg[i][5],
               weight_reg[i][6],
               psum_in_temp[i],
               psum_out_temp[i],
               output_data[i],
               reset);
        }
    // 最终输出结果

    psum_out7 = psum_out_temp[6];  //卷积核大小7的输出
    psum_out5 = psum_out_temp[4];  //卷积核大小7的输出
    psum_out3 = psum_out_temp[2];  //卷积核大小7的输出
}


/*
void PE_chain5(
    ap_int8_t input_data_reg[7],
    ap_int8_t weight_reg[7][7],
    ap_int32_t &psum_out5,
    ap_int8_t output_data[7],
    bool reset
) {
    #pragma HLS PIPELINE II=40
    // 定义每个 PE 独立的 psum_in 和 psum_out
    ap_int32_t psum_in_temp[5] = {0}, psum_out_temp[5] = {0}, psum_mid_temp[5] = {0};

    // 使用 #pragma HLS ARRAY_PARTITION 展开数组
    #pragma HLS ARRAY_PARTITION variable=input_data_reg complete
    #pragma HLS ARRAY_PARTITION variable=weight_reg complete
    #pragma HLS ARRAY_PARTITION variable=output_data complete
    #pragma HLS ARRAY_PARTITION variable=psum_in_temp complete
    #pragma HLS ARRAY_PARTITION variable=psum_out_temp complete
    #pragma HLS ARRAY_PARTITION variable=psum_mid_temp complete

    // 只使用前5个PE
    for (int i = 0; i < 5; i++) {
        psum_in_temp[0] = 0;
        PE(input_data_reg[i],
           weight_reg[i][0], weight_reg[i][1], weight_reg[i][2], weight_reg[i][3],
           weight_reg[i][4], weight_reg[i][5], weight_reg[i][6],
           psum_in_temp[i],
           psum_out_temp[i],
           output_data[i],
           reset);

        if (i < 4) {
            psum_mid_temp[i] = psum_out_temp[i];
            psum_in_temp[i + 1] = psum_mid_temp[i];
        }
    }

    // 输出第5个PE的结果
    psum_out5 = psum_out_temp[4];
}
void PE_chain3(
    ap_int8_t input_data_reg[7],
    ap_int8_t weight_reg[7][7],
    ap_int32_t &psum_out3,
    ap_int8_t output_data[7],
    bool reset
) {
    #pragma HLS PIPELINE II=40
    ap_int32_t psum_in_temp[3] = {0}, psum_out_temp[3] = {0}, psum_mid_temp[3] = {0};

    std::cout << "\n==== PE Chain 3 Debug Info ====\n";
    std::cout << "Input data: [";
    for(int i = 0; i < 7; i++) {
        std::cout << (int)input_data_reg[i] << " ";
    }
    std::cout << "]\n";

    // 只使用前3个PE
    for (int i = 0; i < 3; i++) {
        psum_in_temp[0] = 0;
        
        std::cout << "\nPE[" << i << "] before execution:\n";
        std::cout << "  Input: " << (int)input_data_reg[i] << "\n";
        std::cout << "  Weights: [";
        for(int w = 0; w < 7; w++) {
            std::cout << (int)weight_reg[i][w] << " ";
        }
        std::cout << "]\n";
        std::cout << "  psum_in: " << psum_in_temp[i] << "\n";

        PE(input_data_reg[i],
           weight_reg[i][0], weight_reg[i][1], weight_reg[i][2], weight_reg[i][3],
           weight_reg[i][4], weight_reg[i][5], weight_reg[i][6],
           psum_in_temp[i],
           psum_out_temp[i],
           output_data[i],
           reset);

        std::cout << "PE[" << i << "] after execution:\n";
        std::cout << "  psum_out: " << psum_out_temp[i] << "\n";
        std::cout << "  output_data: " << (int)output_data[i] << "\n";

        if (i < 2) {
           // psum_mid_temp[i] = psum_out_temp[i];
            psum_in_temp[i + 1] = psum_out_temp[i];
            std::cout << "  Passing psum " << psum_mid_temp[i] << " to PE[" << (i+1) << "]\n";
        }
    }

    psum_out3 = psum_out_temp[2];
    std::cout << "\nFinal output: " << psum_out3 << "\n";
    std::cout << "================================\n";
}
*/
