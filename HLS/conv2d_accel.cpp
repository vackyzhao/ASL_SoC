#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>

typedef ap_int<32> ap_int32_t ;     // 8-bit 输入数据 & 权重
typedef ap_int<64> ap_int64_t ;   // 32-bit 部分和 & 输出


// 7 MAC Processing Element (PE) - 直接使用 7 个独立的权重端口
void PE(
    ap_int32_t  input_data,
    ap_int32_t  weight_0,
    ap_int32_t  weight_1,
    ap_int32_t  weight_2,
    ap_int32_t  weight_3,
    ap_int32_t  weight_4,
    ap_int32_t  weight_5,
    ap_int32_t  weight_6,
    ap_int64_t  psum_in,
    ap_int64_t  &psum_out,
    ap_int32_t  &output_data,
    bool reset
) {
	#pragma HLS PIPELINE II=1
    //  7 级独立移位寄存器
    static ap_int32_t  shift_reg_0 = 0;
    static ap_int32_t  shift_reg_1 = 0;
    static ap_int32_t  shift_reg_2 = 0;
    static ap_int32_t  shift_reg_3 = 0;
    static ap_int32_t  shift_reg_4 = 0;
    static ap_int32_t  shift_reg_5 = 0;
    static ap_int32_t  shift_reg_6 = 0;
    static ap_int32_t  prev_input = 0;  // **用于延迟 `input_data`**
    static ap_int64_t  psum_reg = 0;   // **用于累加 `psum_out`**




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
        ap_int64_t  mac_result = 0;
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
    ap_int32_t  input_data_reg[7],
    ap_int32_t  weight_reg[7][7],  // 每个 PE 的权重都是独立的
    ap_int64_t  &psum_out3,
	ap_int64_t  &psum_out5,
	ap_int64_t  &psum_out7,
    ap_int32_t  output_data[7],   // 输出数据改为数组

    bool reset
) {
    //#pragma HLS DATAFLOW
    //#pragma HLS PIPELINE II=7
    // 定义每个 PE 独立的 psum_in 和 psum_out
    ap_int64_t  psum_in_temp[7]  = {0}, psum_out_temp[7] = {0},psum_mid_temp[7] = {0};


    // 使用 #pragma HLS ARRAY_PARTITION 展开数组，确保每个元素都作为寄存器
    #pragma HLS ARRAY_PARTITION variable=input_data_reg complete
    #pragma HLS ARRAY_PARTITION variable=weight_reg complete
    #pragma HLS ARRAY_PARTITION variable=output_data complete
	#pragma HLS ARRAY_PARTITION variable=psum_in_temp complete
	#pragma HLS ARRAY_PARTITION variable=psum_out_temp complete
	//#pragma HLS ARRAY_PARTITION variable=psum_mid_temp complete


    // PE链处理
        PE_chain7_label0:for (int i = 0; i < 7; i++) {

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
void PE_chain7_28(
    ap_int32_t  input_data_reg[34],
    ap_int32_t  weight_reg[7][7],  // 每个 PE 的权重都是独立的
    ap_int64_t  psum_out3[28],
    ap_int64_t  psum_out5[28],
    ap_int64_t  psum_out7[28],
    bool reset
)
{

    #pragma HLS ARRAY_PARTITION variable=input_data_reg complete

    #pragma HLS ARRAY_PARTITION variable=weight_reg complete
    #pragma HLS ARRAY_PARTITION variable=psum_out3 complete
    #pragma HLS ARRAY_PARTITION variable=psum_out5 complete
    #pragma HLS ARRAY_PARTITION variable=psum_out7 complete

    // 定义输出数据数组，用于存储 PE_chain7_1 和 PE_chain7_2 的结果
    ap_int32_t  output_data_temp[28][7];  // 存储每个 PE 输出的数据
	#pragma HLS ARRAY_PARTITION variable=output_data_temp complete

     // 第一个PE_chain的输入延迟寄存器
     static ap_int32_t  delay_regs[7][6];  // [PE编号][延迟级数]
     #pragma HLS ARRAY_PARTITION variable=delay_regs complete dim=0
    // 为第一个PE_chain处理延迟数据
    for(int i = 0; i < 7; i++) {
        #pragma HLS UNROLL
        if(i == 0) {
            output_data_temp[0][i] = input_data_reg[i];  // 最底层PE不延迟
        } else {
            // 移位操作
            for(int j = 5; j > 0; j--) {
                delay_regs[i][j] = delay_regs[i][j-1];
            }
            delay_regs[i][0] = input_data_reg[i];
            
            // 每个PE的输入延迟不同：PE1延迟1周期，PE2延迟2周期，以此类推
            output_data_temp[0][i] = delay_regs[i][i-1];
        }
    }

    // 并行处理28个通道
    for(int ch = 0; ch < 28; ch++) {
        #pragma HLS UNROLL factor=28
        
        // 准备输入数据
        if(ch > 0) {
            for(int i = 0; i < 6; i++) {
                output_data_temp[ch][i] = output_data_temp[ch-1][i+1];
            }
            output_data_temp[ch][6] = input_data_reg[ch+6];
        }

        // 处理当前通道
        PE_chain7(
            output_data_temp[ch],
            weight_reg,
            psum_out3[ch],
            psum_out5[ch],
            psum_out7[ch],
            output_data_temp[ch+1],  // 直接输出到下一个通道
            reset
        );
    }
}




void PE_chain7_28_top(
    ap_int32_t input_data[34],          // 改为slave AXI
    ap_int32_t weight[7][7],            // 改为slave AXI
    ap_int64_t output_data_3[28],       // 改为slave AXI
    ap_int64_t output_data_5[28],       // 改为slave AXI
    ap_int64_t output_data_7[28],       // 改为slave AXI
    bool start,                     
    bool reset,                     
    bool& done                      
) {
    #pragma HLS INTERFACE s_axilite port=input_data bundle=DATA_BUS
    #pragma HLS INTERFACE s_axilite port=weight bundle=DATA_BUS
    #pragma HLS INTERFACE s_axilite port=output_data_3 bundle=DATA_BUS
    #pragma HLS INTERFACE s_axilite port=output_data_5 bundle=DATA_BUS
    #pragma HLS INTERFACE s_axilite port=output_data_7 bundle=DATA_BUS
    #pragma HLS INTERFACE s_axilite port=start bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=reset bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=done bundle=CONTROL_BUS
    #pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS

    // 本地缓存
    ap_int32_t input_local[34];
    ap_int32_t weight_local[7][7];
    ap_int64_t output_3[28], output_5[28], output_7[28];
    
    #pragma HLS ARRAY_PARTITION variable=input_local complete
    #pragma HLS ARRAY_PARTITION variable=weight_local complete dim=0
    #pragma HLS ARRAY_PARTITION variable=output_3 complete
    #pragma HLS ARRAY_PARTITION variable=output_5 complete
    #pragma HLS ARRAY_PARTITION variable=output_7 complete

    static bool processing = false;
    done = false;

    if (reset) {
        processing = false;
        done = false;
        return;
    }

    if (start && !processing) {
        processing = true;

        // 读取输入数据
        READ_INPUT: for(int i = 0; i < 34; i++) {
            #pragma HLS PIPELINE II=1
            input_local[i] = input_data[i];
        }

        // 读取权重数据
        READ_WEIGHT: for(int i = 0; i < 7; i++) {
            for(int j = 0; j < 7; j++) {
                #pragma HLS PIPELINE II=1
                weight_local[i][j] = weight[i][j];
            }
        }

        // 计算
        PE_chain7_28(
            input_local,
            weight_local,
            output_3,
            output_5,
            output_7,
            reset
        );

        // 写回结果
        WRITE_OUTPUT: for(int i = 0; i < 28; i++) {
            #pragma HLS PIPELINE II=1
            output_data_3[i] = output_3[i];
            output_data_5[i] = output_5[i];
            output_data_7[i] = output_7[i];
        }

        processing = false;
        done = true;
    }
}

void conv2d_accel_top(
    ap_int32_t  input_feature[34][34],
    ap_int32_t  weight[7][7],
    ap_int64_t  output_feature[28][28],  // 单一输出特征图
    ap_uint<2> kernel_size,             // 0:3x3, 1:5x5, 2:7x7
    bool start,                         // 开始信号
    bool reset,
    bool &done                          // 完成信号
) {
    #pragma HLS INTERFACE m_axi port=input_feature offset=slave bundle=INPUT
    #pragma HLS INTERFACE m_axi port=weight offset=slave bundle=WEIGHT
    #pragma HLS INTERFACE m_axi port=output_feature offset=slave bundle=OUTPUT
    #pragma HLS INTERFACE s_axilite port=kernel_size bundle=CONTROL
    #pragma HLS INTERFACE s_axilite port=start bundle=CONTROL
    #pragma HLS INTERFACE s_axilite port=reset bundle=CONTROL
    #pragma HLS INTERFACE s_axilite port=done bundle=CONTROL
    #pragma HLS INTERFACE s_axilite port=return bundle=CONTROL

    // 状态标志
    static bool processing = false;
    done = false;

    if (reset) {
        processing = false;
        done = false;
        return;
    }

    if (start && !processing) {
        processing = true;

        // 本地缓存权重
        ap_int32_t  weight_local[7][7];
        #pragma HLS ARRAY_PARTITION variable=weight_local complete dim=0

        WEIGHT_CACHE: for(int i = 0; i < 7; i++) {
            for(int j = 0; j < 7; j++) {
                #pragma HLS PIPELINE II=1
                weight_local[i][j] = weight[i][j];
            }
        }

        // 滑动窗口处理
        ROW_PROCESS: for(int row = 0; row < 28; row++) {
            COL_PROCESS: for(int col = 0; col < 28; col++) {
                #pragma HLS PIPELINE II=1

                // 准备输入数据条带
                ap_int32_t  input_stripe[34];
                #pragma HLS ARRAY_PARTITION variable=input_stripe complete

                PREPARE_STRIPE: for(int i = 0; i < 34; i++) {
                    #pragma HLS UNROLL
                    input_stripe[i] = input_feature[row][col + i];
                }

                // 准备输出缓冲区
                ap_int64_t  psum_out3[28], psum_out5[28], psum_out7[28];
                #pragma HLS ARRAY_PARTITION variable=psum_out3 complete
                #pragma HLS ARRAY_PARTITION variable=psum_out5 complete
                #pragma HLS ARRAY_PARTITION variable=psum_out7 complete

                // 调用PE_chain7_28进行计算
                PE_chain7_28(
                    input_stripe,
                    weight_local,
                    psum_out3,
                    psum_out5,
                    psum_out7,
                    reset
                );

                // 根据kernel_size选择输出
                switch(kernel_size) {
                    case 0:  // 3x3
                        output_feature[row][col] = psum_out3[0];
                        break;
                    case 1:  // 5x5
                        output_feature[row][col] = psum_out5[0];
                        break;
                    case 2:  // 7x7
                        output_feature[row][col] = psum_out7[0];
                        break;
                    default:
                        output_feature[row][col] = 0;
                }
            }
        }

        // 完成处理
        processing = false;
        done = true;
    }
}
