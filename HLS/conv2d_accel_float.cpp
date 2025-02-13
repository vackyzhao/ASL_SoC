#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>

// ���붨��Ϊ16λ���������������Ϊ32λ������
typedef ap_fixed<32,16> ap_fixed32_t;   // 32-bit ������������ & Ȩ�� (16λ��8λ����)
typedef ap_fixed<64,32> ap_fixed64_t;  // 64-bit ����������� (32λ��16λ����)


// 7 MAC Processing Element (PE) - ʹ��16λ�����32λ����Ķ��������г˼�
void PE_float(
    ap_fixed32_t input_data,
    ap_fixed32_t weight_0,
    ap_fixed32_t weight_1,
    ap_fixed32_t weight_2,
    ap_fixed32_t weight_3,
    ap_fixed32_t weight_4,
    ap_fixed32_t weight_5,
    ap_fixed32_t weight_6,
    ap_fixed64_t psum_in,
    ap_fixed64_t &psum_out,
    ap_fixed32_t &output_data,
    bool reset
) {
    #pragma HLS PIPELINE II=1

    // 7 ��������λ�Ĵ���
    static ap_fixed32_t shift_reg_0 = 0;
    static ap_fixed32_t shift_reg_1 = 0;
    static ap_fixed32_t shift_reg_2 = 0;
    static ap_fixed32_t shift_reg_3 = 0;
    static ap_fixed32_t shift_reg_4 = 0;
    static ap_fixed32_t shift_reg_5 = 0;
    static ap_fixed32_t shift_reg_6 = 0;
    static ap_fixed32_t prev_input = 0;  // **�����ӳ� `input_data`**
    static ap_fixed64_t psum_reg = 0;    // **�����ۼ� `psum_out`**

    // ��λ�߼�
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
        // ����ִ����λ�Ĵ�������ȫ���� RAM ���ʣ�
        shift_reg_6 = shift_reg_5;
        shift_reg_5 = shift_reg_4;
        shift_reg_4 = shift_reg_3;
        shift_reg_3 = shift_reg_2;
        shift_reg_2 = shift_reg_1;
        shift_reg_1 = shift_reg_0;
        shift_reg_0 = input_data;

        // ���� 7 MAC �˼�
        ap_fixed64_t mac_result = 0;
        mac_result += shift_reg_0 * weight_0;
        mac_result += shift_reg_1 * weight_1;
        mac_result += shift_reg_2 * weight_2;
        mac_result += shift_reg_3 * weight_3;
        mac_result += shift_reg_4 * weight_4;
        mac_result += shift_reg_5 * weight_5;
        mac_result += shift_reg_6 * weight_6;

        // ���㲿�ֺ�
        psum_reg = mac_result + psum_in;

        std::cout << "  Shift registers: ["
                  << (int)shift_reg_0 << " "
                  << (int)shift_reg_1 << " "
                  << (int)shift_reg_2 << " "
                  << (int)shift_reg_3 << " "
                  << (int)shift_reg_4 << " "
                  << (int)shift_reg_5 << " "
                  << (int)shift_reg_6 << " "
                  << (int)psum_reg << "]\n";
    }

    // �������
    psum_out = psum_reg;
    output_data = prev_input;
    prev_input = input_data;
}

// ���� 7 �������������ݵ� PE_chain7
void PE_chain7_float(
    ap_fixed32_t input_data_reg[7],
    ap_fixed32_t weight_reg[7][7],  // ÿ�� PE ��Ȩ�ض��Ƕ�����
    ap_fixed64_t &psum_out3,
    ap_fixed64_t &psum_out5,
    ap_fixed64_t &psum_out7,
    ap_fixed32_t output_data[7],   // ������ݸ�Ϊ����
    bool reset
) {
    #pragma HLS ARRAY_PARTITION variable=input_data_reg complete
    #pragma HLS ARRAY_PARTITION variable=weight_reg complete
    #pragma HLS ARRAY_PARTITION variable=output_data complete

    ap_fixed64_t psum_in_temp[7] = {0},psum_mid_temp[7] = {0}, psum_out_temp[7] = {0};

    // PE������
    for (int i = 0; i < 7; i++) {
#pragma HLS UNROLL
        if (i == 0) {
            psum_in_temp[0] = 0;
        } else {
        	//psum_mid_temp[i]= psum_out_temp[i - 1];
            psum_in_temp[i] = psum_out_temp[i-1];
        }

        PE_float(input_data_reg[i],
           weight_reg[i][0], weight_reg[i][1], weight_reg[i][2],
           weight_reg[i][3], weight_reg[i][4], weight_reg[i][5],
           weight_reg[i][6],
           psum_in_temp[i],
           psum_out_temp[i],
           output_data[i],
           reset);
    }

    // ����������
    psum_out7 = psum_out_temp[6];  // ����˴�С7�����
    psum_out5 = psum_out_temp[4];  // ����˴�С7�����
    psum_out3 = psum_out_temp[2];  // ����˴�С7�����
}

void PE_chain7_28_float(
    ap_fixed32_t input_data_reg[34],
    ap_fixed32_t weight_reg[7][7],  // ÿ�� PE ��Ȩ�ض��Ƕ�����
    ap_fixed64_t psum_out3[28],
    ap_fixed64_t psum_out5[28],
    ap_fixed64_t psum_out7[28],
    bool reset
) {
    #pragma HLS ARRAY_PARTITION variable=input_data_reg complete
    #pragma HLS ARRAY_PARTITION variable=weight_reg complete
    #pragma HLS ARRAY_PARTITION variable=psum_out3 complete
    #pragma HLS ARRAY_PARTITION variable=psum_out5 complete
    #pragma HLS ARRAY_PARTITION variable=psum_out7 complete

    ap_fixed32_t output_data_temp[28][7];  // �洢ÿ�� PE ���������
    #pragma HLS ARRAY_PARTITION variable=output_data_temp complete

    static ap_fixed32_t delay_regs[7][6];  // [PE���][�ӳټ���]
    #pragma HLS ARRAY_PARTITION variable=delay_regs complete dim=0

    // Ϊ��һ��PE_chain�����ӳ�����
    for (int i = 0; i < 7; i++) {
        #pragma HLS UNROLL
        if (i == 0) {
            output_data_temp[0][i] = input_data_reg[i];  // ��ײ�PE���ӳ�
        } else {
            // ��λ����
            for (int j = 5; j > 0; j--) {
                delay_regs[i][j] = delay_regs[i][j - 1];
            }
            delay_regs[i][0] = input_data_reg[i];

            // ÿ��PE�������ӳٲ�ͬ��PE1�ӳ�1���ڣ�PE2�ӳ�2���ڣ��Դ�����
            output_data_temp[0][i] = delay_regs[i][i - 1];
        }
    }

    // ���д���28��ͨ��
    for (int ch = 0; ch < 28; ch++) {
        #pragma HLS UNROLL factor=28

        // ׼����������
        if (ch > 0) {
            for (int i = 0; i < 6; i++) {
                output_data_temp[ch][i] = output_data_temp[ch - 1][i + 1];
            }
            output_data_temp[ch][6] = input_data_reg[ch + 6];
        }

        // ����ǰͨ��
        PE_chain7_float(
            output_data_temp[ch],
            weight_reg,
            psum_out3[ch],
            psum_out5[ch],
            psum_out7[ch],
            output_data_temp[ch + 1],
            reset
        );
    }
}

void PE_chain7_28_top_float(
    ap_fixed32_t input_data[34],          // ��Ϊslave AXI
    ap_fixed32_t weight[7][7],            // ��Ϊslave AXI
    ap_fixed64_t output_data_3[28],       // ��Ϊslave AXI
    ap_fixed64_t output_data_5[28],       // ��Ϊslave AXI
    ap_fixed64_t output_data_7[28],       // ��Ϊslave AXI
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

    // ���ػ���
    ap_fixed32_t input_local[34];
    ap_fixed32_t weight_local[7][7];
    ap_fixed64_t output_3[28], output_5[28], output_7[28];

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

        // ��ȡ��������
        READ_INPUT: for(int i = 0; i < 34; i++) {
            #pragma HLS PIPELINE II=1
            input_local[i] = input_data[i];
        }

        // ��ȡȨ������
        READ_WEIGHT: for(int i = 0; i < 7; i++) {
            for(int j = 0; j < 7; j++) {
                #pragma HLS PIPELINE II=1
                weight_local[i][j] = weight[i][j];
            }
        }

        // ����
        PE_chain7_28_float(
            input_local,
            weight_local,
            output_3,
            output_5,
            output_7,
            reset
        );

        // д�ؽ��
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
