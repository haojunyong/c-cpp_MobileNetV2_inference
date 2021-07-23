#include "opt.h"
void test_im2col(conv_op* op)
{
    //计算新矩阵的大小并分配内存空间



    //卷积
    int S = op->stride;//步长
    int P = op->padding;//填充大小
    int iw = op->in_w;
    int ih = op->in_h;
    int ow = op->out_w;
    int oh = op->out_h;
    int iwih = iw * ih;
    int iw1 = op->in_w + 2 * P;
    int ih1 = op->in_h + 2 * P;
    int iwih1 = iw1 * ih1;
    int ic = op->in_channels;
    int owoh = op->out_w * op->out_h;
    int o_owoh = op->out_channels * owoh;
    int k = op->kernel_size;
    int kk = op->kernel_size * op->kernel_size;
    int ikk = op->in_channels * kk;
    int i_iwih = op->in_channels * iwih;
    int i_iwih1 = op->in_channels * iwih1;
    float* input_p = (float*)calloc(i_iwih1, sizeof(float));

    //按行存储按行展开，最终要得到的输出
    //im2col
    int row = ikk;
    int col = owoh;
    float* data_matrix = (float*)malloc(row * col * sizeof(float));

    //先padding
    for (int j = 0; j < op->in_channels; j++)
    {
        for (int ii = 0; ii < op->in_h; ii++)
        {
            for (int jj = 0; jj < op->in_w; jj++)
            {
                //printf("%d\n", i * iwih + j * iwih + ii * ih + jj);
                input_p[j * iwih1 + (ii + P) * iw1 + jj + P] = op->input[j * iwih + ii * ih + jj];

            }
        }
    }

    ////img_to_col,input展开为matrix 按照卷积的顺序
    //for (int i_c = 0; i_c < op->in_channels; i_c++)
    //{
    //    for (int o_c = 0; o_c < op->out_channels; o_c++)
    //    {
    //        for (int o_h = 0; o_h < op->out_h; o_h++)
    //        {
    //            for (int o_w = 0; o_w < op->out_w; o_w++)
    //            {
    //                for (int i = 0; i < op->kernel_size; i++)
    //                {
    //                    for (int j = 0; j < op->kernel_size; j++)
    //                    {
    //                        //op->output[o_c * owoh + o_h * op->out_w + o_w] += op->weights[o_c * ikk + i_c * kk + i * k + j] * input_p[i_c * iwih1 + (S * o_h + i) * iw1 + S * o_w + j];
    //                        // data_matrix[(o_c * owoh + o_h * op->out_w + o_w) * kk + i * k + j] = input_p[i_c * iwih1 + (S * o_h + i) * iw1 + S * o_w + j];
    //                        data_matrix[(o_h * ow + o_w) * ikk + i_c * kk + i * k + j] = input_p[i_c * iwih1 + (S * o_h) * iw1 + S * o_w + i * iw1 + j];

    //                    }
    //                }
    //            }
    //        }
    //    }
    //}

    for (int w = 0; w < ikk; w++) //reshape后的矩阵行大小ikk
    {
        //滑动窗口内的偏移量，和当前通道偏移量
        int w_offset = w % k;
        int h_offset = (w / k) % k;
        int i_c = w / k / k;
        for (int o_h = 0; o_h < op->out_h; o_h++)
        {
            for (int o_w = 0; o_w < op->out_w; o_w++)
            {
                data_matrix[(o_h * ow + o_w) * ikk + w] = input_p[i_c * iwih1 + (S * o_h) * iw1 + S * o_w + h_offset * iw1 + w_offset];
                
            }
        }
    }

    //ofstream f0_input("f0_input.csv");
    //for (int c = 0; c < op->in_channels; c++)
    //{
    //    f0_input <<endl;
    //    for (int h= 0; h < ih1; h++)
    //    {
    //        for (int w = 0; w < iw1; w++)
    //        {
    //            f0_input << input_p[c * iwih1 + h * iw1 + w]<<",";
    //        }
    //        f0_input << endl;
    //    }

    //}
    //f0_input.close();
    //ofstream f0_col("f0_col.csv");
    //for (int i = 0; i < owoh; i++)
    //{
    //    for (int j = 0; j < ikk; j++)
    //    {
    //        f0_col << data_matrix[i * ikk + j] << "\t\t";
    //        
    //    }
    //    f0_col << endl;
    //}
    //f0_col.close();

    //卷积核展开
    //for (int o_c = 0; o_c < op->out_channels; o_c++)
    //{
    //    for (int i_c = 0; i_c < op->in_channels; i_c++)
    //    {
    //        for (int i = 0; i < op->kernel_size; i++)
    //        {
    //            for (int j = 0; j < op->kernel_size; j++)
    //            {
    //                kernel_matrix[o_c * ikk + i_c * kk + i * k + j] = op->weights[o_c * ikk + i_c * kk + i * k + j];
    //            }

    //        }
    //    }
    //}
    

    //类似矩阵乘法的卷积计算
    for (int o_c = 0; o_c < op->out_channels; o_c++)
    {
        for (int i = 0; i < owoh; i++)
        {
            for (int j = 0; j < ikk; j++)
            {
                op->output[o_c * owoh + i] += data_matrix[i * ikk + j] * op->weights[o_c * ikk + j];

            }
        }
    }

    free(input_p);
    free(data_matrix);




}
