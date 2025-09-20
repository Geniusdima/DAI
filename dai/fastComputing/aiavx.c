#include "../aiheaders.h"
#include <immintrin.h>

extern void backprop(network * net, float step, float * correct)
{
    for(int i = net->count - 1; i >= 0; i--)
    {
        for(int j = 0; j < net->layers[i].count; j++)
        {
            if(i == net->count - 1)
            {
                net->layers[i].neurons[j].gradient = (net->layers[i].neurons[j].value * (1.0f - net->layers[i].neurons[j].value)) * mse(net->layers[i].neurons[j].value, correct[j]);
            }
            else
            {
                net->layers[i].neurons[j].gradient = 0;

                for(int k = 0; k < net->layers[i+1].count; k++)
                {
                    net->layers[i].neurons[j].gradient += net->layers[i].neurons[j].weight[k] *  net->layers[i+1].neurons[k].gradient;
                }


                net->layers[i].neurons[j].gradient *= (net->layers[i].neurons[j].value * (1.0f - net->layers[i].neurons[j].value));
            }
        }
    }

    for(int i = 0; i < net->count-1; i++)
    {
        for (int j = 0; j < net->layers[i].count; j++)
        {
            for(int k = 0; k < net->layers[i+1].count; k++)
            {
                net->layers[i].neurons[j].weight[k] -= step * net->layers[i+1].neurons[k].gradient * net->layers[i].neurons[j].value;
            }
        }
    }
}




static float horizontal_sum_avx(__m256 v) 
{
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(vlow, vhigh);
    
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    
    return _mm_cvtss_f32(sum128);
}

extern void forwardprop(network * net)
{
    for (int i = 1; i < net->count; i++)
    {
        layer* current_layer = &net->layers[i];
        layer* prev_layer = &net->layers[i-1];
        const int prev_count = prev_layer->count;

        neuron* prev_neurons = prev_layer->neurons;
        neuron* curr_neurons = current_layer->neurons;

        for (int j = 0; j < current_layer->count; j++)
        {
            __m256 sum_vec0 = _mm256_setzero_ps();
            __m256 sum_vec1 = _mm256_setzero_ps();
            int k = 0;

            for (; k <= prev_count - 16; k += 16)
            {
                __m256 weights0 = _mm256_set_ps(
                    prev_neurons[k+7].weight[j],
                    prev_neurons[k+6].weight[j],
                    prev_neurons[k+5].weight[j],
                    prev_neurons[k+4].weight[j],
                    prev_neurons[k+3].weight[j],
                    prev_neurons[k+2].weight[j],
                    prev_neurons[k+1].weight[j],
                    prev_neurons[k+0].weight[j]
                );

                __m256 values0 = _mm256_set_ps(
                    prev_neurons[k+7].value,
                    prev_neurons[k+6].value,
                    prev_neurons[k+5].value,
                    prev_neurons[k+4].value,
                    prev_neurons[k+3].value,
                    prev_neurons[k+2].value,
                    prev_neurons[k+1].value,
                    prev_neurons[k+0].value
                );

                sum_vec0 = _mm256_fmadd_ps(weights0, values0, sum_vec0);

                __m256 weights1 = _mm256_set_ps(
                    prev_neurons[k+15].weight[j],
                    prev_neurons[k+14].weight[j],
                    prev_neurons[k+13].weight[j],
                    prev_neurons[k+12].weight[j],
                    prev_neurons[k+11].weight[j],
                    prev_neurons[k+10].weight[j],
                    prev_neurons[k+9].weight[j],
                    prev_neurons[k+8].weight[j]
                );

                __m256 values1 = _mm256_set_ps(
                    prev_neurons[k+15].value,
                    prev_neurons[k+14].value,
                    prev_neurons[k+13].value,
                    prev_neurons[k+12].value,
                    prev_neurons[k+11].value,
                    prev_neurons[k+10].value,
                    prev_neurons[k+9].value,
                    prev_neurons[k+8].value
                );

                sum_vec1 = _mm256_fmadd_ps(weights1, values1, sum_vec1);
            }

            __m256 total_sum = _mm256_add_ps(sum_vec0, sum_vec1);
            float input = horizontal_sum_avx(total_sum);

            for (; k < prev_count; k++)
            {
                input += prev_neurons[k].weight[j] * prev_neurons[k].value;
            }

            curr_neurons[j].value = sigmoid(input);
        }
    }
}