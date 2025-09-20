#include "../aiheaders.h"

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
                net->layers[i].neurons[j].gradient *=    (net->layers[i].neurons[j].value * (1.0f - net->layers[i].neurons[j].value));
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

extern void forwardprop(network * net)
{
    for (int i = 1; i < net->count; i++)
    {
        for(int j = 0; j < net->layers[i].count;j++)
        {
            float input = 0;
            for (int k = 0; k < net->layers[i-1].count; k++)
            {
                input += (net->layers[i-1].neurons[k].weight[j] * net->layers[i-1].neurons[k].value);
            }
            net->layers[i].neurons[j].value = sigmoid(input);
        }
    }
}