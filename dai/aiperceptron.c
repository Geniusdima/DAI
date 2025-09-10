#include "aiheaders.h"

extern network init_perceptron(schematic schm[], int schematic_count)
{
    network net;
    int layers_count_max = 0;
    net.count_schms = schematic_count;
    net.schms = (schematic*)malloc(sizeof(schematic)*schematic_count);

    
    for (int i = 0; i < schematic_count; i++)
    {
        layers_count_max += schm[i].layer_count;
        net.schms[i] = schm[i];
    }

    net.count = layers_count_max;
    
    net.layers = (layer*)malloc(sizeof(layer)*layers_count_max);

    int curr_layer = 0;

    for (int i = 0; i < schematic_count; i++)
    {
        curr_layer +=schm[i].layer_count;
        for(int j = curr_layer - schm[i].layer_count; j < curr_layer; j++)
        {
            net.layers[j].count= schm[i].neuron_count;
        }
    }


    int layers_count = 0;

    for (int i = 0; i < schematic_count; i++)
    {        
        layers_count += schm[i].layer_count;
        
        for (int j = layers_count - schm[i].layer_count; j < layers_count; j++)
        {
            net.layers[j].neurons = (neuron*) calloc(schm[i].neuron_count, sizeof(neuron));

            for (int k = 0; k < net.layers[j].count; k++)
            {

                if (j == layers_count_max -1)
                {
                    net.layers[j].neurons[k].weight = NULL;
                }
                else
                {
                    net.layers[j].neurons[k].weight = (float*) malloc(sizeof(float) * net.layers[j+1].count);

                    for(int l = 0; l < net.layers[j+1].count; l++)
                    {
                        net.layers[j].neurons[k].weight[l] = init_weight();
                    }
                }
            }            
        }
    }

    net.count = layers_count;
    return net;
}

extern void forwardprop_perceptron(network * net)
{
    if(net->count < 2)
    {
        printf("[ERROR] Low layers in network\n");
        return;
    }
    for (int i = 1; i < net->count; i++)
    {
        for(int j = 0; j < net->layers[i].count;j++)
        {
            float input = 0;
            for (int k= 0; k < net->layers[i-1].count; k++)
            {
                input += (net->layers[i-1].neurons[k].weight[j] * net->layers[i-1].neurons[k].value);
            }
            net->layers[i].neurons[j].value = sigmoid(input);
        }
    }
}

extern void backprop_perceptron(network * net, float step, float * correct, int correct_count)
{
    if(correct_count != net->layers[net->count-1].count)
    {
        printf("[ERROR] Uncorrect data in backpropagation\n");
        return;
    }

    forwardprop_perceptron(net);

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
