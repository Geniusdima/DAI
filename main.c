#include "dai/aiheaders.h"

#define STEP 0.5

int main()
{
    schematic schm[] =
    {
        //input layer
        {.layer_count = 1, .neuron_count = 2},
        //hiden layers
        {.layer_count = 4, .neuron_count = 10},
        //output layer
        {.layer_count = 1, .neuron_count = 2},        
    };

    float arr1[] = {0, 0};
    float arr2[] = {0, 1};
    float arr3[] = {1, 1};
    float arr4[] = {1, 0};

    srand(time(NULL));
    network net;
    net = init_perceptron(schm, 3);

    int r = 0;
    
    for(int i = 0; i < 1000000; i++)
    {
        r = rand() % 4;
        switch (r)
        {
        case  0:
        {
            net.layers[0].neurons[0].value = 1;
            net.layers[0].neurons[1].value = 1;
            backprop_perceptron(&net, STEP, arr1, 2);
            break;
        }        
        case  1:
        {
            net.layers[0].neurons[0].value = 1;
            net.layers[0].neurons[1].value = 0;
            backprop_perceptron(&net, STEP, arr2, 2);
            break;
        }
        case  2:
        {
            net.layers[0].neurons[0].value = 0;
            net.layers[0].neurons[1].value = 0;
            backprop_perceptron(&net, STEP, arr3, 2);
            break;
        }
        case  3:
        {
            net.layers[0].neurons[0].value = 0;
            net.layers[0].neurons[1].value = 1;
            backprop_perceptron(&net, STEP, arr4, 2);
            break;
        }
        default:
            break;
        }
    }
    
    int v1, v2;
    printf("ok\n");
    
    for(int i = 0; i < 4; i++)
    {
        scanf("%d %d",&v1,&v2);

        net.layers[0].neurons[0].value = (float)v1;
        net.layers[0].neurons[1].value = (float)v2;


        forwardprop_perceptron(&net);
        printf("val1: %f\nval2: %f\n",net.layers[net.count-1].neurons[0].value, net.layers[net.count-1].neurons[1].value);        
    }
    save_network(&net,"save.bin");
    cleanup_network(&net);

    return 0;    
}