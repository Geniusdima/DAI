#include "aiheaders.h"

extern float init_weight()
{
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

extern float mse(float predicted, float correct)
{
    return predicted - correct;
}

extern float sigmoid(float x)
{
    return 1.0f/(1.0f+exp(-x));
}

extern float dsigmoid(float x)
{
    return sigmoid(x) * (1.0f - sigmoid(x));
}