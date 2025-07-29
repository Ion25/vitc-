#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

#include "matrix.h"
#include <vector>
#include <cmath>

class AdamOptimizer {
private:
    float learning_rate;
    float beta1;     
    float beta2;      
    float epsilon;      
    int t;             
    
    std::vector<Matrix> m_weights;  
    std::vector<Matrix> v_weights; 
    std::vector<Matrix> m_biases;  
    std::vector<Matrix> v_biases;  
    
    bool initialized;
    
public:
    AdamOptimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f);
    
    void initialize(int num_weight_matrices, int num_bias_matrices);
    
    void updateWeights(Matrix& weights, const Matrix& gradients, int param_index);
    
    void updateBias(Matrix& bias, const Matrix& gradients, int param_index);
    
    void step();
    
    void reset();
    
    void setLearningRate(float lr) { learning_rate = lr; }
    float getLearningRate() const { return learning_rate; }
    int getTimeStep() const { return t; }
};

#endif // ADAM_OPTIMIZER_H