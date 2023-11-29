//
// Created by jacekp on 19.11.23.
//

#ifndef MODERNML_RANDOMGENERATOR_HPP
#define MODERNML_RANDOMGENERATOR_HPP
#include <random>
#include "Concepts.hpp"


template<FloatingPoint datatype>
datatype getRandomNumber(datatype min, datatype max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<datatype> dist(min, max);
    return dist(gen);
}

template<Integral datatype>
datatype getRandomNumber(datatype min, datatype max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<datatype> dist(min, max);
    return dist(gen);
}

#endif //MODERNML_RANDOMGENERATOR_HPP
