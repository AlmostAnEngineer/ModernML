//
// Created by jacekp on 19.11.23.
//

#ifndef MODERNML_CONCEPTS_HPP
#define MODERNML_CONCEPTS_HPP

#include <concepts>

template<typename datatype>
concept FloatingPoint = std::is_floating_point_v<datatype>;

template<typename datatype>
concept Integral = std::integral<datatype>;

#endif //MODERNML_CONCEPTS_HPP
