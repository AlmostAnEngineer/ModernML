#pragma once
#ifndef LinearRegression_HPP
#define LinearRegression_HPP

#include <concepts>

template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename iterator, typename dataType>
class LinearRegression
{
  public:
    explicit LinearRegression(iterator begin, iterator end);
    ~LinearRegression();
};

#endif // !LinearRegression_HPP
