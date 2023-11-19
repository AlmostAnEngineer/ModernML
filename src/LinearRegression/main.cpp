#include "LinearRegression.hpp"
#include "utils/PrettyPrinter.hpp"
#include <chrono>

int main()
{
    ublas::matrix<double> X(50, 2);
    ublas::matrix<double> y(50, 1);

    for (size_t i = 0; i < X.size1(); ++i)
    {
        auto dbl = static_cast<double>(i);
        X(i, 0) = dbl * 1.0;
        X(i, 1) = dbl * 2.0;
        y(i, 0) = dbl + 5;
    }
    printMatrix(y);

    auto start_time = std::chrono::high_resolution_clock::now();
    LinearRegression<double> model(1e-5, 1000, 1e-8);
    auto output = model.fit(X, y);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Execution time: " << duration.count() << " millis" << std::endl;

    auto prediction = model.predict(X);
    printMatrix(prediction);
    if (prediction.size2() != 0)
    {
        auto scores = model.getR2Scores(y, prediction);
        std::cout << "R2Score: " << scores(0,0) << std::endl;
    }
    else
    {
        std::cout << "Error while training or bad shape";
    }

    return 0;
}
