#include <iostream>
#include <fstream>
#include <numbers>

#include <nonlinfunc.hpp>
#include <timestepper.hpp>
#include <autodiff.hpp>

using namespace ASC_ode;

class PendulumAD : public NonlinearFunction
{
private:
  double m_length;
  double m_gravity;

public:
  PendulumAD(double length, double gravity=9.81) : m_length(length), m_gravity(gravity) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
  
  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    T_evaluate<double>(x, f);
  }

  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    Vector<AutoDiff<2>> x_ad(2);
    Vector<AutoDiff<2>> f_ad(2);

    x_ad(0) = AutoDiff<2>(Variable<0, double>(x(0)));
    x_ad(1) = AutoDiff<2>(Variable<1, double>(x(1)));
    T_evaluate<AutoDiff<2>>(x_ad, f_ad);

    for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 2; j++)
         df(i,j) = f_ad(i).deriv()[j];
  }

  template <typename T>
  void T_evaluate (VectorView<T> x, VectorView<T> f) const

  {
    f(0) = x(1);
    f(1) = -m_gravity/m_length*sin(x(0));
  }
};

int main() {
    // Example usage of the PendulumAD class
    double length = 1.0; // Length of the pendulum
    double gravity = 9.81; // Acceleration due to gravity

    double tend = 10.0; // Total simulation time
    int steps = 100; // Number of time steps
    double tau = tend / steps; // Time step size

    Vector<> y = { 0.1, 0.0 }; // Initial conditions: small angle and zero angular velocity
    auto rhs = std::make_shared<PendulumAD>(length, gravity);

    ExplicitEuler stepper(rhs);

    std::ofstream outfile("output_test_pendulum.txt");

    std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
    outfile << 0.0 << "  " << y(0) << " "<< y(1) << std::endl;

    for (int i = 0; i < steps; i++)
    {
        stepper.doStep(tau, y);

        std::cout << (i + 1) * tau << "  " << y(0) << " " << y(1) << std::endl;
        outfile << (i + 1) * tau << "  " << y(0) << " " << y(1) << std::endl;
    }

    outfile.close();
    return 0;
}