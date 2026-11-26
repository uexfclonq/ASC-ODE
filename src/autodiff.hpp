#ifndef AUTODIFF_HPP
#define AUTODIFF_HPP

#include <cstddef>
#include <ostream>
#include <cmath>
#include <array>


namespace ASC_ode
{

  // the N-th independent variable with value v, its gradient is the N-th unit vector
  template <size_t N, typename T = double>
  class Variable
  {
  private:
    T m_val;
  public:
    Variable (T v) : m_val(v) {}
    T value() const { return m_val; }
  };

  // the derivative of a plain number is zero
  template <typename T = double>
  auto derivative (T v, size_t /*index*/) { return T(0); }


  // value and gradient with respect to N independent variables.
  // T can be an AutoDiff again, then we get second derivatives.
  template <size_t N, typename T = double>
  class AutoDiff
  {
  private:
    T m_val;
    std::array<T, N> m_deriv;
  public:
    AutoDiff () : m_val(0), m_deriv{} {}

    // If v is an AutoDiff itself (nested AutoDiff), its derivatives become
    // also the outer derivatives, see demo_autodiff.cpp
    AutoDiff (T v) : m_val(v), m_deriv{}
    {
      for (size_t i = 0; i < N; i++)
        m_deriv[i] = derivative(v, i);
    }

    template <size_t I>
    AutoDiff (Variable<I, T> var) : m_val(var.value()), m_deriv{}
    {
      m_deriv[I] = 1.0;
    }

    T value() const { return m_val; }
    std::array<T, N>& deriv() { return m_deriv; }
    const std::array<T, N>& deriv() const { return m_deriv; }
  };


  template <size_t N, typename T = double>
  auto derivative (AutoDiff<N, T> v, size_t index)
  {
    return v.deriv()[index];
  }



  template <size_t N, typename T>
  std::ostream & operator<< (std::ostream& os, const AutoDiff<N, T>& ad)
  {
    os << "Value: " << ad.value() << ", Deriv: [";
    for (size_t i = 0; i < N; i++)
      {
        os << ad.deriv()[i];
        if (i < N - 1) os << ", ";
      }
    os << "]";
    return os;
  }


  // every operation computes the value, and the gradient by the rules of calculus

  template <size_t N, typename T = double>
  AutoDiff<N, T> operator+ (const AutoDiff<N, T>& a, const AutoDiff<N, T>& b)
  {
    AutoDiff<N, T> result(a.value() + b.value());
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = a.deriv()[i] + b.deriv()[i];
    return result;
  }

  template <size_t N, typename T = double>
  auto operator+ (T a, const AutoDiff<N, T>& b) { return AutoDiff<N, T>(a) + b; }

  template <size_t N, typename T = double>
  AutoDiff<N, T> operator- (const AutoDiff<N, T>& a)
  {
    AutoDiff<N, T> result(-a.value());
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = -a.deriv()[i];
    return result;
  }


  // product rule
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator* (const AutoDiff<N, T>& a, const AutoDiff<N, T>& b)
  {
    AutoDiff<N, T> result(a.value() * b.value());
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = a.deriv()[i] * b.value() + a.value() * b.deriv()[i];
    return result;
  }

  template <size_t N, typename T = double>
  auto operator* (T a, const AutoDiff<N, T>& b)
  {
    return AutoDiff<N, T>(a) * b;
  }

  template <size_t N, typename T = double>
  auto operator* (const AutoDiff<N, T>& a, T b)
  {
    return a * AutoDiff<N, T>(b);
  }

  // sin and cos of doubles must still be found next to the AutoDiff versions
  using std::sin;
  using std::cos;

  // chain rule
  template <size_t N, typename T = double>
  AutoDiff<N, T> sin(const AutoDiff<N, T> &a)
  {
    AutoDiff<N, T> result(sin(a.value()));
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = cos(a.value()) * a.deriv()[i];
    return result;
  }


} // namespace ASC_ode

#endif
