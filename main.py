from typing import Callable, Union

import numpy as np


# x would be in following form
State = Union[float, np.ndarray]
# Slope function would be in the following form
# Would take two arguments and return one argument
SlopeFunction = Callable[[float, State], State]


# Following function would calculate x of the next step
def rk4_step(f:SlopeFunction, x0:State, t0:float, t1:float) -> State:
    """
    One time step of Runge-Kutta method
    f  : function dx_dt(t0, x0)
    x0 : initial condition
    t0 : this step time
    t1 : next step time
    """
    print(f"x at the current step = {x0}")
    print(f"t at the current step t0 = {t0}")
    print(f"t at the next step t1 = {t1}")

    # time step
    delta_t = t1 - t0
    print(f"delta_t = {delta_t}")

    # half of time step
    delta_t_half = delta_t * 0.5
    print(f"delta_t_half = {delta_t_half}")

    t_half = t0 + delta_t_half
    
    # Step 1
    s1 = f(t0, x0)
    print(f"first step : slope at t0 = {s1}")

    # Step 2
    s2 = f(t_half, x0 + s1 * delta_t_half)
    print(f"second step : estimated slope at t[1/2] using x[1/2] from Euler's method = {s2}")

    # Step 3
    s3 = f(t_half, x0 + s2 * delta_t_half)
    print(f"third step : estimated slope at t[1/2] using x[1/2] using slope from second step = {s3}")

    # Step 4
    s4 = f(t1, x0 + s3 * delta_t)
    print(f"fourth step : estimated slope at t[1] using x[1] using slope from third step = {s4}")

    # Step 5
    s = (1.0 / 6.0) * (s1 + (s2 + s3) * 2 + s4)
    print(f"weighted average of the slopes = {s}")

    # Step 6
    x1 = x0 + s * delta_t
    print(f"finally x of the next time step = {x1}")

    return x1


def dx_dt(t:float, x:State) -> State:
  """
  a0 x_dot + a1 x = 0
  a0 x_dot = -a1 x
  x_dot = - (a1 * x) / a0
  """
  a0 = 1
  a1 = 0.5
  return (-a1 / a0 * x)


def main():
  t0_sec = 0.0
  t1_sec = 1.0

  x0 = 4

  x1_rk4 = rk4_step(dx_dt, x0, t0_sec, t1_sec)
  print(f"x of Next step by Heun's method = {x1_rk4}")


if "__main__" == __name__:
  main()
