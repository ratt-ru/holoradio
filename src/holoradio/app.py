import jax
import jax.numpy as jnp
import numpy as np
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec


class HoloRadio(Application):
  def compose(self):
    count_condition = CountCondition(self, count=10)
    data_gen = DataGeneratorOp(self, count_condition, name="data_gen")
    flag_op = FlagOp(self, name="flagger")
    calibrate_op = CalibrationOp(self, name="calibration")
    self.add_flow(data_gen, flag_op, {("vis", "vis"), ("flag", "flag")})
    self.add_flow(data_gen, calibrate_op, {("vis", "vis")})
    self.add_flow(data_gen, calibrate_op, {("weight", "weight")})
    self.add_flow(flag_op, calibrate_op, {("flag", "flag")})


class DataGeneratorOp(Operator):
  def __init__(
    self,
    fragment,
    *args,
    times=128,
    baselines=2016,
    frequencies=128,
    polarizations=4,
    **kwargs,
  ):
    super().__init__(fragment, *args, **kwargs)
    self.times = times
    self.baselines = baselines
    self.frequencies = frequencies
    self.polarizations = polarizations
    self.count = 0

  def setup(self, spec: OperatorSpec):
    spec.output("flag")
    spec.output("vis")
    spec.output("weight")

  def compute(self, op_input, op_output, context):
    shape = (self.times, self.baselines, self.frequencies, self.polarizations)
    key = jax.random.key(42)
    flag = jax.random.randint(key, shape, 0, 8, np.uint8)
    vis = jax.random.normal(key, shape, np.float32) * 0j
    vis += jax.random.normal(key, shape, np.float32) * 1j
    weight = jax.random.normal(key, shape, np.float32)
    print(f"{(vis.nbytes + flag.nbytes + weight.nbytes)/1024.**3:.01f}GB")
    op_output.emit(flag, "flag")
    op_output.emit(vis, "vis")
    op_output.emit(weight, "weight")
    self.count += 1


@jax.jit
def flag_fn_impl(vis, flag):
  return flag | (jnp.abs(vis) > 1.2)


flag_fn = jax.vmap(flag_fn_impl, in_axes=[0, 0], out_axes=0)
flag_fn = flag_fn_impl


class FlagOp(Operator):
  def setup(self, spec: OperatorSpec):
    spec.input("vis")
    spec.input("flag")
    spec.output("flag")

  def compute(self, op_input, op_output, context):
    flag = jnp.array(op_input.receive("flag"))
    vis = jnp.array(op_input.receive("vis"))
    op_output.emit(flag_fn(vis, flag), "flag")


@jax.jit
def sum_fn_impl(vis, weight, flag):
  vis *= weight
  return jnp.nansum(jnp.where(flag != 0, vis, vis.dtype.type(0)))


sum_fn = jax.vmap(sum_fn_impl, in_axes=[0, 0, 0], out_axes=0)
sum_fn = sum_fn_impl


class CalibrationOp(Operator):
  def setup(self, spec: OperatorSpec):
    spec.input("vis")
    spec.input("flag")
    spec.input("weight")

  def compute(self, op_input, op_output, context):
    flag = jnp.array(op_input.receive("flag"))
    vis = jnp.array(op_input.receive("vis"))
    weight = jnp.array(op_input.receive("weight"))

    print(sum_fn(flag, vis, weight))
