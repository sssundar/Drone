# This module will need to synchronizer (with batch delay) the input from the sampler module.
# This module will need to run the Madgwick Complementary Filter.
# This module will need to run Dead Reckoning.
# This module should take the controller as an input so it can drive the Controller when it has enough data to step forward its estimation by a tick.
# If you do this with callbacks at the top level, then you can put a limit on how long the loop goes.


class Estimator(object):
  def __init__(self):
    return
