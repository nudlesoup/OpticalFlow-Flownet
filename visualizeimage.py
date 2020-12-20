import cvbase as cvb
import numpy as np
# to visualize a flow file
cvb.show_flow('inverted.flo')
# to visualize a loaded flow map
flow = np.random.rand(100, 100, 2).astype(np.float32)
cvb.show_flow(flow)
cvb.savae
exit()
