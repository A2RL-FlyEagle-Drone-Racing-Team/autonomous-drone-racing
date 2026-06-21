#
## Rotation 与欧拉角
例：XYZ欧拉角（内旋，滚转x-俯仰y-偏航z，旋转顺序为偏航-俯仰-滚转），四元数顺序为（w, x, y, z）

```python
from scipy.spatial.transform import Rotation

euler_xyz_deg = [80, 0, 120]
rot = Rotation.from_euler("xyz", euler_xyz_deg, degrees=True)
quat = rot.as_quat()  # [0.3213938 , 0.5566704 , 0.66341395, 0.38302222]
euler = rot.as_euler("xyz", degrees=True)  #  [ 80.,   0., 120.]
```