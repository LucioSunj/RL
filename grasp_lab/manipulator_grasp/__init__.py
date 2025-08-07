# 导入路径修复
import sys
import os

# 添加arm目录到Python路径，这样 'from arm.xxx' 的导入就能工作
_arm_path = os.path.join(os.path.dirname(__file__), 'arm')
if _arm_path not in sys.path:
    sys.path.insert(0, _arm_path)

# 清理变量避免污染命名空间
del _arm_path