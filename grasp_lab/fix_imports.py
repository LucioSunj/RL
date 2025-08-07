#!/usr/bin/env python3
"""
修复导入路径错误的脚本
解决从 'arm.xxx' 到相对导入的问题
"""

import os
import re
from pathlib import Path


def fix_imports_in_file(file_path: str, replacements: dict):
    """修复单个文件中的导入语句"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 应用所有替换规则
        for old_import, new_import in replacements.items():
            content = re.sub(old_import, new_import, content)
        
        # 如果内容有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False


def find_python_files(directory: str):
    """递归找到所有Python文件"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # 跳过 __pycache__ 目录
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files


def main():
    """主修复函数"""
    
    # 定义需要修复的导入替换规则
    replacements = {
        # robot.py 中的导入问题
        r'from arm\.geometry import Geometry3D': 'from ..geometry.simplex import Geometry',
        r'from arm\.geometry import (.+)': r'from ..geometry import \1',
        
        # geometry.py 中的导入问题  
        r'from arm\.constanst import (.+)': r'from ...constanst import \1',
        r'from arm\.(.+) import (.+)': r'from ..\.\1 import \2',
        
        # 其他可能的 arm. 开头的导入
        r'from arm\.robot import (.+)': r'from ..robot import \1',
        r'from arm\.motion_planning import (.+)': r'from ..motion_planning import \1',
        r'from arm\.controller import (.+)': r'from ..controller import \1',
        r'from arm\.interface import (.+)': r'from ..interface import \1',
        r'from arm\.utils import (.+)': r'from ..utils import \1',
        
        # 处理一些特殊的导入情况
        r'import arm\.(.+)': r'from .. import \1',
    }
    
    # 特殊处理：如果 Geometry3D 不存在，将其替换为 Geometry
    geometry_replacements = {
        r'Geometry3D': 'Geometry',
    }
    
    print("🔧 开始修复导入路径问题...")
    
    # 获取manipulator_grasp目录下的所有Python文件
    manipulator_grasp_dir = "manipulator_grasp"
    
    if not os.path.exists(manipulator_grasp_dir):
        print(f"❌ 未找到目录: {manipulator_grasp_dir}")
        return
    
    python_files = find_python_files(manipulator_grasp_dir)
    print(f"📁 找到 {len(python_files)} 个Python文件")
    
    fixed_count = 0
    
    # 修复每个文件
    for file_path in python_files:
        # 应用导入修复
        if fix_imports_in_file(file_path, replacements):
            fixed_count += 1
        
        # 应用特殊的类名替换
        fix_imports_in_file(file_path, geometry_replacements)
    
    print(f"\n✅ 完成! 修复了 {fixed_count} 个文件")
    
    # 额外的手动修复提示
    print("\n📝 可能还需要手动检查的问题:")
    print("1. 检查 Geometry3D 是否应该改为 Geometry")
    print("2. 检查是否有其他自定义的导入路径需要调整")
    print("3. 如果还有错误，可能需要调整 __init__.py 文件")


if __name__ == "__main__":
    main()