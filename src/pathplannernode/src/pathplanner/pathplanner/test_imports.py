"""
模块导入测试脚本
用于验证所有模块是否正确创建和导入
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_all_imports():
    """测试所有模块的导入"""
    print("🔍 开始测试模块导入...")
    
    try:
        from ultralytics import settings
        print(settings['datasets_dir']) 
        print("\n1️⃣ 测试注意力计算模块...")
        from attention_computation import (
            compute_internal_attention_from_masked_pc,
            visualize_internal_attention
        )
        print("   ✅ attention_computation 导入成功")
        
        print("\n2️⃣ 试点云处理模块...")
        from pointcloud_processing import PointCloudProcessor
        print("   ✅ pointcloud_processing 导入成功")
        
        print("\n3️⃣ 测试路径生成模块...")
        from path_generation import PathGenerator
        print("   ✅ path_generation 导入成功")
        
        print("\n4️⃣ 测试路径优化模块...")
        from path_optimization import PathOptimizer
        print("   ✅ path_optimization 导入成功")
        
        print("\n5️⃣ 测试局部坐标系模块...")
        from local_coordinate import LocalCoordinateCalculator
        print("   ✅ local_coordinate 导入成功")
        
        print("\n6️⃣ 测试可视化模块...")
        from visualization import PathVisualizer
        print("   ✅ visualization 导入成功")
        
        print("\n7️⃣ 测试工具模块...")
        from utils import preprocess_depth
        print("   ✅ utils 导入成功")

        
        print("\n8️⃣ 测试主入口模块...")
        from attention_path import InteractiveSegmentation, main, GetEdge
        print("   ✅ attention_path 导入成功")
        
        print("\n" + "="*50)
        print("🎉 所有模块导入测试通过！")
        print("="*50)
        
        # 显示模块功能概览
        print("\n📋 可用模块功能:")
        print("  • InteractiveSegmentation - 交互式分割主类")
        print("  • PointCloudProcessor - 点云处理器")
        print("  • PathGenerator - 路径生成器")
        print("  • PathOptimizer - 路径优化器")
        print("  • LocalCoordinateCalculator - 局部坐标系计算器")
        print("  • PathVisualizer - 可视化工具")
        print("  • compute_internal_attention_from_masked_pc - 注意力计算")
        print("  • visualize_internal_attention - 注意力可视化")
        print("  • preprocess_depth - 深度图预处理")
        
        
        return True
        
    except ImportError as e:
        print(f"\n❌ 导入失败：{e}")
        print(f"   错误位置：{e.name if hasattr(e, 'name') else 'unknown'}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ 发生错误：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_all_imports()
    sys.exit(0 if success else 1)
