from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.gp import gp_Pnt

# 创建球体
sphere1 = BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 0), 10.0).Shape()
sphere2 = BRepPrimAPI_MakeSphere(gp_Pnt(0, 16, 0), 5.0).Shape()

# 创建STEP写入器并导出
step_writer = STEPControl_Writer()
step_writer.Transfer(sphere1, STEPControl_AsIs)
step_writer.Transfer(sphere2, STEPControl_AsIs)

status = step_writer.Write("two_spheres.step")

if status == IFSelect_RetDone:
    print("STEP文件创建成功: two_spheres.step")
else:
    print("STEP文件创建失败")