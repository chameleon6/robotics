import sys

xs = map(float, sys.argv[1:])
print xs
h = xs[0]
xs = xs[1:]
n = len(xs)
x_overlap = 0.1
z_overlap = 0.1
print xs

ans =\
'''<?xml version="1.0"?>
<robot xmlns="http://drake.mit.edu"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://drake.mit.edu ../../../doc/drakeURDF.xsd" name="myfirst">
'''

for i in range(1,n):
    size_x = xs[i] - xs[i-1] + x_overlap
    size_z = (n-i) * h + z_overlap
    box_x = (xs[i] + xs[i-1]) / 2 - x_overlap / 2
    box_z = (size_z - z_overlap) / 2 - z_overlap / 2
    s =\
    '''
    \n\n
  <link name="box%d">
    <visual>
      <geometry>
        <box size="%f 100 %f"/>
      </geometry>
      <origin xyz="%f 0 %f"/>
    </visual>
    <collision>
      <geometry>
        <box size="%f 100 %f"/>
      </geometry>
      <origin xyz="%f 0 %f"/>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
      <origin xyz="0.1 0 0"/>
    </inertial>
  </link>
    ''' % (i, size_x, size_z, box_x, box_z, size_x, size_z, box_x, box_z)
    ans += s


ans += \
'''
  <link name="base">
    <visual>
      <geometry>
        <box size="20 100 0.1"/>
      </geometry>
      <origin xyz="0 0 -0.05"/>
    </visual>
    <collision>
      <geometry>
        <box size="20 100 0.1"/>
      </geometry>
      <origin xyz="0 0 -0.05"/>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
      <origin xyz="0.1 0 0"/>
    </inertial>
  </link>
</robot>
'''

f = open('pybox.urdf', 'w')
f.write(ans)
f.close()
