## prepare the xacro file

## gen URDF
    rosrun xacro xacro xacro/robot.xacro -o urdf/wr2.urdf DEBUG:=false  

* for quad-sdk:  

          <gazebo reference='jtoe0'>
            <preserveFixedJoint>true</preserveFixedJoint>
          </gazebo>
          <gazebo reference='jtoe1'>
            <preserveFixedJoint>true</preserveFixedJoint>
          </gazebo>
          <gazebo reference='jtoe2'>
            <preserveFixedJoint>true</preserveFixedJoint>
          </gazebo>
          <gazebo reference='jtoe3'>
            <preserveFixedJoint>true</preserveFixedJoint>
          </gazebo>
          <!-- <gazebo reference='floating_base'>
            <preserveFixedJoint>true</preserveFixedJoint>
          </gazebo> -->
          <gazebo reference='hip0_fixed'>
            <preserveFixedJoint>true</preserveFixedJoint>
          </gazebo>
          <gazebo reference='hip1_fixed'>
            <preserveFixedJoint>true</preserveFixedJoint>
          </gazebo>
          <gazebo reference='hip2_fixed'>
            <preserveFixedJoint>true</preserveFixedJoint>
          </gazebo>
          <gazebo reference='hip3_fixed'>
            <preserveFixedJoint>true</preserveFixedJoint>
          </gazebo>
        </robot>

## gen sdf file 
    gz sdf -p urdf/wr2.urdf > sdf_mesh/wr2.sdf
