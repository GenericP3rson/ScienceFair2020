��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8��

conv2d/kernelVarHandleOp*
shared_nameconv2d/kernel*
dtype0*
_output_shapes
: *
shape:�
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*'
_output_shapes
:�
o
conv2d/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:�*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes	
:�
�
conv2d_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:��* 
shared_nameconv2d_1/kernel
}
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*(
_output_shapes
:��
s
conv2d_1/biasVarHandleOp*
shape:�*
shared_nameconv2d_1/bias*
dtype0*
_output_shapes
: 
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes	
:�
u
dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	�@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	�@
l

dense/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:@
x
dense_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:@	*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:@	
p
dense_1/biasVarHandleOp*
shared_namedense_1/bias*
dtype0*
_output_shapes
: *
shape:	
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:	
^
totalVarHandleOp*
shared_nametotal*
dtype0*
_output_shapes
: *
shape: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shared_namecount*
dtype0*
_output_shapes
: *
shape: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 

NoOpNoOp
�)
ConstConst"/device:CPU:0*�)
value�(B�( B�(
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-2
layer-10
layer_with_weights-3
layer-11
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
 	keras_api
R
!regularization_losses
"	variables
#trainable_variables
$	keras_api
R
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
R
/regularization_losses
0	variables
1trainable_variables
2	keras_api
R
3regularization_losses
4	variables
5trainable_variables
6	keras_api
R
7regularization_losses
8	variables
9trainable_variables
:	keras_api
R
;regularization_losses
<	variables
=trainable_variables
>	keras_api
h

?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
h

Ekernel
Fbias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
 
 
8
0
1
)2
*3
?4
@5
E6
F7
8
0
1
)2
*3
?4
@5
E6
F7
�
regularization_losses
Kmetrics
	variables
trainable_variables
Lnon_trainable_variables

Mlayers
Nlayer_regularization_losses
 
 
 
 
�
regularization_losses
Ometrics
	variables
Pnon_trainable_variables
trainable_variables

Qlayers
Rlayer_regularization_losses
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
Smetrics
	variables
Tnon_trainable_variables
trainable_variables

Ulayers
Vlayer_regularization_losses
 
 
 
�
regularization_losses
Wmetrics
	variables
Xnon_trainable_variables
trainable_variables

Ylayers
Zlayer_regularization_losses
 
 
 
�
!regularization_losses
[metrics
"	variables
\non_trainable_variables
#trainable_variables

]layers
^layer_regularization_losses
 
 
 
�
%regularization_losses
_metrics
&	variables
`non_trainable_variables
'trainable_variables

alayers
blayer_regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
�
+regularization_losses
cmetrics
,	variables
dnon_trainable_variables
-trainable_variables

elayers
flayer_regularization_losses
 
 
 
�
/regularization_losses
gmetrics
0	variables
hnon_trainable_variables
1trainable_variables

ilayers
jlayer_regularization_losses
 
 
 
�
3regularization_losses
kmetrics
4	variables
lnon_trainable_variables
5trainable_variables

mlayers
nlayer_regularization_losses
 
 
 
�
7regularization_losses
ometrics
8	variables
pnon_trainable_variables
9trainable_variables

qlayers
rlayer_regularization_losses
 
 
 
�
;regularization_losses
smetrics
<	variables
tnon_trainable_variables
=trainable_variables

ulayers
vlayer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1
�
Aregularization_losses
wmetrics
B	variables
xnon_trainable_variables
Ctrainable_variables

ylayers
zlayer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

E0
F1
�
Gregularization_losses
{metrics
H	variables
|non_trainable_variables
Itrainable_variables

}layers
~layer_regularization_losses

0
 
N
0
1
2
3
4
5
6
	7

8
9
10
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


�total

�count
�
_fn_kwargs
�regularization_losses
�	variables
�trainable_variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

�0
�1
 
�
�regularization_losses
�metrics
�	variables
�non_trainable_variables
�trainable_variables
�layers
 �layer_regularization_losses
 

�0
�1
 
 *
dtype0*
_output_shapes
: 
�
serving_default_conv2d_inputPlaceholder*$
shape:���������

*
dtype0*/
_output_shapes
:���������


�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������	*
Tin
2	*+
_gradient_op_typePartitionedCall-6302*+
f&R$
"__inference_signature_wrapper_6013
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: *+
_gradient_op_typePartitionedCall-6334*&
f!R
__inference__traced_save_6333*
Tout
2
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biastotalcount*)
f$R"
 __inference__traced_restore_6376*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*
_output_shapes
: *+
_gradient_op_typePartitionedCall-6377��
�+
�
 __inference__traced_restore_6376
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias%
!assignvariableop_6_dense_1_kernel#
assignvariableop_7_dense_1_bias
assignvariableop_8_total
assignvariableop_9_count
identity_11��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
�
RestoreV2/shape_and_slicesConst"/device:CPU:0*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:
�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:z
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:~
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:}
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:x
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0x
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0�
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2: : : : : : : :	 :
 :+ '
%
_user_specified_namefile_prefix: 
�
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_5798

inputs
identity�Q
dropout/rateConst*
valueB
 *��L>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:�����������
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*0
_output_shapes
:����������*
T0�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:����������R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:����������j
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:����������x
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:����������r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������b
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�3
�
__inference__wrapped_model_5602
conv2d_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identity��(sequential/conv2d/BiasAdd/ReadVariableOp�'sequential/conv2d/Conv2D/ReadVariableOp�*sequential/conv2d_1/BiasAdd/ReadVariableOp�)sequential/conv2d_1/Conv2D/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�(sequential/dense_1/MatMul/ReadVariableOp�
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:��
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*0
_output_shapes
:����������*
T0�
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
sequential/activation/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
 sequential/max_pooling2d/MaxPoolMaxPool(sequential/activation/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:�����������
sequential/dropout/IdentityIdentity)sequential/max_pooling2d/MaxPool:output:0*
T0*0
_output_shapes
:�����������
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:���
sequential/conv2d_1/Conv2DConv2D$sequential/dropout/Identity:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:�����������
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
sequential/activation_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*0
_output_shapes
:����������*
T0�
"sequential/max_pooling2d_1/MaxPoolMaxPool*sequential/activation_1/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:�����������
sequential/dropout_1/IdentityIdentity+sequential/max_pooling2d_1/MaxPool:output:0*0
_output_shapes
:����������*
T0q
 sequential/flatten/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
sequential/flatten/ReshapeReshape&sequential/dropout_1/Identity:output:0)sequential/flatten/Reshape/shape:output:0*(
_output_shapes
:����������*
T0�
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�@�
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@	�
sequential/dense_1/MatMulMatMul!sequential/dense/BiasAdd:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
IdentityIdentity#sequential/dense_1/BiasAdd:output:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*N
_input_shapes=
;:���������

::::::::2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:, (
&
_user_specified_nameconv2d_input: : : : : : : : 
�
`
A__inference_dropout_layer_call_and_return_conditional_losses_5732

inputs
identity�Q
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *��L>C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:�����������
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:�����������
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:����������R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*0
_output_shapes
:����������*
T0j
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:����������x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*0
_output_shapes
:����������*

SrcT0
r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*0
_output_shapes
:����������*
T0b
IdentityIdentitydropout/mul_1:z:0*0
_output_shapes
:����������*
T0"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_5739

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�

�
)__inference_sequential_layer_call_fn_6130

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8**
config_proto

GPU 

CPU2J 8*
Tin
2	*'
_output_shapes
:���������	*+
_gradient_op_typePartitionedCall-5948*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5947*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������	*
T0"
identityIdentity:output:0*N
_input_shapes=
;:���������

::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: : : : : 
�K
�
D__inference_sequential_layer_call_and_return_conditional_losses_6081

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:��
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:�����������
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:����������*
T0k
activation/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*
ksize
*
paddingVALID*0
_output_shapes
:����������*
strides
Y
dropout/dropout/rateConst*
valueB
 *��L>*
dtype0*
_output_shapes
: c
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:g
"dropout/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: g
"dropout/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:�����������
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:�����������
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:����������Z
dropout/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
_output_shapes
: *
T0^
dropout/dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?�
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
_output_shapes
: *
T0�
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*0
_output_shapes
:�����������
dropout/dropout/mulMulmax_pooling2d/MaxPool:output:0dropout/dropout/truediv:z:0*
T0*0
_output_shapes
:�����������
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:�����������
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:�����������
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:���
conv2d_1/Conv2DConv2Ddropout/dropout/mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:�����������
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������o
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:����������[
dropout_1/dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *��L>g
dropout_1/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:i
$dropout_1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_1/dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:�����������
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*0
_output_shapes
:����������*
T0�
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*0
_output_shapes
:����������*
T0\
dropout_1/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_1/dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: �
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*0
_output_shapes
:����������*
T0�
dropout_1/dropout/mulMul max_pooling2d_1/MaxPool:output:0dropout_1/dropout/truediv:z:0*0
_output_shapes
:����������*
T0�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*0
_output_shapes
:����������*

SrcT0
�
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*0
_output_shapes
:����������*
T0f
flatten/Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:�
flatten/ReshapeReshapedropout_1/dropout/mul_1:z:0flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�@�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@	�
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:���������	*
T0�
IdentityIdentitydense_1/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*N
_input_shapes=
;:���������

::::::::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : 
�
D
(__inference_dropout_1_layer_call_fn_6233

inputs
identity�
PartitionedCallPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5817*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_5805*
Tout
2i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:����������*
T0"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5634

inputs
identity�
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4������������������������������������{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�)
�
D__inference_sequential_layer_call_and_return_conditional_losses_5986

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-5621*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5615*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:�����������
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5704*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_5698*
Tout
2**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2�
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5640*P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5634*
Tout
2�
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5751*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5739*
Tout
2**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-5662*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5656*
Tout
2**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2�
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5770*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_5764*
Tout
2**
config_proto

GPU 

CPU2J 8�
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5681*R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5675*
Tout
2�
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5817*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_5805*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:�����������
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5834*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5828*
Tout
2�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������@*+
_gradient_op_typePartitionedCall-5857*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5851*
Tout
2�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������	*+
_gradient_op_typePartitionedCall-5884*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_5878*
Tout
2�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*'
_output_shapes
:���������	*
T0"
identityIdentity:output:0*N
_input_shapes=
;:���������

::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
�
B
&__inference_flatten_layer_call_fn_6244

inputs
identity�
PartitionedCallPartitionedCallinputs*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5828*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:����������*
Tin
2*+
_gradient_op_typePartitionedCall-5834a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5675

inputs
identity�
MaxPoolMaxPoolinputs*
ksize
*
paddingVALID*J
_output_shapes8
6:4������������������������������������*
strides
{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4������������������������������������*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�

�
)__inference_sequential_layer_call_fn_6143

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������	*
Tin
2	*+
_gradient_op_typePartitionedCall-5987*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5986*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:���������	*
T0"
identityIdentity:output:0*N
_input_shapes=
;:���������

::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: : : : : 
�
�
'__inference_conv2d_1_layer_call_fn_5667

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5656*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*B
_output_shapes0
.:,����������������������������*+
_gradient_op_typePartitionedCall-5662�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
E
)__inference_activation_layer_call_fn_6153

inputs
identity�
PartitionedCallPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2*+
_gradient_op_typePartitionedCall-5704*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_5698*
Tout
2i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:����������*
T0"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
%__inference_conv2d_layer_call_fn_5626

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*B
_output_shapes0
.:,����������������������������*
Tin
2*+
_gradient_op_typePartitionedCall-5621*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5615�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,����������������������������*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
`
D__inference_activation_layer_call_and_return_conditional_losses_6148

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_5828

inputs
identity^
Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
$__inference_dense_layer_call_fn_6261

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������@*
Tin
2*+
_gradient_op_typePartitionedCall-5857*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5851�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�)
�
D__inference_sequential_layer_call_and_return_conditional_losses_5921
conv2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-5621*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5615*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:�����������
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5704*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_5698*
Tout
2**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2�
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5640*P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5634*
Tout
2�
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2*+
_gradient_op_typePartitionedCall-5751*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5739*
Tout
2�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5662*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5656*
Tout
2�
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5770*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_5764*
Tout
2�
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5675*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5681�
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5817*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_5805*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:�����������
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5834*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5828*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:����������*
Tin
2�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5851*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������@*+
_gradient_op_typePartitionedCall-5857�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������	*
Tin
2*+
_gradient_op_typePartitionedCall-5884*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_5878*
Tout
2�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*N
_input_shapes=
;:���������

::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall: : : : : :, (
&
_user_specified_nameconv2d_input: : : 
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_6239

inputs
identity^
Reshape/shapeConst*
valueB"����   *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*(
_output_shapes
:����������*
T0"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_6178

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
A__inference_dense_1_layer_call_and_return_conditional_losses_5878

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@	i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*.
_input_shapes
:���������@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�,
�
D__inference_sequential_layer_call_and_return_conditional_losses_5896
conv2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2*+
_gradient_op_typePartitionedCall-5621*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5615�
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5704*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_5698*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:�����������
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5640*P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5634*
Tout
2**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2�
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5743*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5732*
Tout
2**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5662*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5656*
Tout
2**
config_proto

GPU 

CPU2J 8�
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2*+
_gradient_op_typePartitionedCall-5770*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_5764*
Tout
2�
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5681*R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5675�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tout
2**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2*+
_gradient_op_typePartitionedCall-5809*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_5798�
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5834*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5828*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:�����������
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5851*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������@*+
_gradient_op_typePartitionedCall-5857�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-5884*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_5878*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������	*
Tin
2�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*N
_input_shapes=
;:���������

::::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall: :, (
&
_user_specified_nameconv2d_input: : : : : : : 
�,
�
D__inference_sequential_layer_call_and_return_conditional_losses_5947

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5621*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5615�
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2*+
_gradient_op_typePartitionedCall-5704*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_5698*
Tout
2�
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5640*P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5634*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:�����������
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5732*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5743�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-5662*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5656*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:�����������
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5770*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_5764*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:�����������
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5675*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5681�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tout
2**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2*+
_gradient_op_typePartitionedCall-5809*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_5798�
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-5834*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5828*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:�����������
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������@*
Tin
2*+
_gradient_op_typePartitionedCall-5857*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5851*
Tout
2�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:���������	*+
_gradient_op_typePartitionedCall-5884*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_5878*
Tout
2�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*N
_input_shapes=
;:���������

::::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
�
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_6223

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*0
_output_shapes
:����������*
T0"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�

�
)__inference_sequential_layer_call_fn_5998
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8**
config_proto

GPU 

CPU2J 8*
Tin
2	*'
_output_shapes
:���������	*+
_gradient_op_typePartitionedCall-5987*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5986*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*N
_input_shapes=
;:���������

::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :, (
&
_user_specified_nameconv2d_input: : : : : 
�+
�
D__inference_sequential_layer_call_and_return_conditional_losses_6117

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:��
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:�����������
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:����������*
T0k
activation/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*0
_output_shapes
:����������*
strides
*
ksize
*
paddingVALIDw
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*0
_output_shapes
:�����������
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:���
conv2d_1/Conv2DConv2Ddropout/Identity:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:�����������
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������o
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*0
_output_shapes
:����������*
strides
*
ksize
*
paddingVALID{
dropout_1/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*0
_output_shapes
:����������f
flatten/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"����   �
flatten/ReshapeReshapedropout_1/Identity:output:0flatten/Reshape/shape:output:0*(
_output_shapes
:����������*
T0�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�@�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:���������@*
T0�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@	�
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
IdentityIdentitydense_1/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*N
_input_shapes=
;:���������

::::::::2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : 
�

�
)__inference_sequential_layer_call_fn_5959
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*+
_gradient_op_typePartitionedCall-5948*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5947*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2	*'
_output_shapes
:���������	�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*N
_input_shapes=
;:���������

::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :, (
&
_user_specified_nameconv2d_input: : : : : : : 
�

�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5656

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:���
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingVALID*B
_output_shapes0
.:,����������������������������*
T0*
strides
�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,�����������������������������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,����������������������������"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
�
_
&__inference_dropout_layer_call_fn_6183

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������*+
_gradient_op_typePartitionedCall-5743*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5732*
Tout
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
?__inference_dense_layer_call_and_return_conditional_losses_6254

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*/
_input_shapes
:����������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�

�
"__inference_signature_wrapper_6013
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*+
_gradient_op_typePartitionedCall-6002*(
f#R!
__inference__wrapped_model_5602*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2	*'
_output_shapes
:���������	�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*N
_input_shapes=
;:���������

::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : 
�
B
&__inference_dropout_layer_call_fn_6188

inputs
identity�
PartitionedCallPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*0
_output_shapes
:����������*
Tin
2*+
_gradient_op_typePartitionedCall-5751*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5739*
Tout
2i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:����������*
T0"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
H
,__inference_max_pooling2d_layer_call_fn_5643

inputs
identity�
PartitionedCallPartitionedCallinputs*+
_gradient_op_typePartitionedCall-5640*P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5634*
Tout
2**
config_proto

GPU 

CPU2J 8*J
_output_shapes8
6:4������������������������������������*
Tin
2�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
`
D__inference_activation_layer_call_and_return_conditional_losses_5698

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
b
F__inference_activation_1_layer_call_and_return_conditional_losses_6193

inputs
identityO
ReluReluinputs*0
_output_shapes
:����������*
T0c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_5805

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:����������d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_6218

inputs
identity�Q
dropout/rateConst*
valueB
 *��L>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:�����������
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:�����������
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:����������R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:����������j
dropout/mulMulinputsdropout/truediv:z:0*0
_output_shapes
:����������*
T0x
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:����������r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������b
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
a
(__inference_dropout_1_layer_call_fn_6228

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*+
_gradient_op_typePartitionedCall-5809*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_5798*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:�����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
b
F__inference_activation_1_layer_call_and_return_conditional_losses_5764

inputs
identityO
ReluReluinputs*0
_output_shapes
:����������*
T0c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�

�
@__inference_conv2d_layer_call_and_return_conditional_losses_5615

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:��
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingVALID*B
_output_shapes0
.:,����������������������������*
T0*
strides
�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:��
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*B
_output_shapes0
.:,����������������������������*
T0�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*B
_output_shapes0
.:,����������������������������*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
&__inference_dense_1_layer_call_fn_6278

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-5884*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_5878*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:���������	*
Tin
2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
�
J
.__inference_max_pooling2d_1_layer_call_fn_5684

inputs
identity�
PartitionedCallPartitionedCallinputs*J
_output_shapes8
6:4������������������������������������*
Tin
2*+
_gradient_op_typePartitionedCall-5681*R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5675*
Tout
2**
config_proto

GPU 

CPU2J 8�
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4������������������������������������*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
`
A__inference_dropout_layer_call_and_return_conditional_losses_6173

inputs
identity�Q
dropout/rateConst*
valueB
 *��L>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:�����������
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: �
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:�����������
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*0
_output_shapes
:����������*
T0R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: �
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:����������j
dropout/mulMulinputsdropout/truediv:z:0*0
_output_shapes
:����������*
T0x
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:����������r
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������b
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
G
+__inference_activation_1_layer_call_fn_6198

inputs
identity�
PartitionedCallPartitionedCallinputs*+
_gradient_op_typePartitionedCall-5770*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_5764*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*0
_output_shapes
:����������i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*/
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
?__inference_dense_layer_call_and_return_conditional_losses_5851

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	�@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
� 
�
__inference__traced_save_6333
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_384c6eb4481d43c5a5f31ca82704b05e/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:
*�
value�B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE�
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:
*'
valueB
B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"/device:CPU:0*
dtypes
2
*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 �
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*q
_input_shapes`
^: :�:�:��:�:	�@:@:@	:	: : : 2
SaveV2_1SaveV2_12(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV2: :	 :
 : :+ '
%
_user_specified_namefile_prefix: : : : : : : 
�
�
A__inference_dense_1_layer_call_and_return_conditional_losses_6271

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@	i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������	"
identityIdentity:output:0*.
_input_shapes
:���������@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
M
conv2d_input=
serving_default_conv2d_input:0���������

;
dense_10
StatefulPartitionedCall:0���������	tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�:
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-2
layer-10
layer_with_weights-3
layer-11
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"�6
_tf_keras_sequential�6{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 10, 10, 3], "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 10, 10, 3], "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
�
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "conv2d_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 10, 10, 3], "config": {"batch_input_shape": [null, 10, 10, 3], "dtype": "float32", "sparse": false, "name": "conv2d_input"}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 10, 10, 3], "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 10, 10, 3], "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
�
regularization_losses
	variables
trainable_variables
 	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
!regularization_losses
"	variables
#trainable_variables
$	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
%regularization_losses
&	variables
'trainable_variables
(	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
�
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�
;regularization_losses
<	variables
=trainable_variables
>	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
�

Ekernel
Fbias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
"
	optimizer
 "
trackable_list_wrapper
X
0
1
)2
*3
?4
@5
E6
F7"
trackable_list_wrapper
X
0
1
)2
*3
?4
@5
E6
F7"
trackable_list_wrapper
�
regularization_losses
Kmetrics
	variables
trainable_variables
Lnon_trainable_variables

Mlayers
Nlayer_regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses
Ometrics
	variables
Pnon_trainable_variables
trainable_variables

Qlayers
Rlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&�2conv2d/kernel
:�2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
Smetrics
	variables
Tnon_trainable_variables
trainable_variables

Ulayers
Vlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses
Wmetrics
	variables
Xnon_trainable_variables
trainable_variables

Ylayers
Zlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
!regularization_losses
[metrics
"	variables
\non_trainable_variables
#trainable_variables

]layers
^layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
%regularization_losses
_metrics
&	variables
`non_trainable_variables
'trainable_variables

alayers
blayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)��2conv2d_1/kernel
:�2conv2d_1/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
�
+regularization_losses
cmetrics
,	variables
dnon_trainable_variables
-trainable_variables

elayers
flayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
/regularization_losses
gmetrics
0	variables
hnon_trainable_variables
1trainable_variables

ilayers
jlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
3regularization_losses
kmetrics
4	variables
lnon_trainable_variables
5trainable_variables

mlayers
nlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
7regularization_losses
ometrics
8	variables
pnon_trainable_variables
9trainable_variables

qlayers
rlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
;regularization_losses
smetrics
<	variables
tnon_trainable_variables
=trainable_variables

ulayers
vlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	�@2dense/kernel
:@2
dense/bias
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
�
Aregularization_losses
wmetrics
B	variables
xnon_trainable_variables
Ctrainable_variables

ylayers
zlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :@	2dense_1/kernel
:	2dense_1/bias
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
�
Gregularization_losses
{metrics
H	variables
|non_trainable_variables
Itrainable_variables

}layers
~layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
	7

8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�
_fn_kwargs
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�metrics
�	variables
�non_trainable_variables
�trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
D__inference_sequential_layer_call_and_return_conditional_losses_6081
D__inference_sequential_layer_call_and_return_conditional_losses_6117
D__inference_sequential_layer_call_and_return_conditional_losses_5921
D__inference_sequential_layer_call_and_return_conditional_losses_5896�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_sequential_layer_call_fn_5959
)__inference_sequential_layer_call_fn_6130
)__inference_sequential_layer_call_fn_5998
)__inference_sequential_layer_call_fn_6143�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_5602�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+
conv2d_input���������


�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
@__inference_conv2d_layer_call_and_return_conditional_losses_5615�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
%__inference_conv2d_layer_call_fn_5626�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
D__inference_activation_layer_call_and_return_conditional_losses_6148�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_activation_layer_call_fn_6153�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5634�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
,__inference_max_pooling2d_layer_call_fn_5643�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
A__inference_dropout_layer_call_and_return_conditional_losses_6178
A__inference_dropout_layer_call_and_return_conditional_losses_6173�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_dropout_layer_call_fn_6188
&__inference_dropout_layer_call_fn_6183�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5656�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
'__inference_conv2d_1_layer_call_fn_5667�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
F__inference_activation_1_layer_call_and_return_conditional_losses_6193�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_activation_1_layer_call_fn_6198�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5675�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
.__inference_max_pooling2d_1_layer_call_fn_5684�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
C__inference_dropout_1_layer_call_and_return_conditional_losses_6223
C__inference_dropout_1_layer_call_and_return_conditional_losses_6218�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_dropout_1_layer_call_fn_6233
(__inference_dropout_1_layer_call_fn_6228�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_flatten_layer_call_and_return_conditional_losses_6239�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_flatten_layer_call_fn_6244�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_dense_layer_call_and_return_conditional_losses_6254�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_dense_layer_call_fn_6261�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_1_layer_call_and_return_conditional_losses_6271�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_1_layer_call_fn_6278�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
6B4
"__inference_signature_wrapper_6013conv2d_input
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
A__inference_flatten_layer_call_and_return_conditional_losses_6239b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
?__inference_dense_layer_call_and_return_conditional_losses_6254]?@0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� �
D__inference_activation_layer_call_and_return_conditional_losses_6148j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� y
&__inference_dense_1_layer_call_fn_6278OEF/�,
%�"
 �
inputs���������@
� "����������	�
%__inference_conv2d_layer_call_fn_5626�I�F
?�<
:�7
inputs+���������������������������
� "3�0,�����������������������������
D__inference_sequential_layer_call_and_return_conditional_losses_6117r)*?@EF?�<
5�2
(�%
inputs���������


p 

 
� "%�"
�
0���������	
� �
)__inference_sequential_layer_call_fn_5959k)*?@EFE�B
;�8
.�+
conv2d_input���������


p

 
� "����������	�
A__inference_dropout_layer_call_and_return_conditional_losses_6173n<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
)__inference_activation_layer_call_fn_6153]8�5
.�+
)�&
inputs����������
� "!������������
D__inference_sequential_layer_call_and_return_conditional_losses_6081r)*?@EF?�<
5�2
(�%
inputs���������


p

 
� "%�"
�
0���������	
� �
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5675�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
C__inference_dropout_1_layer_call_and_return_conditional_losses_6223n<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
C__inference_dropout_1_layer_call_and_return_conditional_losses_6218n<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
A__inference_dropout_layer_call_and_return_conditional_losses_6178n<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� x
$__inference_dense_layer_call_fn_6261P?@0�-
&�#
!�
inputs����������
� "����������@�
.__inference_max_pooling2d_1_layer_call_fn_5684�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
D__inference_sequential_layer_call_and_return_conditional_losses_5921x)*?@EFE�B
;�8
.�+
conv2d_input���������


p 

 
� "%�"
�
0���������	
� �
,__inference_max_pooling2d_layer_call_fn_5643�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
A__inference_dense_1_layer_call_and_return_conditional_losses_6271\EF/�,
%�"
 �
inputs���������@
� "%�"
�
0���������	
� �
@__inference_conv2d_layer_call_and_return_conditional_losses_5615�I�F
?�<
:�7
inputs+���������������������������
� "@�=
6�3
0,����������������������������
� �
)__inference_sequential_layer_call_fn_5998k)*?@EFE�B
;�8
.�+
conv2d_input���������


p 

 
� "����������	�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5656�)*J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
D__inference_sequential_layer_call_and_return_conditional_losses_5896x)*?@EFE�B
;�8
.�+
conv2d_input���������


p

 
� "%�"
�
0���������	
� �
"__inference_signature_wrapper_6013�)*?@EFM�J
� 
C�@
>
conv2d_input.�+
conv2d_input���������

"1�.
,
dense_1!�
dense_1���������	�
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5634�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
&__inference_dropout_layer_call_fn_6183a<�9
2�/
)�&
inputs����������
p
� "!�����������
&__inference_flatten_layer_call_fn_6244U8�5
.�+
)�&
inputs����������
� "������������
&__inference_dropout_layer_call_fn_6188a<�9
2�/
)�&
inputs����������
p 
� "!������������
)__inference_sequential_layer_call_fn_6130e)*?@EF?�<
5�2
(�%
inputs���������


p

 
� "����������	�
__inference__wrapped_model_5602|)*?@EF=�:
3�0
.�+
conv2d_input���������


� "1�.
,
dense_1!�
dense_1���������	�
+__inference_activation_1_layer_call_fn_6198]8�5
.�+
)�&
inputs����������
� "!������������
F__inference_activation_1_layer_call_and_return_conditional_losses_6193j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
)__inference_sequential_layer_call_fn_6143e)*?@EF?�<
5�2
(�%
inputs���������


p 

 
� "����������	�
'__inference_conv2d_1_layer_call_fn_5667�)*J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
(__inference_dropout_1_layer_call_fn_6233a<�9
2�/
)�&
inputs����������
p 
� "!������������
(__inference_dropout_1_layer_call_fn_6228a<�9
2�/
)�&
inputs����������
p
� "!�����������