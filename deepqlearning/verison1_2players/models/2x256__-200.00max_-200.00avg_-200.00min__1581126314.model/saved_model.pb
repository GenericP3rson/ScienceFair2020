М╣
л¤
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
dtypetypeИ
╛
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8ЕЛ

conv2d/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:А*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*'
_output_shapes
:А
o
conv2d/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:А*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes	
:А
Д
conv2d_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:АА* 
shared_nameconv2d_1/kernel
}
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*(
_output_shapes
:АА
s
conv2d_1/biasVarHandleOp*
shared_nameconv2d_1/bias*
dtype0*
_output_shapes
: *
shape:А
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes	
:А
u
dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	А@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	А@
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
countVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 

NoOpNoOp
╞)
ConstConst"/device:CPU:0*Б)
valueў(BЇ( Bэ(
ў
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
 	keras_api
R
!	variables
"trainable_variables
#regularization_losses
$	keras_api
R
%	variables
&trainable_variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
R
/	variables
0trainable_variables
1regularization_losses
2	keras_api
R
3	variables
4trainable_variables
5regularization_losses
6	keras_api
R
7	variables
8trainable_variables
9regularization_losses
:	keras_api
R
;	variables
<trainable_variables
=regularization_losses
>	keras_api
h

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
h

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
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
 
Ъ
	variables
Kmetrics

Llayers
Mnon_trainable_variables
trainable_variables
regularization_losses
Nlayer_regularization_losses
 
 
 
 
Ъ
Ometrics
	variables

Players
Qnon_trainable_variables
trainable_variables
regularization_losses
Rlayer_regularization_losses
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Ъ
Smetrics
	variables

Tlayers
Unon_trainable_variables
trainable_variables
regularization_losses
Vlayer_regularization_losses
 
 
 
Ъ
Wmetrics
	variables

Xlayers
Ynon_trainable_variables
trainable_variables
regularization_losses
Zlayer_regularization_losses
 
 
 
Ъ
[metrics
!	variables

\layers
]non_trainable_variables
"trainable_variables
#regularization_losses
^layer_regularization_losses
 
 
 
Ъ
_metrics
%	variables

`layers
anon_trainable_variables
&trainable_variables
'regularization_losses
blayer_regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
Ъ
cmetrics
+	variables

dlayers
enon_trainable_variables
,trainable_variables
-regularization_losses
flayer_regularization_losses
 
 
 
Ъ
gmetrics
/	variables

hlayers
inon_trainable_variables
0trainable_variables
1regularization_losses
jlayer_regularization_losses
 
 
 
Ъ
kmetrics
3	variables

llayers
mnon_trainable_variables
4trainable_variables
5regularization_losses
nlayer_regularization_losses
 
 
 
Ъ
ometrics
7	variables

players
qnon_trainable_variables
8trainable_variables
9regularization_losses
rlayer_regularization_losses
 
 
 
Ъ
smetrics
;	variables

tlayers
unon_trainable_variables
<trainable_variables
=regularization_losses
vlayer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

?0
@1
 
Ъ
wmetrics
A	variables

xlayers
ynon_trainable_variables
Btrainable_variables
Cregularization_losses
zlayer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1

E0
F1
 
Ъ
{metrics
G	variables

|layers
}non_trainable_variables
Htrainable_variables
Iregularization_losses
~layer_regularization_losses

0
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
 


Аtotal

Бcount
В
_fn_kwargs
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

А0
Б1
 
 
б
Зmetrics
Г	variables
Иlayers
Йnon_trainable_variables
Дtrainable_variables
Еregularization_losses
 Кlayer_regularization_losses
 
 

А0
Б1
 *
dtype0*
_output_shapes
: 
П
serving_default_conv2d_inputPlaceholder*$
shape:         

*
dtype0*/
_output_shapes
:         


Ч
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*+
_gradient_op_typePartitionedCall-1281**
f%R#
!__inference_signature_wrapper_992*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:         	
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
╓
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*+
_gradient_op_typePartitionedCall-1313*&
f!R
__inference__traced_save_1312*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*
_output_shapes
: 
Й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biastotalcount**
config_proto

CPU

GPU 2J 8*
_output_shapes
: *
Tin
2*+
_gradient_op_typePartitionedCall-1356*)
f$R"
 __inference__traced_restore_1355*
Tout
2Л├
╩
_
&__inference_dropout_layer_call_fn_1162

inputs
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputs**
_gradient_op_typePartitionedCall-722*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_711*
Tout
2**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:         А*
Tin
2Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╢

В
)__inference_sequential_layer_call_fn_1122

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8**
_gradient_op_typePartitionedCall-966*L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_965*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:         	В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         	*
T0"
identityIdentity:output:0*N
_input_shapes=
;:         

::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: : : : : 
а
G
+__inference_max_pooling2d_layer_call_fn_622

inputs
identity╝
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-619*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_613*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*J
_output_shapes8
6:4                                    Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
╨
G
+__inference_activation_1_layer_call_fn_1177

inputs
identityб
PartitionedCallPartitionedCallinputs*0
_output_shapes
:         А*
Tin
2**
_gradient_op_typePartitionedCall-749*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_743*
Tout
2**
config_proto

CPU

GPU 2J 8i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:         А*
T0"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Ъ)
╩
C__inference_sequential_layer_call_and_return_conditional_losses_900
conv2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallИ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_594*
Tout
2**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:         А*
Tin
2**
_gradient_op_typePartitionedCall-600╦
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*0
_output_shapes
:         А*
Tin
2**
_gradient_op_typePartitionedCall-683*L
fGRE
C__inference_activation_layer_call_and_return_conditional_losses_677*
Tout
2**
config_proto

CPU

GPU 2J 8═
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0**
_gradient_op_typePartitionedCall-619*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_613*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А─
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:         А*
Tin
2**
_gradient_op_typePartitionedCall-730*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_718д
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-641*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_635*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А╤
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-749*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_743*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А╙
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0**
_gradient_op_typePartitionedCall-660*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_654*
Tout
2**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:         А*
Tin
2╩
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0**
_gradient_op_typePartitionedCall-796*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_784*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А╕
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*(
_output_shapes
:         А**
_gradient_op_typePartitionedCall-813*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_807*
Tout
2**
config_proto

CPU

GPU 2J 8П
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_830*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         @**
_gradient_op_typePartitionedCall-836Э
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         	*
Tin
2**
_gradient_op_typePartitionedCall-863*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_857Ў
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         	"
identityIdentity:output:0*N
_input_shapes=
;:         

::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall: : : :, (
&
_user_specified_nameconv2d_input: : : : : 
П
_
C__inference_activation_layer_call_and_return_conditional_losses_677

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:         Аc
IdentityIdentityRelu:activations:0*0
_output_shapes
:         А*
T0"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
∙
╪
?__inference_dense_layer_call_and_return_conditional_losses_1233

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpг
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         @*
T0"
identityIdentity:output:0*/
_input_shapes
:         А::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Ы
^
@__inference_dropout_layer_call_and_return_conditional_losses_718

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:         А*
T0d

Identity_1IdentityIdentity:output:0*0
_output_shapes
:         А*
T0"!

identity_1Identity_1:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Р
`
D__inference_activation_layer_call_and_return_conditional_losses_1127

inputs
identityO
ReluReluinputs*0
_output_shapes
:         А*
T0c
IdentityIdentityRelu:activations:0*0
_output_shapes
:         А*
T0"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
°
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_1197

inputs
identityИQ
dropout/rateConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Х
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:         АМ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0л
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:         АЭ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:         АR
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Т
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:         Аj
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:         Аx
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:         Аr
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         Аb
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╦
е
$__inference_dense_layer_call_fn_1240

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         @**
_gradient_op_typePartitionedCall-836*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_830*
Tout
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
╞
B
&__inference_dropout_layer_call_fn_1167

inputs
identityЬ
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-730*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_718i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Д,
Р
C__inference_sequential_layer_call_and_return_conditional_losses_875
conv2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallИ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-600*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_594╦
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*0
_output_shapes
:         А*
Tin
2**
_gradient_op_typePartitionedCall-683*L
fGRE
C__inference_activation_layer_call_and_return_conditional_losses_677*
Tout
2**
config_proto

CPU

GPU 2J 8═
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0**
_gradient_op_typePartitionedCall-619*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_613*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А╘
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-722*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_711м
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-641*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_635*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А╤
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-749*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_743*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А╙
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0**
_gradient_op_typePartitionedCall-660*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_654*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А№
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_777*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-788└
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-813*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_807*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         АП
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         @*
Tin
2**
_gradient_op_typePartitionedCall-836*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_830*
Tout
2Э
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-863*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_857*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         	╝
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         	"
identityIdentity:output:0*N
_input_shapes=
;:         

::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : 
Є+
К
C__inference_sequential_layer_call_and_return_conditional_losses_926

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallВ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-600*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_594*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А╦
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-683*L
fGRE
C__inference_activation_layer_call_and_return_conditional_losses_677*
Tout
2═
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-619*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_613*
Tout
2╘
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_711*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-722м
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_635*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-641╤
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-749*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_743*
Tout
2**
config_proto

CPU

GPU 2J 8╙
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-660*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_654*
Tout
2№
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_777*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-788└
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_807*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         А**
_gradient_op_typePartitionedCall-813П
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-836*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_830*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         @*
Tin
2Э
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         	**
_gradient_op_typePartitionedCall-863*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_857*
Tout
2╝
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         	"
identityIdentity:output:0*N
_input_shapes=
;:         

::::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall: : : : : : : :& "
 
_user_specified_nameinputs: 
╢

В
)__inference_sequential_layer_call_fn_1109

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8**
_gradient_op_typePartitionedCall-927*L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_926*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:         	В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	"
identityIdentity:output:0*N
_input_shapes=
;:         

::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
╫K
├
D__inference_sequential_layer_call_and_return_conditional_losses_1060

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOp╣
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:Ай
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:         Ап
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:АУ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:         А*
T0k
activation/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:         Ан
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*
ksize
*
paddingVALID*0
_output_shapes
:         А*
strides
Y
dropout/dropout/rateConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: c
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:g
"dropout/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    g
"dropout/dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: е
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:         Ад
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ├
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:         А╡
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:         АZ
dropout/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: ^
dropout/dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: А
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
_output_shapes
: *
T0к
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*0
_output_shapes
:         АТ
dropout/dropout/mulMulmax_pooling2d/MaxPool:output:0dropout/dropout/truediv:z:0*
T0*0
_output_shapes
:         АИ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:         АК
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*0
_output_shapes
:         А*
T0╛
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:АА└
conv2d_1/Conv2DConv2Ddropout/dropout/mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
paddingVALID*0
_output_shapes
:         А*
T0*
strides
│
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:АЩ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аo
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:         А▒
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:         А[
dropout_1/dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *═╠L>g
dropout_1/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
_output_shapes
:*
T0i
$dropout_1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_1/dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: й
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
dtype0*0
_output_shapes
:         А*
T0к
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0╔
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:         А╗
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:         А\
dropout_1/dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: А
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_1/dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?Ж
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: ░
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*0
_output_shapes
:         АШ
dropout_1/dropout/mulMul max_pooling2d_1/MaxPool:output:0dropout_1/dropout/truediv:z:0*
T0*0
_output_shapes
:         АМ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:         АР
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:         Аf
flatten/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:К
flatten/ReshapeReshapedropout_1/dropout/mul_1:z:0flatten/Reshape/shape:output:0*(
_output_shapes
:         А*
T0п
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А@З
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0м
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @▓
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@	Й
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	░
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         	*
T0р
IdentityIdentitydense_1/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*'
_output_shapes
:         	*
T0"
identityIdentity:output:0*N
_input_shapes=
;:         

::::::::2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp: : : : : : : :& "
 
_user_specified_nameinputs: 
°
\
@__inference_flatten_layer_call_and_return_conditional_losses_807

inputs
identity^
Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╩
D
(__inference_dropout_1_layer_call_fn_1212

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-796*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_784i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:         А*
T0"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Ъ
d
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_654

inputs
identityв
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4                                    {
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4                                    *
T0"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
Ш
b
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_613

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
strides
*
ksize
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
┤ 
╕
__inference__traced_save_1312
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

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_8e9bd4a55fa74e07bfa1326f5af98eca/part*
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Р
SaveV2/tensor_namesConst"/device:CPU:0*╣
valueпBм
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
Б
SaveV2/shape_and_slicesConst"/device:CPU:0*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:
╗
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2
h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
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
:├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:Ц
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*q
_input_shapes`
^: :А:А:АА:А:	А@:@:@	:	: : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : 
а
з
&__inference_conv2d_1_layer_call_fn_646

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-641*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_635*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*B
_output_shapes0
.:,                           АЭ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А"
identityIdentity:output:0*I
_input_shapes8
6:,                           А::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
╟

З
(__inference_sequential_layer_call_fn_938
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИвStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8**
_gradient_op_typePartitionedCall-927*L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_926*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:         	В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         	*
T0"
identityIdentity:output:0*N
_input_shapes=
;:         

::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :, (
&
_user_specified_nameconv2d_input: : : 
Э
`
B__inference_dropout_1_layer_call_and_return_conditional_losses_784

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:         Аd

Identity_1IdentityIdentity:output:0*0
_output_shapes
:         А*
T0"!

identity_1Identity_1:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Ў
`
A__inference_dropout_layer_call_and_return_conditional_losses_1152

inputs
identityИQ
dropout/rateConst*
valueB
 *═╠L>*
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
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  А?Х
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*0
_output_shapes
:         А*
T0М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: л
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*0
_output_shapes
:         А*
T0Э
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:         АR
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Т
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:         Аj
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:         Аx
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:         Аr
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         Аb
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Т
b
F__inference_activation_1_layer_call_and_return_conditional_losses_1172

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:         Аc
IdentityIdentityRelu:activations:0*0
_output_shapes
:         А*
T0"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╟

З
(__inference_sequential_layer_call_fn_977
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИвStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8**
_gradient_op_typePartitionedCall-966*L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_965*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*'
_output_shapes
:         	В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	"
identityIdentity:output:0*N
_input_shapes=
;:         

::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : 
╠
E
)__inference_activation_layer_call_fn_1132

inputs
identityЯ
PartitionedCallPartitionedCallinputs**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-683*L
fGRE
C__inference_activation_layer_call_and_return_conditional_losses_677*
Tout
2i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
С
a
E__inference_activation_1_layer_call_and_return_conditional_losses_743

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:         Аc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
д
I
-__inference_max_pooling2d_1_layer_call_fn_663

inputs
identity╛
PartitionedCallPartitionedCallinputs**
_gradient_op_typePartitionedCall-660*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_654*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*J
_output_shapes8
6:4                                    Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
╬
a
(__inference_dropout_1_layer_call_fn_1207

inputs
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputs**
_gradient_op_typePartitionedCall-788*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_777*
Tout
2**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:         А*
Tin
2Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╢
B
&__inference_flatten_layer_call_fn_1223

inputs
identityФ
PartitionedCallPartitionedCallinputs**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:         А*
Tin
2**
_gradient_op_typePartitionedCall-813*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_807*
Tout
2a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
о3
╙
__inference__wrapped_model_581
conv2d_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identityИв(sequential/conv2d/BiasAdd/ReadVariableOpв'sequential/conv2d/Conv2D/ReadVariableOpв*sequential/conv2d_1/BiasAdd/ReadVariableOpв)sequential/conv2d_1/Conv2D/ReadVariableOpв'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв)sequential/dense_1/BiasAdd/ReadVariableOpв(sequential/dense_1/MatMul/ReadVariableOp╧
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:А┼
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*0
_output_shapes
:         А*
T0┼
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:А┤
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АБ
sequential/activation/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:         А├
 sequential/max_pooling2d/MaxPoolMaxPool(sequential/activation/Relu:activations:0*
ksize
*
paddingVALID*0
_output_shapes
:         А*
strides
Н
sequential/dropout/IdentityIdentity)sequential/max_pooling2d/MaxPool:output:0*
T0*0
_output_shapes
:         А╘
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:ААс
sequential/conv2d_1/Conv2DConv2D$sequential/dropout/Identity:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
paddingVALID*0
_output_shapes
:         А*
T0*
strides
╔
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:А║
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЕ
sequential/activation_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:         А╟
"sequential/max_pooling2d_1/MaxPoolMaxPool*sequential/activation_1/Relu:activations:0*0
_output_shapes
:         А*
strides
*
ksize
*
paddingVALIDС
sequential/dropout_1/IdentityIdentity+sequential/max_pooling2d_1/MaxPool:output:0*
T0*0
_output_shapes
:         Аq
 sequential/flatten/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:л
sequential/flatten/ReshapeReshape&sequential/dropout_1/Identity:output:0)sequential/flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:         А┼
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А@и
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @┬
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@й
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╚
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@	к
sequential/dense_1/MatMulMatMul!sequential/dense/BiasAdd:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	╞
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	п
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	├
IdentityIdentity#sequential/dense_1/BiasAdd:output:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*'
_output_shapes
:         	*
T0"
identityIdentity:output:0*N
_input_shapes=
;:         

::::::::2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp: : : :, (
&
_user_specified_nameconv2d_input: : : : : 
Ю+
├
D__inference_sequential_layer_call_and_return_conditional_losses_1096

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOp╣
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:Ай
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*0
_output_shapes
:         А*
T0п
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:АУ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аk
activation/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:         Ан
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:         Аw
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*0
_output_shapes
:         А╛
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:АА└
conv2d_1/Conv2DConv2Ddropout/Identity:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:         А│
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:АЩ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:         А*
T0o
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:         А▒
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*0
_output_shapes
:         А*
strides
*
ksize
*
paddingVALID{
dropout_1/IdentityIdentity max_pooling2d_1/MaxPool:output:0*0
_output_shapes
:         А*
T0f
flatten/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:К
flatten/ReshapeReshapedropout_1/Identity:output:0flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:         Ап
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А@З
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0м
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @▓
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@	Й
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	░
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         	*
T0р
IdentityIdentitydense_1/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*'
_output_shapes
:         	*
T0"
identityIdentity:output:0*N
_input_shapes=
;:         

::::::::2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp: : : : : : : :& "
 
_user_specified_nameinputs: 
Ъ

┌
A__inference_conv2d_1_layer_call_and_return_conditional_losses_635

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpм
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*(
_output_shapes
:ААн
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*B
_output_shapes0
.:,                           Аб
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           Ад
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*B
_output_shapes0
.:,                           А*
T0"
identityIdentity:output:0*I
_input_shapes8
6:,                           А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Ы

А
!__inference_signature_wrapper_992
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityИвStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         	*
Tin
2	**
_gradient_op_typePartitionedCall-981*'
f"R 
__inference__wrapped_model_581В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         	*
T0"
identityIdentity:output:0*N
_input_shapes=
;:         

::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : 
°
╫
>__inference_dense_layer_call_and_return_conditional_losses_830

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpг
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
И)
─
C__inference_sequential_layer_call_and_return_conditional_losses_965

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallВ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-600*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_594*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А╦
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:         А*
Tin
2**
_gradient_op_typePartitionedCall-683*L
fGRE
C__inference_activation_layer_call_and_return_conditional_losses_677*
Tout
2═
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-619*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_613*
Tout
2**
config_proto

CPU

GPU 2J 8─
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0**
_gradient_op_typePartitionedCall-730*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_718*
Tout
2**
config_proto

CPU

GPU 2J 8*0
_output_shapes
:         А*
Tin
2д
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-641*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_635*
Tout
2╤
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0**
_gradient_op_typePartitionedCall-749*N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_743*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А╙
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-660*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_654*
Tout
2**
config_proto

CPU

GPU 2J 8╩
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_784*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:         А**
_gradient_op_typePartitionedCall-796╕
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_807*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         А**
_gradient_op_typePartitionedCall-813П
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:         @**
_gradient_op_typePartitionedCall-836*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_830*
Tout
2**
config_proto

CPU

GPU 2J 8Э
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         	*
Tin
2**
_gradient_op_typePartitionedCall-863*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_857Ў
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         	"
identityIdentity:output:0*N
_input_shapes=
;:         

::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
ї
_
@__inference_dropout_layer_call_and_return_conditional_losses_711

inputs
identityИQ
dropout/rateConst*
valueB
 *═╠L>*
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
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  А?Х
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:         АМ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: л
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:         АЭ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:         АR
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Т
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*0
_output_shapes
:         А*
T0j
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:         Аx
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:         Аr
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*0
_output_shapes
:         А*
T0b
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
∙
]
A__inference_flatten_layer_call_and_return_conditional_losses_1218

inputs
identity^
Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:         А*
T0Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Ы
е
$__inference_conv2d_layer_call_fn_605

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
_gradient_op_typePartitionedCall-600*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_594*
Tout
2**
config_proto

CPU

GPU 2J 8*B
_output_shapes0
.:,                           А*
Tin
2Э
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
°
┘
@__inference_dense_1_layer_call_and_return_conditional_losses_857

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@	i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         	"
identityIdentity:output:0*.
_input_shapes
:         @::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
╬
з
&__inference_dense_1_layer_call_fn_1257

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         	**
_gradient_op_typePartitionedCall-863*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_857В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         	"
identityIdentity:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
ў
a
B__inference_dropout_1_layer_call_and_return_conditional_losses_777

inputs
identityИQ
dropout/rateConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Х
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:         АМ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: л
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:         АЭ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:         АR
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0Т
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:         Аj
dropout/mulMulinputsdropout/truediv:z:0*0
_output_shapes
:         А*
T0x
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:         Аr
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         Аb
IdentityIdentitydropout/mul_1:z:0*0
_output_shapes
:         А*
T0"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Я+
Ч
 __inference__traced_restore_1355
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
identity_11ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1У
RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:
*╣
valueпBм
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEД
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:
*'
valueB
B B B B B B B B B B ╨
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes
2
*<
_output_shapes*
(::::::::::L
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
:В
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:А
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
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

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0Б
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

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:x
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0*
dtype0*
_output_shapes
 М
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
:╡
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 л
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0╕
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
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 
Ц

╪
?__inference_conv2d_layer_call_and_return_conditional_losses_594

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpл
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*'
_output_shapes
:Ан
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*B
_output_shapes0
.:,                           Аб
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:АР
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           Ад
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,                           А"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Ь
_
A__inference_dropout_layer_call_and_return_conditional_losses_1157

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:         Аd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         А"!

identity_1Identity_1:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Ю
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_1202

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:         Аd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         А"!

identity_1Identity_1:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
∙
┌
A__inference_dense_1_layer_call_and_return_conditional_losses_1250

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@	i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         	а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         	*
T0Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         	*
T0"
identityIdentity:output:0*.
_input_shapes
:         @::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*╝
serving_defaultи
M
conv2d_input=
serving_default_conv2d_input:0         

;
dense_10
StatefulPartitionedCall:0         	tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:уж
Г:
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
Л_default_save_signature
+М&call_and_return_all_conditional_losses
Н__call__"п6
_tf_keras_sequentialР6{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 10, 10, 3], "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 10, 10, 3], "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
╜
	variables
trainable_variables
regularization_losses
	keras_api
+О&call_and_return_all_conditional_losses
П__call__"м
_tf_keras_layerТ{"class_name": "InputLayer", "name": "conv2d_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 10, 10, 3], "config": {"batch_input_shape": [null, 10, 10, 3], "dtype": "float32", "sparse": false, "name": "conv2d_input"}}
в

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+Р&call_and_return_all_conditional_losses
С__call__"√
_tf_keras_layerс{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 10, 10, 3], "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 10, 10, 3], "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
Э
	variables
trainable_variables
regularization_losses
 	keras_api
+Т&call_and_return_all_conditional_losses
У__call__"М
_tf_keras_layerЄ{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
√
!	variables
"trainable_variables
#regularization_losses
$	keras_api
+Ф&call_and_return_all_conditional_losses
Х__call__"ъ
_tf_keras_layer╨{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
н
%	variables
&trainable_variables
'regularization_losses
(	keras_api
+Ц&call_and_return_all_conditional_losses
Ч__call__"Ь
_tf_keras_layerВ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
є

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
+Ш&call_and_return_all_conditional_losses
Щ__call__"╠
_tf_keras_layer▓{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}}
б
/	variables
0trainable_variables
1regularization_losses
2	keras_api
+Ъ&call_and_return_all_conditional_losses
Ы__call__"Р
_tf_keras_layerЎ{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
 
3	variables
4trainable_variables
5regularization_losses
6	keras_api
+Ь&call_and_return_all_conditional_losses
Э__call__"ю
_tf_keras_layer╘{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
▒
7	variables
8trainable_variables
9regularization_losses
:	keras_api
+Ю&call_and_return_all_conditional_losses
Я__call__"а
_tf_keras_layerЖ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
о
;	variables
<trainable_variables
=regularization_losses
>	keras_api
+а&call_and_return_all_conditional_losses
б__call__"Э
_tf_keras_layerГ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Є

?kernel
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
+в&call_and_return_all_conditional_losses
г__call__"╦
_tf_keras_layer▒{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
Ї

Ekernel
Fbias
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
+д&call_and_return_all_conditional_losses
е__call__"═
_tf_keras_layer│{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
"
	optimizer
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
 "
trackable_list_wrapper
╗
	variables
Kmetrics

Llayers
Mnon_trainable_variables
trainable_variables
regularization_losses
Nlayer_regularization_losses
Н__call__
Л_default_save_signature
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
-
жserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
Ometrics
	variables

Players
Qnon_trainable_variables
trainable_variables
regularization_losses
Rlayer_regularization_losses
П__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
(:&А2conv2d/kernel
:А2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
Smetrics
	variables

Tlayers
Unon_trainable_variables
trainable_variables
regularization_losses
Vlayer_regularization_losses
С__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
Wmetrics
	variables

Xlayers
Ynon_trainable_variables
trainable_variables
regularization_losses
Zlayer_regularization_losses
У__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
[metrics
!	variables

\layers
]non_trainable_variables
"trainable_variables
#regularization_losses
^layer_regularization_losses
Х__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
_metrics
%	variables

`layers
anon_trainable_variables
&trainable_variables
'regularization_losses
blayer_regularization_losses
Ч__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
+:)АА2conv2d_1/kernel
:А2conv2d_1/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
cmetrics
+	variables

dlayers
enon_trainable_variables
,trainable_variables
-regularization_losses
flayer_regularization_losses
Щ__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
gmetrics
/	variables

hlayers
inon_trainable_variables
0trainable_variables
1regularization_losses
jlayer_regularization_losses
Ы__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
kmetrics
3	variables

llayers
mnon_trainable_variables
4trainable_variables
5regularization_losses
nlayer_regularization_losses
Э__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
ometrics
7	variables

players
qnon_trainable_variables
8trainable_variables
9regularization_losses
rlayer_regularization_losses
Я__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
smetrics
;	variables

tlayers
unon_trainable_variables
<trainable_variables
=regularization_losses
vlayer_regularization_losses
б__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
:	А@2dense/kernel
:@2
dense/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
wmetrics
A	variables

xlayers
ynon_trainable_variables
Btrainable_variables
Cregularization_losses
zlayer_regularization_losses
г__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
 :@	2dense_1/kernel
:	2dense_1/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
Э
{metrics
G	variables

|layers
}non_trainable_variables
Htrainable_variables
Iregularization_losses
~layer_regularization_losses
е__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
 "
trackable_list_wrapper
г

Аtotal

Бcount
В
_fn_kwargs
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
+з&call_and_return_all_conditional_losses
и__call__"х
_tf_keras_layer╦{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
А0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
Зmetrics
Г	variables
Иlayers
Йnon_trainable_variables
Дtrainable_variables
Еregularization_losses
 Кlayer_regularization_losses
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
щ2ц
__inference__wrapped_model_581├
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+
conv2d_input         


▄2┘
D__inference_sequential_layer_call_and_return_conditional_losses_1096
C__inference_sequential_layer_call_and_return_conditional_losses_900
D__inference_sequential_layer_call_and_return_conditional_losses_1060
C__inference_sequential_layer_call_and_return_conditional_losses_875└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ё2э
)__inference_sequential_layer_call_fn_1109
(__inference_sequential_layer_call_fn_938
)__inference_sequential_layer_call_fn_1122
(__inference_sequential_layer_call_fn_977└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
Ю2Ы
?__inference_conv2d_layer_call_and_return_conditional_losses_594╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Г2А
$__inference_conv2d_layer_call_fn_605╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
ю2ы
D__inference_activation_layer_call_and_return_conditional_losses_1127в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_activation_layer_call_fn_1132в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
о2л
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_613р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
У2Р
+__inference_max_pooling2d_layer_call_fn_622р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
└2╜
A__inference_dropout_layer_call_and_return_conditional_losses_1152
A__inference_dropout_layer_call_and_return_conditional_losses_1157┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
К2З
&__inference_dropout_layer_call_fn_1162
&__inference_dropout_layer_call_fn_1167┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
б2Ю
A__inference_conv2d_1_layer_call_and_return_conditional_losses_635╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Ж2Г
&__inference_conv2d_1_layer_call_fn_646╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
Ё2э
F__inference_activation_1_layer_call_and_return_conditional_losses_1172в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_1_layer_call_fn_1177в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
░2н
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_654р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Х2Т
-__inference_max_pooling2d_1_layer_call_fn_663р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
─2┴
C__inference_dropout_1_layer_call_and_return_conditional_losses_1197
C__inference_dropout_1_layer_call_and_return_conditional_losses_1202┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
О2Л
(__inference_dropout_1_layer_call_fn_1207
(__inference_dropout_1_layer_call_fn_1212┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ы2ш
A__inference_flatten_layer_call_and_return_conditional_losses_1218в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_flatten_layer_call_fn_1223в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щ2ц
?__inference_dense_layer_call_and_return_conditional_losses_1233в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬2╦
$__inference_dense_layer_call_fn_1240в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_dense_1_layer_call_and_return_conditional_losses_1250в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_dense_1_layer_call_fn_1257в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
5B3
!__inference_signature_wrapper_992conv2d_input
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 ╒
?__inference_conv2d_layer_call_and_return_conditional_losses_594СIвF
?в<
:К7
inputs+                           
к "@в=
6К3
0,                           А
Ъ ║
D__inference_sequential_layer_call_and_return_conditional_losses_1096r)*?@EF?в<
5в2
(К%
inputs         


p 

 
к "%в"
К
0         	
Ъ Л
&__inference_dropout_layer_call_fn_1162a<в9
2в/
)К&
inputs         А
p
к "!К         А╡
C__inference_dropout_1_layer_call_and_return_conditional_losses_1197n<в9
2в/
)К&
inputs         А
p
к ".в+
$К!
0         А
Ъ ┴
+__inference_max_pooling2d_layer_call_fn_622СRвO
HвE
CК@
inputs4                                    
к ";К84                                    
&__inference_flatten_layer_call_fn_1223U8в5
.в+
)К&
inputs         А
к "К         АЧ
(__inference_sequential_layer_call_fn_938k)*?@EFEвB
;в8
.К+
conv2d_input         


p

 
к "К         	Л
&__inference_dropout_layer_call_fn_1167a<в9
2в/
)К&
inputs         А
p 
к "!К         АМ
+__inference_activation_1_layer_call_fn_1177]8в5
.в+
)К&
inputs         А
к "!К         А┤
F__inference_activation_1_layer_call_and_return_conditional_losses_1172j8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ Т
)__inference_sequential_layer_call_fn_1109e)*?@EF?в<
5в2
(К%
inputs         


p

 
к "К         	Т
)__inference_sequential_layer_call_fn_1122e)*?@EF?в<
5в2
(К%
inputs         


p 

 
к "К         	Ю
__inference__wrapped_model_581|)*?@EF=в:
3в0
.К+
conv2d_input         


к "1к.
,
dense_1!К
dense_1         	Н
(__inference_dropout_1_layer_call_fn_1212a<в9
2в/
)К&
inputs         А
p 
к "!К         АН
(__inference_dropout_1_layer_call_fn_1207a<в9
2в/
)К&
inputs         А
p
к "!К         АЧ
(__inference_sequential_layer_call_fn_977k)*?@EFEвB
;в8
.К+
conv2d_input         


p 

 
к "К         	з
A__inference_flatten_layer_call_and_return_conditional_losses_1218b8в5
.в+
)К&
inputs         А
к "&в#
К
0         А
Ъ ┐
C__inference_sequential_layer_call_and_return_conditional_losses_900x)*?@EFEвB
;в8
.К+
conv2d_input         


p 

 
к "%в"
К
0         	
Ъ а
?__inference_dense_layer_call_and_return_conditional_losses_1233]?@0в-
&в#
!К
inputs         А
к "%в"
К
0         @
Ъ щ
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_613ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ▓
D__inference_activation_layer_call_and_return_conditional_losses_1127j8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ y
&__inference_dense_1_layer_call_fn_1257OEF/в,
%в"
 К
inputs         @
к "К         	▓
!__inference_signature_wrapper_992М)*?@EFMвJ
в 
Cк@
>
conv2d_input.К+
conv2d_input         

"1к.
,
dense_1!К
dense_1         	│
A__inference_dropout_layer_call_and_return_conditional_losses_1152n<в9
2в/
)К&
inputs         А
p
к ".в+
$К!
0         А
Ъ К
)__inference_activation_layer_call_fn_1132]8в5
.в+
)К&
inputs         А
к "!К         А╡
C__inference_dropout_1_layer_call_and_return_conditional_losses_1202n<в9
2в/
)К&
inputs         А
p 
к ".в+
$К!
0         А
Ъ ║
D__inference_sequential_layer_call_and_return_conditional_losses_1060r)*?@EF?в<
5в2
(К%
inputs         


p

 
к "%в"
К
0         	
Ъ ┐
C__inference_sequential_layer_call_and_return_conditional_losses_875x)*?@EFEвB
;в8
.К+
conv2d_input         


p

 
к "%в"
К
0         	
Ъ │
A__inference_dropout_layer_call_and_return_conditional_losses_1157n<в9
2в/
)К&
inputs         А
p 
к ".в+
$К!
0         А
Ъ x
$__inference_dense_layer_call_fn_1240P?@0в-
&в#
!К
inputs         А
к "К         @├
-__inference_max_pooling2d_1_layer_call_fn_663СRвO
HвE
CК@
inputs4                                    
к ";К84                                    н
$__inference_conv2d_layer_call_fn_605ДIвF
?в<
:К7
inputs+                           
к "3К0,                           А░
&__inference_conv2d_1_layer_call_fn_646Е)*JвG
@в=
;К8
inputs,                           А
к "3К0,                           А╪
A__inference_conv2d_1_layer_call_and_return_conditional_losses_635Т)*JвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ы
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_654ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ б
A__inference_dense_1_layer_call_and_return_conditional_losses_1250\EF/в,
%в"
 К
inputs         @
к "%в"
К
0         	
Ъ 