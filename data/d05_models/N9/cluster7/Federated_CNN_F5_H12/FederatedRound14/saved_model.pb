њП
±ь
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resourceИ
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
П
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeКнout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*
2.12.0-rc12v2.12.0-rc0-46-g0d8efc960d28њВ
r
dense_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_namedense_66/bias
k
!dense_66/bias/Read/ReadVariableOpReadVariableOpdense_66/bias*
_output_shapes
:<*
dtype0
z
dense_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: <* 
shared_namedense_66/kernel
s
#dense_66/kernel/Read/ReadVariableOpReadVariableOpdense_66/kernel*
_output_shapes

: <*
dtype0
r
dense_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_65/bias
k
!dense_65/bias/Read/ReadVariableOpReadVariableOpdense_65/bias*
_output_shapes
: *
dtype0
z
dense_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_65/kernel
s
#dense_65/kernel/Read/ReadVariableOpReadVariableOpdense_65/kernel*
_output_shapes

: *
dtype0
§
&batch_normalization_31/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_31/moving_variance
Э
:batch_normalization_31/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_31/moving_variance*
_output_shapes
:*
dtype0
Ь
"batch_normalization_31/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_31/moving_mean
Х
6batch_normalization_31/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_31/moving_mean*
_output_shapes
:*
dtype0
О
batch_normalization_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_31/beta
З
/batch_normalization_31/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_31/beta*
_output_shapes
:*
dtype0
Р
batch_normalization_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_31/gamma
Й
0batch_normalization_31/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_31/gamma*
_output_shapes
:*
dtype0
t
conv1d_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_31/bias
m
"conv1d_31/bias/Read/ReadVariableOpReadVariableOpconv1d_31/bias*
_output_shapes
:*
dtype0
А
conv1d_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_31/kernel
y
$conv1d_31/kernel/Read/ReadVariableOpReadVariableOpconv1d_31/kernel*"
_output_shapes
:*
dtype0
§
&batch_normalization_30/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_30/moving_variance
Э
:batch_normalization_30/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_30/moving_variance*
_output_shapes
:*
dtype0
Ь
"batch_normalization_30/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_30/moving_mean
Х
6batch_normalization_30/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_30/moving_mean*
_output_shapes
:*
dtype0
О
batch_normalization_30/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_30/beta
З
/batch_normalization_30/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_30/beta*
_output_shapes
:*
dtype0
Р
batch_normalization_30/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_30/gamma
Й
0batch_normalization_30/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_30/gamma*
_output_shapes
:*
dtype0
t
conv1d_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_30/bias
m
"conv1d_30/bias/Read/ReadVariableOpReadVariableOpconv1d_30/bias*
_output_shapes
:*
dtype0
А
conv1d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_30/kernel
y
$conv1d_30/kernel/Read/ReadVariableOpReadVariableOpconv1d_30/kernel*"
_output_shapes
:*
dtype0
§
&batch_normalization_29/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_29/moving_variance
Э
:batch_normalization_29/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_29/moving_variance*
_output_shapes
:*
dtype0
Ь
"batch_normalization_29/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_29/moving_mean
Х
6batch_normalization_29/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_29/moving_mean*
_output_shapes
:*
dtype0
О
batch_normalization_29/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_29/beta
З
/batch_normalization_29/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_29/beta*
_output_shapes
:*
dtype0
Р
batch_normalization_29/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_29/gamma
Й
0batch_normalization_29/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_29/gamma*
_output_shapes
:*
dtype0
t
conv1d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_29/bias
m
"conv1d_29/bias/Read/ReadVariableOpReadVariableOpconv1d_29/bias*
_output_shapes
:*
dtype0
А
conv1d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_29/kernel
y
$conv1d_29/kernel/Read/ReadVariableOpReadVariableOpconv1d_29/kernel*"
_output_shapes
:*
dtype0
§
&batch_normalization_28/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_28/moving_variance
Э
:batch_normalization_28/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_28/moving_variance*
_output_shapes
:*
dtype0
Ь
"batch_normalization_28/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_28/moving_mean
Х
6batch_normalization_28/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_28/moving_mean*
_output_shapes
:*
dtype0
О
batch_normalization_28/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_28/beta
З
/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_28/beta*
_output_shapes
:*
dtype0
Р
batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_28/gamma
Й
0batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_28/gamma*
_output_shapes
:*
dtype0
t
conv1d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_28/bias
m
"conv1d_28/bias/Read/ReadVariableOpReadVariableOpconv1d_28/bias*
_output_shapes
:*
dtype0
А
conv1d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1d_28/kernel
y
$conv1d_28/kernel/Read/ReadVariableOpReadVariableOpconv1d_28/kernel*"
_output_shapes
:*
dtype0
А
serving_default_InputPlaceholder*+
_output_shapes
:€€€€€€€€€*
dtype0* 
shape:€€€€€€€€€
“
StatefulPartitionedCallStatefulPartitionedCallserving_default_Inputconv1d_28/kernelconv1d_28/bias&batch_normalization_28/moving_variancebatch_normalization_28/gamma"batch_normalization_28/moving_meanbatch_normalization_28/betaconv1d_29/kernelconv1d_29/bias&batch_normalization_29/moving_variancebatch_normalization_29/gamma"batch_normalization_29/moving_meanbatch_normalization_29/betaconv1d_30/kernelconv1d_30/bias&batch_normalization_30/moving_variancebatch_normalization_30/gamma"batch_normalization_30/moving_meanbatch_normalization_30/betaconv1d_31/kernelconv1d_31/bias&batch_normalization_31/moving_variancebatch_normalization_31/gamma"batch_normalization_31/moving_meanbatch_normalization_31/betadense_65/kerneldense_65/biasdense_66/kerneldense_66/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_signature_wrapper_51799755

NoOpNoOp
Кg
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*≈f
valueїfBЄf B±f
Й
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer-10
layer_with_weights-8
layer-11
layer-12
layer_with_weights-9
layer-13
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
О
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
»
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias
 &_jit_compiled_convolution_op*
’
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-axis
	.gamma
/beta
0moving_mean
1moving_variance*
»
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias
 :_jit_compiled_convolution_op*
’
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance*
»
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias
 N_jit_compiled_convolution_op*
’
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance*
»
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias
 b_jit_compiled_convolution_op*
’
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses
iaxis
	jgamma
kbeta
lmoving_mean
mmoving_variance*
О
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses* 
¶
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias*
®
|	variables
}trainable_variables
~regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
В_random_generator* 
Ѓ
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses
Йkernel
	Кbias*
Ф
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
П__call__
+Р&call_and_return_all_conditional_losses* 
№
$0
%1
.2
/3
04
15
86
97
B8
C9
D10
E11
L12
M13
V14
W15
X16
Y17
`18
a19
j20
k21
l22
m23
z24
{25
Й26
К27*
Ь
$0
%1
.2
/3
84
95
B6
C7
L8
M9
V10
W11
`12
a13
j14
k15
z16
{17
Й18
К19*
* 
µ
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Цtrace_0
Чtrace_1
Шtrace_2
Щtrace_3* 
:
Ъtrace_0
Ыtrace_1
Ьtrace_2
Эtrace_3* 
* 

Юserving_default* 
* 
* 
* 
Ц
Яnon_trainable_variables
†layers
°metrics
 Ґlayer_regularization_losses
£layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

§trace_0
•trace_1* 

¶trace_0
Іtrace_1* 

$0
%1*

$0
%1*
* 
Ш
®non_trainable_variables
©layers
™metrics
 Ђlayer_regularization_losses
ђlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

≠trace_0* 

Ѓtrace_0* 
`Z
VARIABLE_VALUEconv1d_28/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_28/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
.0
/1
02
13*

.0
/1*
* 
Ш
ѓnon_trainable_variables
∞layers
±metrics
 ≤layer_regularization_losses
≥layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

іtrace_0
µtrace_1* 

ґtrace_0
Јtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_28/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_28/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_28/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_28/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 
Ш
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Љlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

љtrace_0* 

Њtrace_0* 
`Z
VARIABLE_VALUEconv1d_29/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_29/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
B0
C1
D2
E3*

B0
C1*
* 
Ш
њnon_trainable_variables
јlayers
Ѕmetrics
 ¬layer_regularization_losses
√layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

ƒtrace_0
≈trace_1* 

∆trace_0
«trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_29/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_29/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_29/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_29/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

L0
M1*
* 
Ш
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

Ќtrace_0* 

ќtrace_0* 
`Z
VARIABLE_VALUEconv1d_30/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_30/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
V0
W1
X2
Y3*

V0
W1*
* 
Ш
ѕnon_trainable_variables
–layers
—metrics
 “layer_regularization_losses
”layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

‘trace_0
’trace_1* 

÷trace_0
„trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_30/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_30/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_30/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_30/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

`0
a1*

`0
a1*
* 
Ш
Ўnon_trainable_variables
ўlayers
Џmetrics
 џlayer_regularization_losses
№layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

Ёtrace_0* 

ёtrace_0* 
`Z
VARIABLE_VALUEconv1d_31/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv1d_31/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
j0
k1
l2
m3*

j0
k1*
* 
Ш
яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

дtrace_0
еtrace_1* 

жtrace_0
зtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_31/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_31/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_31/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_31/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses* 

нtrace_0* 

оtrace_0* 

z0
{1*

z0
{1*
* 
Ш
пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*

фtrace_0* 

хtrace_0* 
_Y
VARIABLE_VALUEdense_65/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_65/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Щ
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
|	variables
}trainable_variables
~regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses* 

ыtrace_0
ьtrace_1* 

эtrace_0
юtrace_1* 
* 

Й0
К1*

Й0
К1*
* 
Ю
€non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses*

Дtrace_0* 

Еtrace_0* 
_Y
VARIABLE_VALUEdense_66/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_66/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
Л	variables
Мtrainable_variables
Нregularization_losses
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses* 

Лtrace_0* 

Мtrace_0* 
<
00
11
D2
E3
X4
Y5
l6
m7*
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

00
11*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

D0
E1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

X0
Y1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

l0
m1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Э
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv1d_28/kernelconv1d_28/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv1d_29/kernelconv1d_29/biasbatch_normalization_29/gammabatch_normalization_29/beta"batch_normalization_29/moving_mean&batch_normalization_29/moving_varianceconv1d_30/kernelconv1d_30/biasbatch_normalization_30/gammabatch_normalization_30/beta"batch_normalization_30/moving_mean&batch_normalization_30/moving_varianceconv1d_31/kernelconv1d_31/biasbatch_normalization_31/gammabatch_normalization_31/beta"batch_normalization_31/moving_mean&batch_normalization_31/moving_variancedense_65/kerneldense_65/biasdense_66/kerneldense_66/biasConst*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_save_51800962
Ш
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_28/kernelconv1d_28/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv1d_29/kernelconv1d_29/biasbatch_normalization_29/gammabatch_normalization_29/beta"batch_normalization_29/moving_mean&batch_normalization_29/moving_varianceconv1d_30/kernelconv1d_30/biasbatch_normalization_30/gammabatch_normalization_30/beta"batch_normalization_30/moving_mean&batch_normalization_30/moving_varianceconv1d_31/kernelconv1d_31/biasbatch_normalization_31/gammabatch_normalization_31/beta"batch_normalization_31/moving_mean&batch_normalization_31/moving_variancedense_65/kerneldense_65/biasdense_66/kerneldense_66/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference__traced_restore_51801056•ƒ
њ
b
F__inference_lambda_7_layer_call_and_return_conditional_losses_51798934

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    э€€€    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         и
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€*

begin_mask*
end_maskb
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≠
I
-__inference_reshape_22_layer_call_fn_51800758

inputs
identityЈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_reshape_22_layer_call_and_return_conditional_losses_51799121d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€<:O K
'
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs
•
I
-__inference_dropout_15_layer_call_fn_51800717

inputs
identity≥
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_51799202`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
а
‘
9__inference_batch_normalization_30_layer_call_fn_51800504

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51798780|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
 
Ц
G__inference_conv1d_31_layer_call_and_return_conditional_losses_51799045

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:ђ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Т
≥
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51800361

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
њ
b
F__inference_lambda_7_layer_call_and_return_conditional_losses_51800256

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    э€€€    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         и
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€*

begin_mask*
end_maskb
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ
b
F__inference_lambda_7_layer_call_and_return_conditional_losses_51800248

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    э€€€    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         и
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€*

begin_mask*
end_maskb
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
и

d
H__inference_reshape_22_layer_call_and_return_conditional_losses_51799121

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :П
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€<:O K
'
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs
С
u
Y__inference_global_average_pooling1d_14_layer_call_and_return_conditional_losses_51800687

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
 
Ц
G__inference_conv1d_30_layer_call_and_return_conditional_losses_51799014

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:ђ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
 
Ц
G__inference_conv1d_31_layer_call_and_return_conditional_losses_51800596

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:ђ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
шK
џ
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799124	
input(
conv1d_28_51798953: 
conv1d_28_51798955:-
batch_normalization_28_51798958:-
batch_normalization_28_51798960:-
batch_normalization_28_51798962:-
batch_normalization_28_51798964:(
conv1d_29_51798984: 
conv1d_29_51798986:-
batch_normalization_29_51798989:-
batch_normalization_29_51798991:-
batch_normalization_29_51798993:-
batch_normalization_29_51798995:(
conv1d_30_51799015: 
conv1d_30_51799017:-
batch_normalization_30_51799020:-
batch_normalization_30_51799022:-
batch_normalization_30_51799024:-
batch_normalization_30_51799026:(
conv1d_31_51799046: 
conv1d_31_51799048:-
batch_normalization_31_51799051:-
batch_normalization_31_51799053:-
batch_normalization_31_51799055:-
batch_normalization_31_51799057:#
dense_65_51799073: 
dense_65_51799075: #
dense_66_51799103: <
dense_66_51799105:<
identityИҐ.batch_normalization_28/StatefulPartitionedCallҐ.batch_normalization_29/StatefulPartitionedCallҐ.batch_normalization_30/StatefulPartitionedCallҐ.batch_normalization_31/StatefulPartitionedCallҐ!conv1d_28/StatefulPartitionedCallҐ!conv1d_29/StatefulPartitionedCallҐ!conv1d_30/StatefulPartitionedCallҐ!conv1d_31/StatefulPartitionedCallҐ dense_65/StatefulPartitionedCallҐ dense_66/StatefulPartitionedCallҐ"dropout_15/StatefulPartitionedCallљ
lambda_7/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lambda_7_layer_call_and_return_conditional_losses_51798934Щ
!conv1d_28/StatefulPartitionedCallStatefulPartitionedCall!lambda_7/PartitionedCall:output:0conv1d_28_51798953conv1d_28_51798955*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_28_layer_call_and_return_conditional_losses_51798952Ъ
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv1d_28/StatefulPartitionedCall:output:0batch_normalization_28_51798958batch_normalization_28_51798960batch_normalization_28_51798962batch_normalization_28_51798964*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51798616ѓ
!conv1d_29/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0conv1d_29_51798984conv1d_29_51798986*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_29_layer_call_and_return_conditional_losses_51798983Ъ
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall*conv1d_29/StatefulPartitionedCall:output:0batch_normalization_29_51798989batch_normalization_29_51798991batch_normalization_29_51798993batch_normalization_29_51798995*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51798698ѓ
!conv1d_30/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0conv1d_30_51799015conv1d_30_51799017*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_30_layer_call_and_return_conditional_losses_51799014Ъ
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall*conv1d_30/StatefulPartitionedCall:output:0batch_normalization_30_51799020batch_normalization_30_51799022batch_normalization_30_51799024batch_normalization_30_51799026*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51798780ѓ
!conv1d_31/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_30/StatefulPartitionedCall:output:0conv1d_31_51799046conv1d_31_51799048*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_31_layer_call_and_return_conditional_losses_51799045Ъ
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCall*conv1d_31/StatefulPartitionedCall:output:0batch_normalization_31_51799051batch_normalization_31_51799053batch_normalization_31_51799055batch_normalization_31_51799057*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51798862С
+global_average_pooling1d_14/PartitionedCallPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *b
f]R[
Y__inference_global_average_pooling1d_14_layer_call_and_return_conditional_losses_51798916§
 dense_65/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_14/PartitionedCall:output:0dense_65_51799073dense_65_51799075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_51799072с
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_51799090Ы
 dense_66/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_66_51799103dense_66_51799105*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_51799102е
reshape_22/PartitionedCallPartitionedCall)dense_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_reshape_22_layer_call_and_return_conditional_losses_51799121v
IdentityIdentity#reshape_22/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Е
NoOpNoOp/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall/^batch_normalization_30/StatefulPartitionedCall/^batch_normalization_31/StatefulPartitionedCall"^conv1d_28/StatefulPartitionedCall"^conv1d_29/StatefulPartitionedCall"^conv1d_30/StatefulPartitionedCall"^conv1d_31/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2F
!conv1d_28/StatefulPartitionedCall!conv1d_28/StatefulPartitionedCall2F
!conv1d_29/StatefulPartitionedCall!conv1d_29/StatefulPartitionedCall2F
!conv1d_30/StatefulPartitionedCall!conv1d_30/StatefulPartitionedCall2F
!conv1d_31/StatefulPartitionedCall!conv1d_31/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall:R N
+
_output_shapes
:€€€€€€€€€

_user_specified_nameInput
Т
≥
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51800466

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ыK
№
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799288

inputs(
conv1d_28_51799218: 
conv1d_28_51799220:-
batch_normalization_28_51799223:-
batch_normalization_28_51799225:-
batch_normalization_28_51799227:-
batch_normalization_28_51799229:(
conv1d_29_51799232: 
conv1d_29_51799234:-
batch_normalization_29_51799237:-
batch_normalization_29_51799239:-
batch_normalization_29_51799241:-
batch_normalization_29_51799243:(
conv1d_30_51799246: 
conv1d_30_51799248:-
batch_normalization_30_51799251:-
batch_normalization_30_51799253:-
batch_normalization_30_51799255:-
batch_normalization_30_51799257:(
conv1d_31_51799260: 
conv1d_31_51799262:-
batch_normalization_31_51799265:-
batch_normalization_31_51799267:-
batch_normalization_31_51799269:-
batch_normalization_31_51799271:#
dense_65_51799275: 
dense_65_51799277: #
dense_66_51799281: <
dense_66_51799283:<
identityИҐ.batch_normalization_28/StatefulPartitionedCallҐ.batch_normalization_29/StatefulPartitionedCallҐ.batch_normalization_30/StatefulPartitionedCallҐ.batch_normalization_31/StatefulPartitionedCallҐ!conv1d_28/StatefulPartitionedCallҐ!conv1d_29/StatefulPartitionedCallҐ!conv1d_30/StatefulPartitionedCallҐ!conv1d_31/StatefulPartitionedCallҐ dense_65/StatefulPartitionedCallҐ dense_66/StatefulPartitionedCallҐ"dropout_15/StatefulPartitionedCallЊ
lambda_7/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lambda_7_layer_call_and_return_conditional_losses_51798934Щ
!conv1d_28/StatefulPartitionedCallStatefulPartitionedCall!lambda_7/PartitionedCall:output:0conv1d_28_51799218conv1d_28_51799220*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_28_layer_call_and_return_conditional_losses_51798952Ъ
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv1d_28/StatefulPartitionedCall:output:0batch_normalization_28_51799223batch_normalization_28_51799225batch_normalization_28_51799227batch_normalization_28_51799229*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51798616ѓ
!conv1d_29/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0conv1d_29_51799232conv1d_29_51799234*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_29_layer_call_and_return_conditional_losses_51798983Ъ
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall*conv1d_29/StatefulPartitionedCall:output:0batch_normalization_29_51799237batch_normalization_29_51799239batch_normalization_29_51799241batch_normalization_29_51799243*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51798698ѓ
!conv1d_30/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0conv1d_30_51799246conv1d_30_51799248*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_30_layer_call_and_return_conditional_losses_51799014Ъ
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall*conv1d_30/StatefulPartitionedCall:output:0batch_normalization_30_51799251batch_normalization_30_51799253batch_normalization_30_51799255batch_normalization_30_51799257*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51798780ѓ
!conv1d_31/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_30/StatefulPartitionedCall:output:0conv1d_31_51799260conv1d_31_51799262*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_31_layer_call_and_return_conditional_losses_51799045Ъ
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCall*conv1d_31/StatefulPartitionedCall:output:0batch_normalization_31_51799265batch_normalization_31_51799267batch_normalization_31_51799269batch_normalization_31_51799271*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51798862С
+global_average_pooling1d_14/PartitionedCallPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *b
f]R[
Y__inference_global_average_pooling1d_14_layer_call_and_return_conditional_losses_51798916§
 dense_65/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_14/PartitionedCall:output:0dense_65_51799275dense_65_51799277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_51799072с
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_51799090Ы
 dense_66/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_66_51799281dense_66_51799283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_51799102е
reshape_22/PartitionedCallPartitionedCall)dense_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_reshape_22_layer_call_and_return_conditional_losses_51799121v
IdentityIdentity#reshape_22/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Е
NoOpNoOp/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall/^batch_normalization_30/StatefulPartitionedCall/^batch_normalization_31/StatefulPartitionedCall"^conv1d_28/StatefulPartitionedCall"^conv1d_29/StatefulPartitionedCall"^conv1d_30/StatefulPartitionedCall"^conv1d_31/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2F
!conv1d_28/StatefulPartitionedCall!conv1d_28/StatefulPartitionedCall2F
!conv1d_29/StatefulPartitionedCall!conv1d_29/StatefulPartitionedCall2F
!conv1d_30/StatefulPartitionedCall!conv1d_30/StatefulPartitionedCall2F
!conv1d_31/StatefulPartitionedCall!conv1d_31/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
 
Ц
G__inference_conv1d_29_layer_call_and_return_conditional_losses_51798983

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:ђ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
а
‘
9__inference_batch_normalization_29_layer_call_fn_51800399

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51798698|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
А&
н
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51800656

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Г
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:Ф
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:і
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ƒ
г
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799877

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16: 

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: 

unknown_25: <

unknown_26:<
identityИҐStatefulPartitionedCallќ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799423s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Б
—
&__inference_signature_wrapper_51799755	
input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16: 

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: 

unknown_25: <

unknown_26:<
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__wrapped_model_51798581s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:€€€€€€€€€

_user_specified_nameInput
д”
Г
!__inference__traced_save_51800962
file_prefix=
'read_disablecopyonread_conv1d_28_kernel:5
'read_1_disablecopyonread_conv1d_28_bias:C
5read_2_disablecopyonread_batch_normalization_28_gamma:B
4read_3_disablecopyonread_batch_normalization_28_beta:I
;read_4_disablecopyonread_batch_normalization_28_moving_mean:M
?read_5_disablecopyonread_batch_normalization_28_moving_variance:?
)read_6_disablecopyonread_conv1d_29_kernel:5
'read_7_disablecopyonread_conv1d_29_bias:C
5read_8_disablecopyonread_batch_normalization_29_gamma:B
4read_9_disablecopyonread_batch_normalization_29_beta:J
<read_10_disablecopyonread_batch_normalization_29_moving_mean:N
@read_11_disablecopyonread_batch_normalization_29_moving_variance:@
*read_12_disablecopyonread_conv1d_30_kernel:6
(read_13_disablecopyonread_conv1d_30_bias:D
6read_14_disablecopyonread_batch_normalization_30_gamma:C
5read_15_disablecopyonread_batch_normalization_30_beta:J
<read_16_disablecopyonread_batch_normalization_30_moving_mean:N
@read_17_disablecopyonread_batch_normalization_30_moving_variance:@
*read_18_disablecopyonread_conv1d_31_kernel:6
(read_19_disablecopyonread_conv1d_31_bias:D
6read_20_disablecopyonread_batch_normalization_31_gamma:C
5read_21_disablecopyonread_batch_normalization_31_beta:J
<read_22_disablecopyonread_batch_normalization_31_moving_mean:N
@read_23_disablecopyonread_batch_normalization_31_moving_variance:;
)read_24_disablecopyonread_dense_65_kernel: 5
'read_25_disablecopyonread_dense_65_bias: ;
)read_26_disablecopyonread_dense_66_kernel: <5
'read_27_disablecopyonread_dense_66_bias:<
savev2_const
identity_57ИҐMergeV2CheckpointsҐRead/DisableCopyOnReadҐRead/ReadVariableOpҐRead_1/DisableCopyOnReadҐRead_1/ReadVariableOpҐRead_10/DisableCopyOnReadҐRead_10/ReadVariableOpҐRead_11/DisableCopyOnReadҐRead_11/ReadVariableOpҐRead_12/DisableCopyOnReadҐRead_12/ReadVariableOpҐRead_13/DisableCopyOnReadҐRead_13/ReadVariableOpҐRead_14/DisableCopyOnReadҐRead_14/ReadVariableOpҐRead_15/DisableCopyOnReadҐRead_15/ReadVariableOpҐRead_16/DisableCopyOnReadҐRead_16/ReadVariableOpҐRead_17/DisableCopyOnReadҐRead_17/ReadVariableOpҐRead_18/DisableCopyOnReadҐRead_18/ReadVariableOpҐRead_19/DisableCopyOnReadҐRead_19/ReadVariableOpҐRead_2/DisableCopyOnReadҐRead_2/ReadVariableOpҐRead_20/DisableCopyOnReadҐRead_20/ReadVariableOpҐRead_21/DisableCopyOnReadҐRead_21/ReadVariableOpҐRead_22/DisableCopyOnReadҐRead_22/ReadVariableOpҐRead_23/DisableCopyOnReadҐRead_23/ReadVariableOpҐRead_24/DisableCopyOnReadҐRead_24/ReadVariableOpҐRead_25/DisableCopyOnReadҐRead_25/ReadVariableOpҐRead_26/DisableCopyOnReadҐRead_26/ReadVariableOpҐRead_27/DisableCopyOnReadҐRead_27/ReadVariableOpҐRead_3/DisableCopyOnReadҐRead_3/ReadVariableOpҐRead_4/DisableCopyOnReadҐRead_4/ReadVariableOpҐRead_5/DisableCopyOnReadҐRead_5/ReadVariableOpҐRead_6/DisableCopyOnReadҐRead_6/ReadVariableOpҐRead_7/DisableCopyOnReadҐRead_7/ReadVariableOpҐRead_8/DisableCopyOnReadҐRead_8/ReadVariableOpҐRead_9/DisableCopyOnReadҐRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_conv1d_28_kernel"/device:CPU:0*
_output_shapes
 І
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_conv1d_28_kernel^Read/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0m
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:e

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*"
_output_shapes
:{
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_conv1d_28_bias"/device:CPU:0*
_output_shapes
 £
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_conv1d_28_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:Й
Read_2/DisableCopyOnReadDisableCopyOnRead5read_2_disablecopyonread_batch_normalization_28_gamma"/device:CPU:0*
_output_shapes
 ±
Read_2/ReadVariableOpReadVariableOp5read_2_disablecopyonread_batch_normalization_28_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:И
Read_3/DisableCopyOnReadDisableCopyOnRead4read_3_disablecopyonread_batch_normalization_28_beta"/device:CPU:0*
_output_shapes
 ∞
Read_3/ReadVariableOpReadVariableOp4read_3_disablecopyonread_batch_normalization_28_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:П
Read_4/DisableCopyOnReadDisableCopyOnRead;read_4_disablecopyonread_batch_normalization_28_moving_mean"/device:CPU:0*
_output_shapes
 Ј
Read_4/ReadVariableOpReadVariableOp;read_4_disablecopyonread_batch_normalization_28_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:У
Read_5/DisableCopyOnReadDisableCopyOnRead?read_5_disablecopyonread_batch_normalization_28_moving_variance"/device:CPU:0*
_output_shapes
 ї
Read_5/ReadVariableOpReadVariableOp?read_5_disablecopyonread_batch_normalization_28_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_6/DisableCopyOnReadDisableCopyOnRead)read_6_disablecopyonread_conv1d_29_kernel"/device:CPU:0*
_output_shapes
 ≠
Read_6/ReadVariableOpReadVariableOp)read_6_disablecopyonread_conv1d_29_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0r
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*"
_output_shapes
:{
Read_7/DisableCopyOnReadDisableCopyOnRead'read_7_disablecopyonread_conv1d_29_bias"/device:CPU:0*
_output_shapes
 £
Read_7/ReadVariableOpReadVariableOp'read_7_disablecopyonread_conv1d_29_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:Й
Read_8/DisableCopyOnReadDisableCopyOnRead5read_8_disablecopyonread_batch_normalization_29_gamma"/device:CPU:0*
_output_shapes
 ±
Read_8/ReadVariableOpReadVariableOp5read_8_disablecopyonread_batch_normalization_29_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:И
Read_9/DisableCopyOnReadDisableCopyOnRead4read_9_disablecopyonread_batch_normalization_29_beta"/device:CPU:0*
_output_shapes
 ∞
Read_9/ReadVariableOpReadVariableOp4read_9_disablecopyonread_batch_normalization_29_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:С
Read_10/DisableCopyOnReadDisableCopyOnRead<read_10_disablecopyonread_batch_normalization_29_moving_mean"/device:CPU:0*
_output_shapes
 Ї
Read_10/ReadVariableOpReadVariableOp<read_10_disablecopyonread_batch_normalization_29_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:Х
Read_11/DisableCopyOnReadDisableCopyOnRead@read_11_disablecopyonread_batch_normalization_29_moving_variance"/device:CPU:0*
_output_shapes
 Њ
Read_11/ReadVariableOpReadVariableOp@read_11_disablecopyonread_batch_normalization_29_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_12/DisableCopyOnReadDisableCopyOnRead*read_12_disablecopyonread_conv1d_30_kernel"/device:CPU:0*
_output_shapes
 ∞
Read_12/ReadVariableOpReadVariableOp*read_12_disablecopyonread_conv1d_30_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*"
_output_shapes
:}
Read_13/DisableCopyOnReadDisableCopyOnRead(read_13_disablecopyonread_conv1d_30_bias"/device:CPU:0*
_output_shapes
 ¶
Read_13/ReadVariableOpReadVariableOp(read_13_disablecopyonread_conv1d_30_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:Л
Read_14/DisableCopyOnReadDisableCopyOnRead6read_14_disablecopyonread_batch_normalization_30_gamma"/device:CPU:0*
_output_shapes
 і
Read_14/ReadVariableOpReadVariableOp6read_14_disablecopyonread_batch_normalization_30_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:К
Read_15/DisableCopyOnReadDisableCopyOnRead5read_15_disablecopyonread_batch_normalization_30_beta"/device:CPU:0*
_output_shapes
 ≥
Read_15/ReadVariableOpReadVariableOp5read_15_disablecopyonread_batch_normalization_30_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:С
Read_16/DisableCopyOnReadDisableCopyOnRead<read_16_disablecopyonread_batch_normalization_30_moving_mean"/device:CPU:0*
_output_shapes
 Ї
Read_16/ReadVariableOpReadVariableOp<read_16_disablecopyonread_batch_normalization_30_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:Х
Read_17/DisableCopyOnReadDisableCopyOnRead@read_17_disablecopyonread_batch_normalization_30_moving_variance"/device:CPU:0*
_output_shapes
 Њ
Read_17/ReadVariableOpReadVariableOp@read_17_disablecopyonread_batch_normalization_30_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_18/DisableCopyOnReadDisableCopyOnRead*read_18_disablecopyonread_conv1d_31_kernel"/device:CPU:0*
_output_shapes
 ∞
Read_18/ReadVariableOpReadVariableOp*read_18_disablecopyonread_conv1d_31_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*"
_output_shapes
:}
Read_19/DisableCopyOnReadDisableCopyOnRead(read_19_disablecopyonread_conv1d_31_bias"/device:CPU:0*
_output_shapes
 ¶
Read_19/ReadVariableOpReadVariableOp(read_19_disablecopyonread_conv1d_31_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:Л
Read_20/DisableCopyOnReadDisableCopyOnRead6read_20_disablecopyonread_batch_normalization_31_gamma"/device:CPU:0*
_output_shapes
 і
Read_20/ReadVariableOpReadVariableOp6read_20_disablecopyonread_batch_normalization_31_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:К
Read_21/DisableCopyOnReadDisableCopyOnRead5read_21_disablecopyonread_batch_normalization_31_beta"/device:CPU:0*
_output_shapes
 ≥
Read_21/ReadVariableOpReadVariableOp5read_21_disablecopyonread_batch_normalization_31_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:С
Read_22/DisableCopyOnReadDisableCopyOnRead<read_22_disablecopyonread_batch_normalization_31_moving_mean"/device:CPU:0*
_output_shapes
 Ї
Read_22/ReadVariableOpReadVariableOp<read_22_disablecopyonread_batch_normalization_31_moving_mean^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:Х
Read_23/DisableCopyOnReadDisableCopyOnRead@read_23_disablecopyonread_batch_normalization_31_moving_variance"/device:CPU:0*
_output_shapes
 Њ
Read_23/ReadVariableOpReadVariableOp@read_23_disablecopyonread_batch_normalization_31_moving_variance^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_24/DisableCopyOnReadDisableCopyOnRead)read_24_disablecopyonread_dense_65_kernel"/device:CPU:0*
_output_shapes
 Ђ
Read_24/ReadVariableOpReadVariableOp)read_24_disablecopyonread_dense_65_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes

: |
Read_25/DisableCopyOnReadDisableCopyOnRead'read_25_disablecopyonread_dense_65_bias"/device:CPU:0*
_output_shapes
 •
Read_25/ReadVariableOpReadVariableOp'read_25_disablecopyonread_dense_65_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_26/DisableCopyOnReadDisableCopyOnRead)read_26_disablecopyonread_dense_66_kernel"/device:CPU:0*
_output_shapes
 Ђ
Read_26/ReadVariableOpReadVariableOp)read_26_disablecopyonread_dense_66_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: <*
dtype0o
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: <e
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes

: <|
Read_27/DisableCopyOnReadDisableCopyOnRead'read_27_disablecopyonread_dense_66_bias"/device:CPU:0*
_output_shapes
 •
Read_27/ReadVariableOpReadVariableOp'read_27_disablecopyonread_dense_66_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:<*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:<a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:< 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*у
valueйBжB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHІ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B „
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *+
dtypes!
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_56Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_57IdentityIdentity_56:output:0^NoOp*
T0*
_output_shapes
: П
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_57Identity_57:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ж{
я
$__inference__traced_restore_51801056
file_prefix7
!assignvariableop_conv1d_28_kernel:/
!assignvariableop_1_conv1d_28_bias:=
/assignvariableop_2_batch_normalization_28_gamma:<
.assignvariableop_3_batch_normalization_28_beta:C
5assignvariableop_4_batch_normalization_28_moving_mean:G
9assignvariableop_5_batch_normalization_28_moving_variance:9
#assignvariableop_6_conv1d_29_kernel:/
!assignvariableop_7_conv1d_29_bias:=
/assignvariableop_8_batch_normalization_29_gamma:<
.assignvariableop_9_batch_normalization_29_beta:D
6assignvariableop_10_batch_normalization_29_moving_mean:H
:assignvariableop_11_batch_normalization_29_moving_variance::
$assignvariableop_12_conv1d_30_kernel:0
"assignvariableop_13_conv1d_30_bias:>
0assignvariableop_14_batch_normalization_30_gamma:=
/assignvariableop_15_batch_normalization_30_beta:D
6assignvariableop_16_batch_normalization_30_moving_mean:H
:assignvariableop_17_batch_normalization_30_moving_variance::
$assignvariableop_18_conv1d_31_kernel:0
"assignvariableop_19_conv1d_31_bias:>
0assignvariableop_20_batch_normalization_31_gamma:=
/assignvariableop_21_batch_normalization_31_beta:D
6assignvariableop_22_batch_normalization_31_moving_mean:H
:assignvariableop_23_batch_normalization_31_moving_variance:5
#assignvariableop_24_dense_65_kernel: /
!assignvariableop_25_dense_65_bias: 5
#assignvariableop_26_dense_66_kernel: </
!assignvariableop_27_dense_66_bias:<
identity_29ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ќ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*у
valueйBжB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH™
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ∞
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*И
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOpAssignVariableOp!assignvariableop_conv1d_28_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv1d_28_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_28_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_28_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_28_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:–
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_28_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv1d_29_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv1d_29_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_29_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_29_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_29_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_29_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv1d_30_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv1d_30_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_30_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_30_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_30_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_30_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv1d_31_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv1d_31_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_31_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_31_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_31_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_31_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_65_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_65_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_26AssignVariableOp#assignvariableop_26_dense_66_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_27AssignVariableOp!assignvariableop_27_dense_66_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Ј
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: §
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Т
≥
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51800676

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
 
Ц
G__inference_conv1d_28_layer_call_and_return_conditional_losses_51798952

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:ђ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
А&
н
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51800446

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Г
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:Ф
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:і
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
А&
н
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51798698

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Г
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:Ф
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:і
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
А&
н
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51798780

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Г
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:Ф
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:і
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Э

ч
F__inference_dense_65_layer_call_and_return_conditional_losses_51799072

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
 
Ц
G__inference_conv1d_30_layer_call_and_return_conditional_losses_51800491

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:ђ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
№
Э
,__inference_conv1d_28_layer_call_fn_51800265

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_28_layer_call_and_return_conditional_losses_51798952s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
и

d
H__inference_reshape_22_layer_call_and_return_conditional_losses_51800771

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :П
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€<:O K
'
_output_shapes
:€€€€€€€€€<
 
_user_specified_nameinputs
А&
н
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51798862

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Г
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:Ф
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:і
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
в
‘
9__inference_batch_normalization_29_layer_call_fn_51800412

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51798718|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Т
≥
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51798882

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
в
‘
9__inference_batch_normalization_28_layer_call_fn_51800307

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51798636|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
№
Э
,__inference_conv1d_30_layer_call_fn_51800475

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_30_layer_call_and_return_conditional_losses_51799014s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
А&
н
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51798616

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Г
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:Ф
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:і
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Т
≥
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51798800

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ЌЈ
љ
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51800085

inputsK
5conv1d_28_conv1d_expanddims_1_readvariableop_resource:7
)conv1d_28_biasadd_readvariableop_resource:L
>batch_normalization_28_assignmovingavg_readvariableop_resource:N
@batch_normalization_28_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_28_batchnorm_mul_readvariableop_resource:F
8batch_normalization_28_batchnorm_readvariableop_resource:K
5conv1d_29_conv1d_expanddims_1_readvariableop_resource:7
)conv1d_29_biasadd_readvariableop_resource:L
>batch_normalization_29_assignmovingavg_readvariableop_resource:N
@batch_normalization_29_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_29_batchnorm_mul_readvariableop_resource:F
8batch_normalization_29_batchnorm_readvariableop_resource:K
5conv1d_30_conv1d_expanddims_1_readvariableop_resource:7
)conv1d_30_biasadd_readvariableop_resource:L
>batch_normalization_30_assignmovingavg_readvariableop_resource:N
@batch_normalization_30_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_30_batchnorm_mul_readvariableop_resource:F
8batch_normalization_30_batchnorm_readvariableop_resource:K
5conv1d_31_conv1d_expanddims_1_readvariableop_resource:7
)conv1d_31_biasadd_readvariableop_resource:L
>batch_normalization_31_assignmovingavg_readvariableop_resource:N
@batch_normalization_31_assignmovingavg_1_readvariableop_resource:J
<batch_normalization_31_batchnorm_mul_readvariableop_resource:F
8batch_normalization_31_batchnorm_readvariableop_resource:9
'dense_65_matmul_readvariableop_resource: 6
(dense_65_biasadd_readvariableop_resource: 9
'dense_66_matmul_readvariableop_resource: <6
(dense_66_biasadd_readvariableop_resource:<
identityИҐ&batch_normalization_28/AssignMovingAvgҐ5batch_normalization_28/AssignMovingAvg/ReadVariableOpҐ(batch_normalization_28/AssignMovingAvg_1Ґ7batch_normalization_28/AssignMovingAvg_1/ReadVariableOpҐ/batch_normalization_28/batchnorm/ReadVariableOpҐ3batch_normalization_28/batchnorm/mul/ReadVariableOpҐ&batch_normalization_29/AssignMovingAvgҐ5batch_normalization_29/AssignMovingAvg/ReadVariableOpҐ(batch_normalization_29/AssignMovingAvg_1Ґ7batch_normalization_29/AssignMovingAvg_1/ReadVariableOpҐ/batch_normalization_29/batchnorm/ReadVariableOpҐ3batch_normalization_29/batchnorm/mul/ReadVariableOpҐ&batch_normalization_30/AssignMovingAvgҐ5batch_normalization_30/AssignMovingAvg/ReadVariableOpҐ(batch_normalization_30/AssignMovingAvg_1Ґ7batch_normalization_30/AssignMovingAvg_1/ReadVariableOpҐ/batch_normalization_30/batchnorm/ReadVariableOpҐ3batch_normalization_30/batchnorm/mul/ReadVariableOpҐ&batch_normalization_31/AssignMovingAvgҐ5batch_normalization_31/AssignMovingAvg/ReadVariableOpҐ(batch_normalization_31/AssignMovingAvg_1Ґ7batch_normalization_31/AssignMovingAvg_1/ReadVariableOpҐ/batch_normalization_31/batchnorm/ReadVariableOpҐ3batch_normalization_31/batchnorm/mul/ReadVariableOpҐ conv1d_28/BiasAdd/ReadVariableOpҐ,conv1d_28/Conv1D/ExpandDims_1/ReadVariableOpҐ conv1d_29/BiasAdd/ReadVariableOpҐ,conv1d_29/Conv1D/ExpandDims_1/ReadVariableOpҐ conv1d_30/BiasAdd/ReadVariableOpҐ,conv1d_30/Conv1D/ExpandDims_1/ReadVariableOpҐ conv1d_31/BiasAdd/ReadVariableOpҐ,conv1d_31/Conv1D/ExpandDims_1/ReadVariableOpҐdense_65/BiasAdd/ReadVariableOpҐdense_65/MatMul/ReadVariableOpҐdense_66/BiasAdd/ReadVariableOpҐdense_66/MatMul/ReadVariableOpq
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    э€€€    s
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         М
lambda_7/strided_sliceStridedSliceinputs%lambda_7/strided_slice/stack:output:0'lambda_7/strided_slice/stack_1:output:0'lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€*

begin_mask*
end_maskj
conv1d_28/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ѓ
conv1d_28/Conv1D/ExpandDims
ExpandDimslambda_7/strided_slice:output:0(conv1d_28/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€¶
,conv1d_28/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_28_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_28/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Њ
conv1d_28/Conv1D/ExpandDims_1
ExpandDims4conv1d_28/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_28/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 
conv1d_28/Conv1DConv2D$conv1d_28/Conv1D/ExpandDims:output:0&conv1d_28/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
Ф
conv1d_28/Conv1D/SqueezeSqueezeconv1d_28/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€Ж
 conv1d_28/BiasAdd/ReadVariableOpReadVariableOp)conv1d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv1d_28/BiasAddBiasAdd!conv1d_28/Conv1D/Squeeze:output:0(conv1d_28/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€h
conv1d_28/ReluReluconv1d_28/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ж
5batch_normalization_28/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       «
#batch_normalization_28/moments/meanMeanconv1d_28/Relu:activations:0>batch_normalization_28/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(Ц
+batch_normalization_28/moments/StopGradientStopGradient,batch_normalization_28/moments/mean:output:0*
T0*"
_output_shapes
:ѕ
0batch_normalization_28/moments/SquaredDifferenceSquaredDifferenceconv1d_28/Relu:activations:04batch_normalization_28/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€К
9batch_normalization_28/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       з
'batch_normalization_28/moments/varianceMean4batch_normalization_28/moments/SquaredDifference:z:0Bbatch_normalization_28/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(Ь
&batch_normalization_28/moments/SqueezeSqueeze,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Ґ
(batch_normalization_28/moments/Squeeze_1Squeeze0batch_normalization_28/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_28/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<∞
5batch_normalization_28/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_28_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0∆
*batch_normalization_28/AssignMovingAvg/subSub=batch_normalization_28/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_28/moments/Squeeze:output:0*
T0*
_output_shapes
:љ
*batch_normalization_28/AssignMovingAvg/mulMul.batch_normalization_28/AssignMovingAvg/sub:z:05batch_normalization_28/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:И
&batch_normalization_28/AssignMovingAvgAssignSubVariableOp>batch_normalization_28_assignmovingavg_readvariableop_resource.batch_normalization_28/AssignMovingAvg/mul:z:06^batch_normalization_28/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_28/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<і
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0ћ
,batch_normalization_28/AssignMovingAvg_1/subSub?batch_normalization_28/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_28/moments/Squeeze_1:output:0*
T0*
_output_shapes
:√
,batch_normalization_28/AssignMovingAvg_1/mulMul0batch_normalization_28/AssignMovingAvg_1/sub:z:07batch_normalization_28/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Р
(batch_normalization_28/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_28_assignmovingavg_1_readvariableop_resource0batch_normalization_28/AssignMovingAvg_1/mul:z:08^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:ґ
$batch_normalization_28/batchnorm/addAddV21batch_normalization_28/moments/Squeeze_1:output:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:ђ
3batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0є
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:0;batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ђ
&batch_normalization_28/batchnorm/mul_1Mulconv1d_28/Relu:activations:0(batch_normalization_28/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€≠
&batch_normalization_28/batchnorm/mul_2Mul/batch_normalization_28/moments/Squeeze:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:§
/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0µ
$batch_normalization_28/batchnorm/subSub7batch_normalization_28/batchnorm/ReadVariableOp:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ї
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€j
conv1d_29/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€є
conv1d_29/Conv1D/ExpandDims
ExpandDims*batch_normalization_28/batchnorm/add_1:z:0(conv1d_29/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€¶
,conv1d_29/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_29_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_29/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Њ
conv1d_29/Conv1D/ExpandDims_1
ExpandDims4conv1d_29/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_29/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 
conv1d_29/Conv1DConv2D$conv1d_29/Conv1D/ExpandDims:output:0&conv1d_29/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
Ф
conv1d_29/Conv1D/SqueezeSqueezeconv1d_29/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€Ж
 conv1d_29/BiasAdd/ReadVariableOpReadVariableOp)conv1d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv1d_29/BiasAddBiasAdd!conv1d_29/Conv1D/Squeeze:output:0(conv1d_29/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€h
conv1d_29/ReluReluconv1d_29/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ж
5batch_normalization_29/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       «
#batch_normalization_29/moments/meanMeanconv1d_29/Relu:activations:0>batch_normalization_29/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(Ц
+batch_normalization_29/moments/StopGradientStopGradient,batch_normalization_29/moments/mean:output:0*
T0*"
_output_shapes
:ѕ
0batch_normalization_29/moments/SquaredDifferenceSquaredDifferenceconv1d_29/Relu:activations:04batch_normalization_29/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€К
9batch_normalization_29/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       з
'batch_normalization_29/moments/varianceMean4batch_normalization_29/moments/SquaredDifference:z:0Bbatch_normalization_29/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(Ь
&batch_normalization_29/moments/SqueezeSqueeze,batch_normalization_29/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Ґ
(batch_normalization_29/moments/Squeeze_1Squeeze0batch_normalization_29/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_29/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<∞
5batch_normalization_29/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_29_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0∆
*batch_normalization_29/AssignMovingAvg/subSub=batch_normalization_29/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_29/moments/Squeeze:output:0*
T0*
_output_shapes
:љ
*batch_normalization_29/AssignMovingAvg/mulMul.batch_normalization_29/AssignMovingAvg/sub:z:05batch_normalization_29/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:И
&batch_normalization_29/AssignMovingAvgAssignSubVariableOp>batch_normalization_29_assignmovingavg_readvariableop_resource.batch_normalization_29/AssignMovingAvg/mul:z:06^batch_normalization_29/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_29/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<і
7batch_normalization_29/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_29_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0ћ
,batch_normalization_29/AssignMovingAvg_1/subSub?batch_normalization_29/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_29/moments/Squeeze_1:output:0*
T0*
_output_shapes
:√
,batch_normalization_29/AssignMovingAvg_1/mulMul0batch_normalization_29/AssignMovingAvg_1/sub:z:07batch_normalization_29/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Р
(batch_normalization_29/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_29_assignmovingavg_1_readvariableop_resource0batch_normalization_29/AssignMovingAvg_1/mul:z:08^batch_normalization_29/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_29/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:ґ
$batch_normalization_29/batchnorm/addAddV21batch_normalization_29/moments/Squeeze_1:output:0/batch_normalization_29/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_29/batchnorm/RsqrtRsqrt(batch_normalization_29/batchnorm/add:z:0*
T0*
_output_shapes
:ђ
3batch_normalization_29/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_29_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0є
$batch_normalization_29/batchnorm/mulMul*batch_normalization_29/batchnorm/Rsqrt:y:0;batch_normalization_29/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ђ
&batch_normalization_29/batchnorm/mul_1Mulconv1d_29/Relu:activations:0(batch_normalization_29/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€≠
&batch_normalization_29/batchnorm/mul_2Mul/batch_normalization_29/moments/Squeeze:output:0(batch_normalization_29/batchnorm/mul:z:0*
T0*
_output_shapes
:§
/batch_normalization_29/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_29_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0µ
$batch_normalization_29/batchnorm/subSub7batch_normalization_29/batchnorm/ReadVariableOp:value:0*batch_normalization_29/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ї
&batch_normalization_29/batchnorm/add_1AddV2*batch_normalization_29/batchnorm/mul_1:z:0(batch_normalization_29/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€j
conv1d_30/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€є
conv1d_30/Conv1D/ExpandDims
ExpandDims*batch_normalization_29/batchnorm/add_1:z:0(conv1d_30/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€¶
,conv1d_30/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_30_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_30/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Њ
conv1d_30/Conv1D/ExpandDims_1
ExpandDims4conv1d_30/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_30/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 
conv1d_30/Conv1DConv2D$conv1d_30/Conv1D/ExpandDims:output:0&conv1d_30/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
Ф
conv1d_30/Conv1D/SqueezeSqueezeconv1d_30/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€Ж
 conv1d_30/BiasAdd/ReadVariableOpReadVariableOp)conv1d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv1d_30/BiasAddBiasAdd!conv1d_30/Conv1D/Squeeze:output:0(conv1d_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€h
conv1d_30/ReluReluconv1d_30/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ж
5batch_normalization_30/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       «
#batch_normalization_30/moments/meanMeanconv1d_30/Relu:activations:0>batch_normalization_30/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(Ц
+batch_normalization_30/moments/StopGradientStopGradient,batch_normalization_30/moments/mean:output:0*
T0*"
_output_shapes
:ѕ
0batch_normalization_30/moments/SquaredDifferenceSquaredDifferenceconv1d_30/Relu:activations:04batch_normalization_30/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€К
9batch_normalization_30/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       з
'batch_normalization_30/moments/varianceMean4batch_normalization_30/moments/SquaredDifference:z:0Bbatch_normalization_30/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(Ь
&batch_normalization_30/moments/SqueezeSqueeze,batch_normalization_30/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Ґ
(batch_normalization_30/moments/Squeeze_1Squeeze0batch_normalization_30/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_30/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<∞
5batch_normalization_30/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_30_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0∆
*batch_normalization_30/AssignMovingAvg/subSub=batch_normalization_30/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_30/moments/Squeeze:output:0*
T0*
_output_shapes
:љ
*batch_normalization_30/AssignMovingAvg/mulMul.batch_normalization_30/AssignMovingAvg/sub:z:05batch_normalization_30/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:И
&batch_normalization_30/AssignMovingAvgAssignSubVariableOp>batch_normalization_30_assignmovingavg_readvariableop_resource.batch_normalization_30/AssignMovingAvg/mul:z:06^batch_normalization_30/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_30/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<і
7batch_normalization_30/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_30_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0ћ
,batch_normalization_30/AssignMovingAvg_1/subSub?batch_normalization_30/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_30/moments/Squeeze_1:output:0*
T0*
_output_shapes
:√
,batch_normalization_30/AssignMovingAvg_1/mulMul0batch_normalization_30/AssignMovingAvg_1/sub:z:07batch_normalization_30/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Р
(batch_normalization_30/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_30_assignmovingavg_1_readvariableop_resource0batch_normalization_30/AssignMovingAvg_1/mul:z:08^batch_normalization_30/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_30/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:ґ
$batch_normalization_30/batchnorm/addAddV21batch_normalization_30/moments/Squeeze_1:output:0/batch_normalization_30/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_30/batchnorm/RsqrtRsqrt(batch_normalization_30/batchnorm/add:z:0*
T0*
_output_shapes
:ђ
3batch_normalization_30/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_30_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0є
$batch_normalization_30/batchnorm/mulMul*batch_normalization_30/batchnorm/Rsqrt:y:0;batch_normalization_30/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ђ
&batch_normalization_30/batchnorm/mul_1Mulconv1d_30/Relu:activations:0(batch_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€≠
&batch_normalization_30/batchnorm/mul_2Mul/batch_normalization_30/moments/Squeeze:output:0(batch_normalization_30/batchnorm/mul:z:0*
T0*
_output_shapes
:§
/batch_normalization_30/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_30_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0µ
$batch_normalization_30/batchnorm/subSub7batch_normalization_30/batchnorm/ReadVariableOp:value:0*batch_normalization_30/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ї
&batch_normalization_30/batchnorm/add_1AddV2*batch_normalization_30/batchnorm/mul_1:z:0(batch_normalization_30/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€j
conv1d_31/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€є
conv1d_31/Conv1D/ExpandDims
ExpandDims*batch_normalization_30/batchnorm/add_1:z:0(conv1d_31/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€¶
,conv1d_31/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_31_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_31/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Њ
conv1d_31/Conv1D/ExpandDims_1
ExpandDims4conv1d_31/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_31/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 
conv1d_31/Conv1DConv2D$conv1d_31/Conv1D/ExpandDims:output:0&conv1d_31/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
Ф
conv1d_31/Conv1D/SqueezeSqueezeconv1d_31/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€Ж
 conv1d_31/BiasAdd/ReadVariableOpReadVariableOp)conv1d_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv1d_31/BiasAddBiasAdd!conv1d_31/Conv1D/Squeeze:output:0(conv1d_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€h
conv1d_31/ReluReluconv1d_31/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ж
5batch_normalization_31/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       «
#batch_normalization_31/moments/meanMeanconv1d_31/Relu:activations:0>batch_normalization_31/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(Ц
+batch_normalization_31/moments/StopGradientStopGradient,batch_normalization_31/moments/mean:output:0*
T0*"
_output_shapes
:ѕ
0batch_normalization_31/moments/SquaredDifferenceSquaredDifferenceconv1d_31/Relu:activations:04batch_normalization_31/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€К
9batch_normalization_31/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       з
'batch_normalization_31/moments/varianceMean4batch_normalization_31/moments/SquaredDifference:z:0Bbatch_normalization_31/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(Ь
&batch_normalization_31/moments/SqueezeSqueeze,batch_normalization_31/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Ґ
(batch_normalization_31/moments/Squeeze_1Squeeze0batch_normalization_31/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 q
,batch_normalization_31/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<∞
5batch_normalization_31/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_31_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0∆
*batch_normalization_31/AssignMovingAvg/subSub=batch_normalization_31/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_31/moments/Squeeze:output:0*
T0*
_output_shapes
:љ
*batch_normalization_31/AssignMovingAvg/mulMul.batch_normalization_31/AssignMovingAvg/sub:z:05batch_normalization_31/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:И
&batch_normalization_31/AssignMovingAvgAssignSubVariableOp>batch_normalization_31_assignmovingavg_readvariableop_resource.batch_normalization_31/AssignMovingAvg/mul:z:06^batch_normalization_31/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_31/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<і
7batch_normalization_31/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_31_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0ћ
,batch_normalization_31/AssignMovingAvg_1/subSub?batch_normalization_31/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_31/moments/Squeeze_1:output:0*
T0*
_output_shapes
:√
,batch_normalization_31/AssignMovingAvg_1/mulMul0batch_normalization_31/AssignMovingAvg_1/sub:z:07batch_normalization_31/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Р
(batch_normalization_31/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_31_assignmovingavg_1_readvariableop_resource0batch_normalization_31/AssignMovingAvg_1/mul:z:08^batch_normalization_31/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_31/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:ґ
$batch_normalization_31/batchnorm/addAddV21batch_normalization_31/moments/Squeeze_1:output:0/batch_normalization_31/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_31/batchnorm/RsqrtRsqrt(batch_normalization_31/batchnorm/add:z:0*
T0*
_output_shapes
:ђ
3batch_normalization_31/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_31_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0є
$batch_normalization_31/batchnorm/mulMul*batch_normalization_31/batchnorm/Rsqrt:y:0;batch_normalization_31/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ђ
&batch_normalization_31/batchnorm/mul_1Mulconv1d_31/Relu:activations:0(batch_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€≠
&batch_normalization_31/batchnorm/mul_2Mul/batch_normalization_31/moments/Squeeze:output:0(batch_normalization_31/batchnorm/mul:z:0*
T0*
_output_shapes
:§
/batch_normalization_31/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_31_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0µ
$batch_normalization_31/batchnorm/subSub7batch_normalization_31/batchnorm/ReadVariableOp:value:0*batch_normalization_31/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ї
&batch_normalization_31/batchnorm/add_1AddV2*batch_normalization_31/batchnorm/mul_1:z:0(batch_normalization_31/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€t
2global_average_pooling1d_14/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :√
 global_average_pooling1d_14/MeanMean*batch_normalization_31/batchnorm/add_1:z:0;global_average_pooling1d_14/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ю
dense_65/MatMulMatMul)global_average_pooling1d_14/Mean:output:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0С
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ b
dense_65/ReluReludense_65/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ]
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?П
dropout_15/dropout/MulMuldense_65/Relu:activations:0!dropout_15/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ q
dropout_15/dropout/ShapeShapedense_65/Relu:activations:0*
T0*
_output_shapes
::нѕЃ
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seed*f
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>«
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ _
dropout_15/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    њ
dropout_15/dropout/SelectV2SelectV2#dropout_15/dropout/GreaterEqual:z:0dropout_15/dropout/Mul:z:0#dropout_15/dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ж
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource*
_output_shapes

: <*
dtype0Щ
dense_66/MatMulMatMul$dropout_15/dropout/SelectV2:output:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€<Д
dense_66/BiasAdd/ReadVariableOpReadVariableOp(dense_66_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0С
dense_66/BiasAddBiasAdddense_66/MatMul:product:0'dense_66/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€<g
reshape_22/ShapeShapedense_66/BiasAdd:output:0*
T0*
_output_shapes
::нѕh
reshape_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
reshape_22/strided_sliceStridedSlicereshape_22/Shape:output:0'reshape_22/strided_slice/stack:output:0)reshape_22/strided_slice/stack_1:output:0)reshape_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_22/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :ї
reshape_22/Reshape/shapePack!reshape_22/strided_slice:output:0#reshape_22/Reshape/shape/1:output:0#reshape_22/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:С
reshape_22/ReshapeReshapedense_66/BiasAdd:output:0!reshape_22/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€n
IdentityIdentityreshape_22/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ћ
NoOpNoOp'^batch_normalization_28/AssignMovingAvg6^batch_normalization_28/AssignMovingAvg/ReadVariableOp)^batch_normalization_28/AssignMovingAvg_18^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_28/batchnorm/ReadVariableOp4^batch_normalization_28/batchnorm/mul/ReadVariableOp'^batch_normalization_29/AssignMovingAvg6^batch_normalization_29/AssignMovingAvg/ReadVariableOp)^batch_normalization_29/AssignMovingAvg_18^batch_normalization_29/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_29/batchnorm/ReadVariableOp4^batch_normalization_29/batchnorm/mul/ReadVariableOp'^batch_normalization_30/AssignMovingAvg6^batch_normalization_30/AssignMovingAvg/ReadVariableOp)^batch_normalization_30/AssignMovingAvg_18^batch_normalization_30/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_30/batchnorm/ReadVariableOp4^batch_normalization_30/batchnorm/mul/ReadVariableOp'^batch_normalization_31/AssignMovingAvg6^batch_normalization_31/AssignMovingAvg/ReadVariableOp)^batch_normalization_31/AssignMovingAvg_18^batch_normalization_31/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_31/batchnorm/ReadVariableOp4^batch_normalization_31/batchnorm/mul/ReadVariableOp!^conv1d_28/BiasAdd/ReadVariableOp-^conv1d_28/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_29/BiasAdd/ReadVariableOp-^conv1d_29/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_30/BiasAdd/ReadVariableOp-^conv1d_30/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_31/BiasAdd/ReadVariableOp-^conv1d_31/Conv1D/ExpandDims_1/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp ^dense_66/BiasAdd/ReadVariableOp^dense_66/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2n
5batch_normalization_28/AssignMovingAvg/ReadVariableOp5batch_normalization_28/AssignMovingAvg/ReadVariableOp2r
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp2T
(batch_normalization_28/AssignMovingAvg_1(batch_normalization_28/AssignMovingAvg_12P
&batch_normalization_28/AssignMovingAvg&batch_normalization_28/AssignMovingAvg2b
/batch_normalization_28/batchnorm/ReadVariableOp/batch_normalization_28/batchnorm/ReadVariableOp2j
3batch_normalization_28/batchnorm/mul/ReadVariableOp3batch_normalization_28/batchnorm/mul/ReadVariableOp2n
5batch_normalization_29/AssignMovingAvg/ReadVariableOp5batch_normalization_29/AssignMovingAvg/ReadVariableOp2r
7batch_normalization_29/AssignMovingAvg_1/ReadVariableOp7batch_normalization_29/AssignMovingAvg_1/ReadVariableOp2T
(batch_normalization_29/AssignMovingAvg_1(batch_normalization_29/AssignMovingAvg_12P
&batch_normalization_29/AssignMovingAvg&batch_normalization_29/AssignMovingAvg2b
/batch_normalization_29/batchnorm/ReadVariableOp/batch_normalization_29/batchnorm/ReadVariableOp2j
3batch_normalization_29/batchnorm/mul/ReadVariableOp3batch_normalization_29/batchnorm/mul/ReadVariableOp2n
5batch_normalization_30/AssignMovingAvg/ReadVariableOp5batch_normalization_30/AssignMovingAvg/ReadVariableOp2r
7batch_normalization_30/AssignMovingAvg_1/ReadVariableOp7batch_normalization_30/AssignMovingAvg_1/ReadVariableOp2T
(batch_normalization_30/AssignMovingAvg_1(batch_normalization_30/AssignMovingAvg_12P
&batch_normalization_30/AssignMovingAvg&batch_normalization_30/AssignMovingAvg2b
/batch_normalization_30/batchnorm/ReadVariableOp/batch_normalization_30/batchnorm/ReadVariableOp2j
3batch_normalization_30/batchnorm/mul/ReadVariableOp3batch_normalization_30/batchnorm/mul/ReadVariableOp2n
5batch_normalization_31/AssignMovingAvg/ReadVariableOp5batch_normalization_31/AssignMovingAvg/ReadVariableOp2r
7batch_normalization_31/AssignMovingAvg_1/ReadVariableOp7batch_normalization_31/AssignMovingAvg_1/ReadVariableOp2T
(batch_normalization_31/AssignMovingAvg_1(batch_normalization_31/AssignMovingAvg_12P
&batch_normalization_31/AssignMovingAvg&batch_normalization_31/AssignMovingAvg2b
/batch_normalization_31/batchnorm/ReadVariableOp/batch_normalization_31/batchnorm/ReadVariableOp2j
3batch_normalization_31/batchnorm/mul/ReadVariableOp3batch_normalization_31/batchnorm/mul/ReadVariableOp2D
 conv1d_28/BiasAdd/ReadVariableOp conv1d_28/BiasAdd/ReadVariableOp2\
,conv1d_28/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_28/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_29/BiasAdd/ReadVariableOp conv1d_29/BiasAdd/ReadVariableOp2\
,conv1d_29/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_29/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_30/BiasAdd/ReadVariableOp conv1d_30/BiasAdd/ReadVariableOp2\
,conv1d_30/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_30/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_31/BiasAdd/ReadVariableOp conv1d_31/BiasAdd/ReadVariableOp2\
,conv1d_31/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_31/Conv1D/ExpandDims_1/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2B
dense_66/BiasAdd/ReadVariableOpdense_66/BiasAdd/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
±
G
+__inference_lambda_7_layer_call_fn_51800235

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lambda_7_layer_call_and_return_conditional_losses_51798934d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Љ
г
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799816

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16: 

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: 

unknown_25: <

unknown_26:<
identityИҐStatefulPartitionedCall∆
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*6
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799288s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
а
‘
9__inference_batch_normalization_31_layer_call_fn_51800609

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51798862|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
шЗ
Е#
#__inference__wrapped_model_51798581	
input`
Jfederated_cnn_f5_h12_conv1d_28_conv1d_expanddims_1_readvariableop_resource:L
>federated_cnn_f5_h12_conv1d_28_biasadd_readvariableop_resource:[
Mfederated_cnn_f5_h12_batch_normalization_28_batchnorm_readvariableop_resource:_
Qfederated_cnn_f5_h12_batch_normalization_28_batchnorm_mul_readvariableop_resource:]
Ofederated_cnn_f5_h12_batch_normalization_28_batchnorm_readvariableop_1_resource:]
Ofederated_cnn_f5_h12_batch_normalization_28_batchnorm_readvariableop_2_resource:`
Jfederated_cnn_f5_h12_conv1d_29_conv1d_expanddims_1_readvariableop_resource:L
>federated_cnn_f5_h12_conv1d_29_biasadd_readvariableop_resource:[
Mfederated_cnn_f5_h12_batch_normalization_29_batchnorm_readvariableop_resource:_
Qfederated_cnn_f5_h12_batch_normalization_29_batchnorm_mul_readvariableop_resource:]
Ofederated_cnn_f5_h12_batch_normalization_29_batchnorm_readvariableop_1_resource:]
Ofederated_cnn_f5_h12_batch_normalization_29_batchnorm_readvariableop_2_resource:`
Jfederated_cnn_f5_h12_conv1d_30_conv1d_expanddims_1_readvariableop_resource:L
>federated_cnn_f5_h12_conv1d_30_biasadd_readvariableop_resource:[
Mfederated_cnn_f5_h12_batch_normalization_30_batchnorm_readvariableop_resource:_
Qfederated_cnn_f5_h12_batch_normalization_30_batchnorm_mul_readvariableop_resource:]
Ofederated_cnn_f5_h12_batch_normalization_30_batchnorm_readvariableop_1_resource:]
Ofederated_cnn_f5_h12_batch_normalization_30_batchnorm_readvariableop_2_resource:`
Jfederated_cnn_f5_h12_conv1d_31_conv1d_expanddims_1_readvariableop_resource:L
>federated_cnn_f5_h12_conv1d_31_biasadd_readvariableop_resource:[
Mfederated_cnn_f5_h12_batch_normalization_31_batchnorm_readvariableop_resource:_
Qfederated_cnn_f5_h12_batch_normalization_31_batchnorm_mul_readvariableop_resource:]
Ofederated_cnn_f5_h12_batch_normalization_31_batchnorm_readvariableop_1_resource:]
Ofederated_cnn_f5_h12_batch_normalization_31_batchnorm_readvariableop_2_resource:N
<federated_cnn_f5_h12_dense_65_matmul_readvariableop_resource: K
=federated_cnn_f5_h12_dense_65_biasadd_readvariableop_resource: N
<federated_cnn_f5_h12_dense_66_matmul_readvariableop_resource: <K
=federated_cnn_f5_h12_dense_66_biasadd_readvariableop_resource:<
identityИҐDFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOpҐFFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp_1ҐFFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp_2ҐHFederated_CNN_F5_H12/batch_normalization_28/batchnorm/mul/ReadVariableOpҐDFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOpҐFFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp_1ҐFFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp_2ҐHFederated_CNN_F5_H12/batch_normalization_29/batchnorm/mul/ReadVariableOpҐDFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOpҐFFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp_1ҐFFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp_2ҐHFederated_CNN_F5_H12/batch_normalization_30/batchnorm/mul/ReadVariableOpҐDFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOpҐFFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp_1ҐFFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp_2ҐHFederated_CNN_F5_H12/batch_normalization_31/batchnorm/mul/ReadVariableOpҐ5Federated_CNN_F5_H12/conv1d_28/BiasAdd/ReadVariableOpҐAFederated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims_1/ReadVariableOpҐ5Federated_CNN_F5_H12/conv1d_29/BiasAdd/ReadVariableOpҐAFederated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims_1/ReadVariableOpҐ5Federated_CNN_F5_H12/conv1d_30/BiasAdd/ReadVariableOpҐAFederated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims_1/ReadVariableOpҐ5Federated_CNN_F5_H12/conv1d_31/BiasAdd/ReadVariableOpҐAFederated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims_1/ReadVariableOpҐ4Federated_CNN_F5_H12/dense_65/BiasAdd/ReadVariableOpҐ3Federated_CNN_F5_H12/dense_65/MatMul/ReadVariableOpҐ4Federated_CNN_F5_H12/dense_66/BiasAdd/ReadVariableOpҐ3Federated_CNN_F5_H12/dense_66/MatMul/ReadVariableOpЖ
1Federated_CNN_F5_H12/lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    э€€€    И
3Federated_CNN_F5_H12/lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            И
3Federated_CNN_F5_H12/lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         я
+Federated_CNN_F5_H12/lambda_7/strided_sliceStridedSliceinput:Federated_CNN_F5_H12/lambda_7/strided_slice/stack:output:0<Federated_CNN_F5_H12/lambda_7/strided_slice/stack_1:output:0<Federated_CNN_F5_H12/lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask
4Federated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€н
0Federated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims
ExpandDims4Federated_CNN_F5_H12/lambda_7/strided_slice:output:0=Federated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€–
AFederated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJfederated_cnn_f5_h12_conv1d_28_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0x
6Federated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : э
2Federated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims_1
ExpandDimsIFederated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims_1/ReadVariableOp:value:0?Federated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Й
%Federated_CNN_F5_H12/conv1d_28/Conv1DConv2D9Federated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims:output:0;Federated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
Њ
-Federated_CNN_F5_H12/conv1d_28/Conv1D/SqueezeSqueeze.Federated_CNN_F5_H12/conv1d_28/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€∞
5Federated_CNN_F5_H12/conv1d_28/BiasAdd/ReadVariableOpReadVariableOp>federated_cnn_f5_h12_conv1d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ё
&Federated_CNN_F5_H12/conv1d_28/BiasAddBiasAdd6Federated_CNN_F5_H12/conv1d_28/Conv1D/Squeeze:output:0=Federated_CNN_F5_H12/conv1d_28/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Т
#Federated_CNN_F5_H12/conv1d_28/ReluRelu/Federated_CNN_F5_H12/conv1d_28/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€ќ
DFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOpMfederated_cnn_f5_h12_batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0А
;Federated_CNN_F5_H12/batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:ы
9Federated_CNN_F5_H12/batch_normalization_28/batchnorm/addAddV2LFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp:value:0DFederated_CNN_F5_H12/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:®
;Federated_CNN_F5_H12/batch_normalization_28/batchnorm/RsqrtRsqrt=Federated_CNN_F5_H12/batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:÷
HFederated_CNN_F5_H12/batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOpQfederated_cnn_f5_h12_batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0ш
9Federated_CNN_F5_H12/batch_normalization_28/batchnorm/mulMul?Federated_CNN_F5_H12/batch_normalization_28/batchnorm/Rsqrt:y:0PFederated_CNN_F5_H12/batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:к
;Federated_CNN_F5_H12/batch_normalization_28/batchnorm/mul_1Mul1Federated_CNN_F5_H12/conv1d_28/Relu:activations:0=Federated_CNN_F5_H12/batch_normalization_28/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€“
FFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp_1ReadVariableOpOfederated_cnn_f5_h12_batch_normalization_28_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ц
;Federated_CNN_F5_H12/batch_normalization_28/batchnorm/mul_2MulNFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp_1:value:0=Federated_CNN_F5_H12/batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:“
FFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp_2ReadVariableOpOfederated_cnn_f5_h12_batch_normalization_28_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ц
9Federated_CNN_F5_H12/batch_normalization_28/batchnorm/subSubNFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp_2:value:0?Federated_CNN_F5_H12/batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ъ
;Federated_CNN_F5_H12/batch_normalization_28/batchnorm/add_1AddV2?Federated_CNN_F5_H12/batch_normalization_28/batchnorm/mul_1:z:0=Federated_CNN_F5_H12/batch_normalization_28/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€
4Federated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€ш
0Federated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims
ExpandDims?Federated_CNN_F5_H12/batch_normalization_28/batchnorm/add_1:z:0=Federated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€–
AFederated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJfederated_cnn_f5_h12_conv1d_29_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0x
6Federated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : э
2Federated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims_1
ExpandDimsIFederated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims_1/ReadVariableOp:value:0?Federated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Й
%Federated_CNN_F5_H12/conv1d_29/Conv1DConv2D9Federated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims:output:0;Federated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
Њ
-Federated_CNN_F5_H12/conv1d_29/Conv1D/SqueezeSqueeze.Federated_CNN_F5_H12/conv1d_29/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€∞
5Federated_CNN_F5_H12/conv1d_29/BiasAdd/ReadVariableOpReadVariableOp>federated_cnn_f5_h12_conv1d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ё
&Federated_CNN_F5_H12/conv1d_29/BiasAddBiasAdd6Federated_CNN_F5_H12/conv1d_29/Conv1D/Squeeze:output:0=Federated_CNN_F5_H12/conv1d_29/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Т
#Federated_CNN_F5_H12/conv1d_29/ReluRelu/Federated_CNN_F5_H12/conv1d_29/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€ќ
DFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOpReadVariableOpMfederated_cnn_f5_h12_batch_normalization_29_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0А
;Federated_CNN_F5_H12/batch_normalization_29/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:ы
9Federated_CNN_F5_H12/batch_normalization_29/batchnorm/addAddV2LFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp:value:0DFederated_CNN_F5_H12/batch_normalization_29/batchnorm/add/y:output:0*
T0*
_output_shapes
:®
;Federated_CNN_F5_H12/batch_normalization_29/batchnorm/RsqrtRsqrt=Federated_CNN_F5_H12/batch_normalization_29/batchnorm/add:z:0*
T0*
_output_shapes
:÷
HFederated_CNN_F5_H12/batch_normalization_29/batchnorm/mul/ReadVariableOpReadVariableOpQfederated_cnn_f5_h12_batch_normalization_29_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0ш
9Federated_CNN_F5_H12/batch_normalization_29/batchnorm/mulMul?Federated_CNN_F5_H12/batch_normalization_29/batchnorm/Rsqrt:y:0PFederated_CNN_F5_H12/batch_normalization_29/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:к
;Federated_CNN_F5_H12/batch_normalization_29/batchnorm/mul_1Mul1Federated_CNN_F5_H12/conv1d_29/Relu:activations:0=Federated_CNN_F5_H12/batch_normalization_29/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€“
FFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp_1ReadVariableOpOfederated_cnn_f5_h12_batch_normalization_29_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ц
;Federated_CNN_F5_H12/batch_normalization_29/batchnorm/mul_2MulNFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp_1:value:0=Federated_CNN_F5_H12/batch_normalization_29/batchnorm/mul:z:0*
T0*
_output_shapes
:“
FFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp_2ReadVariableOpOfederated_cnn_f5_h12_batch_normalization_29_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ц
9Federated_CNN_F5_H12/batch_normalization_29/batchnorm/subSubNFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp_2:value:0?Federated_CNN_F5_H12/batch_normalization_29/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ъ
;Federated_CNN_F5_H12/batch_normalization_29/batchnorm/add_1AddV2?Federated_CNN_F5_H12/batch_normalization_29/batchnorm/mul_1:z:0=Federated_CNN_F5_H12/batch_normalization_29/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€
4Federated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€ш
0Federated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims
ExpandDims?Federated_CNN_F5_H12/batch_normalization_29/batchnorm/add_1:z:0=Federated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€–
AFederated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJfederated_cnn_f5_h12_conv1d_30_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0x
6Federated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : э
2Federated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims_1
ExpandDimsIFederated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims_1/ReadVariableOp:value:0?Federated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Й
%Federated_CNN_F5_H12/conv1d_30/Conv1DConv2D9Federated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims:output:0;Federated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
Њ
-Federated_CNN_F5_H12/conv1d_30/Conv1D/SqueezeSqueeze.Federated_CNN_F5_H12/conv1d_30/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€∞
5Federated_CNN_F5_H12/conv1d_30/BiasAdd/ReadVariableOpReadVariableOp>federated_cnn_f5_h12_conv1d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ё
&Federated_CNN_F5_H12/conv1d_30/BiasAddBiasAdd6Federated_CNN_F5_H12/conv1d_30/Conv1D/Squeeze:output:0=Federated_CNN_F5_H12/conv1d_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Т
#Federated_CNN_F5_H12/conv1d_30/ReluRelu/Federated_CNN_F5_H12/conv1d_30/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€ќ
DFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOpReadVariableOpMfederated_cnn_f5_h12_batch_normalization_30_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0А
;Federated_CNN_F5_H12/batch_normalization_30/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:ы
9Federated_CNN_F5_H12/batch_normalization_30/batchnorm/addAddV2LFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp:value:0DFederated_CNN_F5_H12/batch_normalization_30/batchnorm/add/y:output:0*
T0*
_output_shapes
:®
;Federated_CNN_F5_H12/batch_normalization_30/batchnorm/RsqrtRsqrt=Federated_CNN_F5_H12/batch_normalization_30/batchnorm/add:z:0*
T0*
_output_shapes
:÷
HFederated_CNN_F5_H12/batch_normalization_30/batchnorm/mul/ReadVariableOpReadVariableOpQfederated_cnn_f5_h12_batch_normalization_30_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0ш
9Federated_CNN_F5_H12/batch_normalization_30/batchnorm/mulMul?Federated_CNN_F5_H12/batch_normalization_30/batchnorm/Rsqrt:y:0PFederated_CNN_F5_H12/batch_normalization_30/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:к
;Federated_CNN_F5_H12/batch_normalization_30/batchnorm/mul_1Mul1Federated_CNN_F5_H12/conv1d_30/Relu:activations:0=Federated_CNN_F5_H12/batch_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€“
FFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp_1ReadVariableOpOfederated_cnn_f5_h12_batch_normalization_30_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ц
;Federated_CNN_F5_H12/batch_normalization_30/batchnorm/mul_2MulNFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp_1:value:0=Federated_CNN_F5_H12/batch_normalization_30/batchnorm/mul:z:0*
T0*
_output_shapes
:“
FFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp_2ReadVariableOpOfederated_cnn_f5_h12_batch_normalization_30_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ц
9Federated_CNN_F5_H12/batch_normalization_30/batchnorm/subSubNFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp_2:value:0?Federated_CNN_F5_H12/batch_normalization_30/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ъ
;Federated_CNN_F5_H12/batch_normalization_30/batchnorm/add_1AddV2?Federated_CNN_F5_H12/batch_normalization_30/batchnorm/mul_1:z:0=Federated_CNN_F5_H12/batch_normalization_30/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€
4Federated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€ш
0Federated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims
ExpandDims?Federated_CNN_F5_H12/batch_normalization_30/batchnorm/add_1:z:0=Federated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€–
AFederated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJfederated_cnn_f5_h12_conv1d_31_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0x
6Federated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : э
2Federated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims_1
ExpandDimsIFederated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims_1/ReadVariableOp:value:0?Federated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Й
%Federated_CNN_F5_H12/conv1d_31/Conv1DConv2D9Federated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims:output:0;Federated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
Њ
-Federated_CNN_F5_H12/conv1d_31/Conv1D/SqueezeSqueeze.Federated_CNN_F5_H12/conv1d_31/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€∞
5Federated_CNN_F5_H12/conv1d_31/BiasAdd/ReadVariableOpReadVariableOp>federated_cnn_f5_h12_conv1d_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ё
&Federated_CNN_F5_H12/conv1d_31/BiasAddBiasAdd6Federated_CNN_F5_H12/conv1d_31/Conv1D/Squeeze:output:0=Federated_CNN_F5_H12/conv1d_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Т
#Federated_CNN_F5_H12/conv1d_31/ReluRelu/Federated_CNN_F5_H12/conv1d_31/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€ќ
DFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOpReadVariableOpMfederated_cnn_f5_h12_batch_normalization_31_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0А
;Federated_CNN_F5_H12/batch_normalization_31/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:ы
9Federated_CNN_F5_H12/batch_normalization_31/batchnorm/addAddV2LFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp:value:0DFederated_CNN_F5_H12/batch_normalization_31/batchnorm/add/y:output:0*
T0*
_output_shapes
:®
;Federated_CNN_F5_H12/batch_normalization_31/batchnorm/RsqrtRsqrt=Federated_CNN_F5_H12/batch_normalization_31/batchnorm/add:z:0*
T0*
_output_shapes
:÷
HFederated_CNN_F5_H12/batch_normalization_31/batchnorm/mul/ReadVariableOpReadVariableOpQfederated_cnn_f5_h12_batch_normalization_31_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0ш
9Federated_CNN_F5_H12/batch_normalization_31/batchnorm/mulMul?Federated_CNN_F5_H12/batch_normalization_31/batchnorm/Rsqrt:y:0PFederated_CNN_F5_H12/batch_normalization_31/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:к
;Federated_CNN_F5_H12/batch_normalization_31/batchnorm/mul_1Mul1Federated_CNN_F5_H12/conv1d_31/Relu:activations:0=Federated_CNN_F5_H12/batch_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€“
FFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp_1ReadVariableOpOfederated_cnn_f5_h12_batch_normalization_31_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0ц
;Federated_CNN_F5_H12/batch_normalization_31/batchnorm/mul_2MulNFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp_1:value:0=Federated_CNN_F5_H12/batch_normalization_31/batchnorm/mul:z:0*
T0*
_output_shapes
:“
FFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp_2ReadVariableOpOfederated_cnn_f5_h12_batch_normalization_31_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0ц
9Federated_CNN_F5_H12/batch_normalization_31/batchnorm/subSubNFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp_2:value:0?Federated_CNN_F5_H12/batch_normalization_31/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ъ
;Federated_CNN_F5_H12/batch_normalization_31/batchnorm/add_1AddV2?Federated_CNN_F5_H12/batch_normalization_31/batchnorm/mul_1:z:0=Federated_CNN_F5_H12/batch_normalization_31/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Й
GFederated_CNN_F5_H12/global_average_pooling1d_14/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :В
5Federated_CNN_F5_H12/global_average_pooling1d_14/MeanMean?Federated_CNN_F5_H12/batch_normalization_31/batchnorm/add_1:z:0PFederated_CNN_F5_H12/global_average_pooling1d_14/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€∞
3Federated_CNN_F5_H12/dense_65/MatMul/ReadVariableOpReadVariableOp<federated_cnn_f5_h12_dense_65_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ё
$Federated_CNN_F5_H12/dense_65/MatMulMatMul>Federated_CNN_F5_H12/global_average_pooling1d_14/Mean:output:0;Federated_CNN_F5_H12/dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ѓ
4Federated_CNN_F5_H12/dense_65/BiasAdd/ReadVariableOpReadVariableOp=federated_cnn_f5_h12_dense_65_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0–
%Federated_CNN_F5_H12/dense_65/BiasAddBiasAdd.Federated_CNN_F5_H12/dense_65/MatMul:product:0<Federated_CNN_F5_H12/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ М
"Federated_CNN_F5_H12/dense_65/ReluRelu.Federated_CNN_F5_H12/dense_65/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ш
(Federated_CNN_F5_H12/dropout_15/IdentityIdentity0Federated_CNN_F5_H12/dense_65/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ ∞
3Federated_CNN_F5_H12/dense_66/MatMul/ReadVariableOpReadVariableOp<federated_cnn_f5_h12_dense_66_matmul_readvariableop_resource*
_output_shapes

: <*
dtype0–
$Federated_CNN_F5_H12/dense_66/MatMulMatMul1Federated_CNN_F5_H12/dropout_15/Identity:output:0;Federated_CNN_F5_H12/dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€<Ѓ
4Federated_CNN_F5_H12/dense_66/BiasAdd/ReadVariableOpReadVariableOp=federated_cnn_f5_h12_dense_66_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0–
%Federated_CNN_F5_H12/dense_66/BiasAddBiasAdd.Federated_CNN_F5_H12/dense_66/MatMul:product:0<Federated_CNN_F5_H12/dense_66/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€<С
%Federated_CNN_F5_H12/reshape_22/ShapeShape.Federated_CNN_F5_H12/dense_66/BiasAdd:output:0*
T0*
_output_shapes
::нѕ}
3Federated_CNN_F5_H12/reshape_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5Federated_CNN_F5_H12/reshape_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5Federated_CNN_F5_H12/reshape_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-Federated_CNN_F5_H12/reshape_22/strided_sliceStridedSlice.Federated_CNN_F5_H12/reshape_22/Shape:output:0<Federated_CNN_F5_H12/reshape_22/strided_slice/stack:output:0>Federated_CNN_F5_H12/reshape_22/strided_slice/stack_1:output:0>Federated_CNN_F5_H12/reshape_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/Federated_CNN_F5_H12/reshape_22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :q
/Federated_CNN_F5_H12/reshape_22/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :П
-Federated_CNN_F5_H12/reshape_22/Reshape/shapePack6Federated_CNN_F5_H12/reshape_22/strided_slice:output:08Federated_CNN_F5_H12/reshape_22/Reshape/shape/1:output:08Federated_CNN_F5_H12/reshape_22/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:–
'Federated_CNN_F5_H12/reshape_22/ReshapeReshape.Federated_CNN_F5_H12/dense_66/BiasAdd:output:06Federated_CNN_F5_H12/reshape_22/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€Г
IdentityIdentity0Federated_CNN_F5_H12/reshape_22/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€†
NoOpNoOpE^Federated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOpG^Federated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp_1G^Federated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp_2I^Federated_CNN_F5_H12/batch_normalization_28/batchnorm/mul/ReadVariableOpE^Federated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOpG^Federated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp_1G^Federated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp_2I^Federated_CNN_F5_H12/batch_normalization_29/batchnorm/mul/ReadVariableOpE^Federated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOpG^Federated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp_1G^Federated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp_2I^Federated_CNN_F5_H12/batch_normalization_30/batchnorm/mul/ReadVariableOpE^Federated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOpG^Federated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp_1G^Federated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp_2I^Federated_CNN_F5_H12/batch_normalization_31/batchnorm/mul/ReadVariableOp6^Federated_CNN_F5_H12/conv1d_28/BiasAdd/ReadVariableOpB^Federated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims_1/ReadVariableOp6^Federated_CNN_F5_H12/conv1d_29/BiasAdd/ReadVariableOpB^Federated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims_1/ReadVariableOp6^Federated_CNN_F5_H12/conv1d_30/BiasAdd/ReadVariableOpB^Federated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims_1/ReadVariableOp6^Federated_CNN_F5_H12/conv1d_31/BiasAdd/ReadVariableOpB^Federated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims_1/ReadVariableOp5^Federated_CNN_F5_H12/dense_65/BiasAdd/ReadVariableOp4^Federated_CNN_F5_H12/dense_65/MatMul/ReadVariableOp5^Federated_CNN_F5_H12/dense_66/BiasAdd/ReadVariableOp4^Federated_CNN_F5_H12/dense_66/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Р
FFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp_1FFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp_12Р
FFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp_2FFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp_22М
DFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOpDFederated_CNN_F5_H12/batch_normalization_28/batchnorm/ReadVariableOp2Ф
HFederated_CNN_F5_H12/batch_normalization_28/batchnorm/mul/ReadVariableOpHFederated_CNN_F5_H12/batch_normalization_28/batchnorm/mul/ReadVariableOp2Р
FFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp_1FFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp_12Р
FFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp_2FFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp_22М
DFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOpDFederated_CNN_F5_H12/batch_normalization_29/batchnorm/ReadVariableOp2Ф
HFederated_CNN_F5_H12/batch_normalization_29/batchnorm/mul/ReadVariableOpHFederated_CNN_F5_H12/batch_normalization_29/batchnorm/mul/ReadVariableOp2Р
FFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp_1FFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp_12Р
FFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp_2FFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp_22М
DFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOpDFederated_CNN_F5_H12/batch_normalization_30/batchnorm/ReadVariableOp2Ф
HFederated_CNN_F5_H12/batch_normalization_30/batchnorm/mul/ReadVariableOpHFederated_CNN_F5_H12/batch_normalization_30/batchnorm/mul/ReadVariableOp2Р
FFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp_1FFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp_12Р
FFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp_2FFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp_22М
DFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOpDFederated_CNN_F5_H12/batch_normalization_31/batchnorm/ReadVariableOp2Ф
HFederated_CNN_F5_H12/batch_normalization_31/batchnorm/mul/ReadVariableOpHFederated_CNN_F5_H12/batch_normalization_31/batchnorm/mul/ReadVariableOp2n
5Federated_CNN_F5_H12/conv1d_28/BiasAdd/ReadVariableOp5Federated_CNN_F5_H12/conv1d_28/BiasAdd/ReadVariableOp2Ж
AFederated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims_1/ReadVariableOpAFederated_CNN_F5_H12/conv1d_28/Conv1D/ExpandDims_1/ReadVariableOp2n
5Federated_CNN_F5_H12/conv1d_29/BiasAdd/ReadVariableOp5Federated_CNN_F5_H12/conv1d_29/BiasAdd/ReadVariableOp2Ж
AFederated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims_1/ReadVariableOpAFederated_CNN_F5_H12/conv1d_29/Conv1D/ExpandDims_1/ReadVariableOp2n
5Federated_CNN_F5_H12/conv1d_30/BiasAdd/ReadVariableOp5Federated_CNN_F5_H12/conv1d_30/BiasAdd/ReadVariableOp2Ж
AFederated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims_1/ReadVariableOpAFederated_CNN_F5_H12/conv1d_30/Conv1D/ExpandDims_1/ReadVariableOp2n
5Federated_CNN_F5_H12/conv1d_31/BiasAdd/ReadVariableOp5Federated_CNN_F5_H12/conv1d_31/BiasAdd/ReadVariableOp2Ж
AFederated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims_1/ReadVariableOpAFederated_CNN_F5_H12/conv1d_31/Conv1D/ExpandDims_1/ReadVariableOp2l
4Federated_CNN_F5_H12/dense_65/BiasAdd/ReadVariableOp4Federated_CNN_F5_H12/dense_65/BiasAdd/ReadVariableOp2j
3Federated_CNN_F5_H12/dense_65/MatMul/ReadVariableOp3Federated_CNN_F5_H12/dense_65/MatMul/ReadVariableOp2l
4Federated_CNN_F5_H12/dense_66/BiasAdd/ReadVariableOp4Federated_CNN_F5_H12/dense_66/BiasAdd/ReadVariableOp2j
3Federated_CNN_F5_H12/dense_66/MatMul/ReadVariableOp3Federated_CNN_F5_H12/dense_66/MatMul/ReadVariableOp:R N
+
_output_shapes
:€€€€€€€€€

_user_specified_nameInput
…	
ч
F__inference_dense_66_layer_call_and_return_conditional_losses_51800753

inputs0
matmul_readvariableop_resource: <-
biasadd_readvariableop_resource:<
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: <*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€<_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
є
в
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799347	
input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16: 

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: 

unknown_25: <

unknown_26:<
identityИҐStatefulPartitionedCall≈
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*6
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799288s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:€€€€€€€€€

_user_specified_nameInput
∆
Ш
+__inference_dense_66_layer_call_fn_51800743

inputs
unknown: <
	unknown_0:<
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_51799102o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
в
‘
9__inference_batch_normalization_30_layer_call_fn_51800517

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51798800|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ч
f
-__inference_dropout_15_layer_call_fn_51800712

inputs
identityИҐStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_51799090o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
в
‘
9__inference_batch_normalization_31_layer_call_fn_51800622

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51798882|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
№
Э
,__inference_conv1d_29_layer_call_fn_51800370

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_29_layer_call_and_return_conditional_losses_51798983s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
џ
f
H__inference_dropout_15_layer_call_and_return_conditional_losses_51799202

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
 
Ц
G__inference_conv1d_29_layer_call_and_return_conditional_losses_51800386

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:ђ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
А&
н
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51800551

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Г
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:Ф
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:і
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
А&
н
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51800341

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Г
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:Ф
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:і
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12"
AssignMovingAvgAssignMovingAvg24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
 
Ц
G__inference_conv1d_28_layer_call_and_return_conditional_losses_51800281

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Б
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€Т
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : †
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:ђ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
А
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Б
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€Д
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
…	
ч
F__inference_dense_66_layer_call_and_return_conditional_losses_51799102

inputs0
matmul_readvariableop_resource: <-
biasadd_readvariableop_resource:<
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: <*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€<_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
№
Э
,__inference_conv1d_31_layer_call_fn_51800580

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_31_layer_call_and_return_conditional_losses_51799045s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
®

g
H__inference_dropout_15_layer_call_and_return_conditional_losses_51800729

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::нѕШ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Т
≥
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51800571

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
а
‘
9__inference_batch_normalization_28_layer_call_fn_51800294

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51798616|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Т
≥
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51798636

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
∆
Ш
+__inference_dense_65_layer_call_fn_51800696

inputs
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_51799072o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
С
u
Y__inference_global_average_pooling1d_14_layer_call_and_return_conditional_losses_51798916

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
±
G
+__inference_lambda_7_layer_call_fn_51800240

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lambda_7_layer_call_and_return_conditional_losses_51799134d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
«∆
Э
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51800230

inputsK
5conv1d_28_conv1d_expanddims_1_readvariableop_resource:7
)conv1d_28_biasadd_readvariableop_resource:F
8batch_normalization_28_batchnorm_readvariableop_resource:J
<batch_normalization_28_batchnorm_mul_readvariableop_resource:H
:batch_normalization_28_batchnorm_readvariableop_1_resource:H
:batch_normalization_28_batchnorm_readvariableop_2_resource:K
5conv1d_29_conv1d_expanddims_1_readvariableop_resource:7
)conv1d_29_biasadd_readvariableop_resource:F
8batch_normalization_29_batchnorm_readvariableop_resource:J
<batch_normalization_29_batchnorm_mul_readvariableop_resource:H
:batch_normalization_29_batchnorm_readvariableop_1_resource:H
:batch_normalization_29_batchnorm_readvariableop_2_resource:K
5conv1d_30_conv1d_expanddims_1_readvariableop_resource:7
)conv1d_30_biasadd_readvariableop_resource:F
8batch_normalization_30_batchnorm_readvariableop_resource:J
<batch_normalization_30_batchnorm_mul_readvariableop_resource:H
:batch_normalization_30_batchnorm_readvariableop_1_resource:H
:batch_normalization_30_batchnorm_readvariableop_2_resource:K
5conv1d_31_conv1d_expanddims_1_readvariableop_resource:7
)conv1d_31_biasadd_readvariableop_resource:F
8batch_normalization_31_batchnorm_readvariableop_resource:J
<batch_normalization_31_batchnorm_mul_readvariableop_resource:H
:batch_normalization_31_batchnorm_readvariableop_1_resource:H
:batch_normalization_31_batchnorm_readvariableop_2_resource:9
'dense_65_matmul_readvariableop_resource: 6
(dense_65_biasadd_readvariableop_resource: 9
'dense_66_matmul_readvariableop_resource: <6
(dense_66_biasadd_readvariableop_resource:<
identityИҐ/batch_normalization_28/batchnorm/ReadVariableOpҐ1batch_normalization_28/batchnorm/ReadVariableOp_1Ґ1batch_normalization_28/batchnorm/ReadVariableOp_2Ґ3batch_normalization_28/batchnorm/mul/ReadVariableOpҐ/batch_normalization_29/batchnorm/ReadVariableOpҐ1batch_normalization_29/batchnorm/ReadVariableOp_1Ґ1batch_normalization_29/batchnorm/ReadVariableOp_2Ґ3batch_normalization_29/batchnorm/mul/ReadVariableOpҐ/batch_normalization_30/batchnorm/ReadVariableOpҐ1batch_normalization_30/batchnorm/ReadVariableOp_1Ґ1batch_normalization_30/batchnorm/ReadVariableOp_2Ґ3batch_normalization_30/batchnorm/mul/ReadVariableOpҐ/batch_normalization_31/batchnorm/ReadVariableOpҐ1batch_normalization_31/batchnorm/ReadVariableOp_1Ґ1batch_normalization_31/batchnorm/ReadVariableOp_2Ґ3batch_normalization_31/batchnorm/mul/ReadVariableOpҐ conv1d_28/BiasAdd/ReadVariableOpҐ,conv1d_28/Conv1D/ExpandDims_1/ReadVariableOpҐ conv1d_29/BiasAdd/ReadVariableOpҐ,conv1d_29/Conv1D/ExpandDims_1/ReadVariableOpҐ conv1d_30/BiasAdd/ReadVariableOpҐ,conv1d_30/Conv1D/ExpandDims_1/ReadVariableOpҐ conv1d_31/BiasAdd/ReadVariableOpҐ,conv1d_31/Conv1D/ExpandDims_1/ReadVariableOpҐdense_65/BiasAdd/ReadVariableOpҐdense_65/MatMul/ReadVariableOpҐdense_66/BiasAdd/ReadVariableOpҐdense_66/MatMul/ReadVariableOpq
lambda_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    э€€€    s
lambda_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            s
lambda_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         М
lambda_7/strided_sliceStridedSliceinputs%lambda_7/strided_slice/stack:output:0'lambda_7/strided_slice/stack_1:output:0'lambda_7/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€*

begin_mask*
end_maskj
conv1d_28/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€Ѓ
conv1d_28/Conv1D/ExpandDims
ExpandDimslambda_7/strided_slice:output:0(conv1d_28/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€¶
,conv1d_28/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_28_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_28/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Њ
conv1d_28/Conv1D/ExpandDims_1
ExpandDims4conv1d_28/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_28/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 
conv1d_28/Conv1DConv2D$conv1d_28/Conv1D/ExpandDims:output:0&conv1d_28/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
Ф
conv1d_28/Conv1D/SqueezeSqueezeconv1d_28/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€Ж
 conv1d_28/BiasAdd/ReadVariableOpReadVariableOp)conv1d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv1d_28/BiasAddBiasAdd!conv1d_28/Conv1D/Squeeze:output:0(conv1d_28/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€h
conv1d_28/ReluReluconv1d_28/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€§
/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Љ
$batch_normalization_28/batchnorm/addAddV27batch_normalization_28/batchnorm/ReadVariableOp:value:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:ђ
3batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0є
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:0;batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ђ
&batch_normalization_28/batchnorm/mul_1Mulconv1d_28/Relu:activations:0(batch_normalization_28/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€®
1batch_normalization_28/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_28_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0Ј
&batch_normalization_28/batchnorm/mul_2Mul9batch_normalization_28/batchnorm/ReadVariableOp_1:value:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:®
1batch_normalization_28/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_28_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0Ј
$batch_normalization_28/batchnorm/subSub9batch_normalization_28/batchnorm/ReadVariableOp_2:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ї
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€j
conv1d_29/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€є
conv1d_29/Conv1D/ExpandDims
ExpandDims*batch_normalization_28/batchnorm/add_1:z:0(conv1d_29/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€¶
,conv1d_29/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_29_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_29/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Њ
conv1d_29/Conv1D/ExpandDims_1
ExpandDims4conv1d_29/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_29/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 
conv1d_29/Conv1DConv2D$conv1d_29/Conv1D/ExpandDims:output:0&conv1d_29/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
Ф
conv1d_29/Conv1D/SqueezeSqueezeconv1d_29/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€Ж
 conv1d_29/BiasAdd/ReadVariableOpReadVariableOp)conv1d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv1d_29/BiasAddBiasAdd!conv1d_29/Conv1D/Squeeze:output:0(conv1d_29/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€h
conv1d_29/ReluReluconv1d_29/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€§
/batch_normalization_29/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_29_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_29/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Љ
$batch_normalization_29/batchnorm/addAddV27batch_normalization_29/batchnorm/ReadVariableOp:value:0/batch_normalization_29/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_29/batchnorm/RsqrtRsqrt(batch_normalization_29/batchnorm/add:z:0*
T0*
_output_shapes
:ђ
3batch_normalization_29/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_29_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0є
$batch_normalization_29/batchnorm/mulMul*batch_normalization_29/batchnorm/Rsqrt:y:0;batch_normalization_29/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ђ
&batch_normalization_29/batchnorm/mul_1Mulconv1d_29/Relu:activations:0(batch_normalization_29/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€®
1batch_normalization_29/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_29_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0Ј
&batch_normalization_29/batchnorm/mul_2Mul9batch_normalization_29/batchnorm/ReadVariableOp_1:value:0(batch_normalization_29/batchnorm/mul:z:0*
T0*
_output_shapes
:®
1batch_normalization_29/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_29_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0Ј
$batch_normalization_29/batchnorm/subSub9batch_normalization_29/batchnorm/ReadVariableOp_2:value:0*batch_normalization_29/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ї
&batch_normalization_29/batchnorm/add_1AddV2*batch_normalization_29/batchnorm/mul_1:z:0(batch_normalization_29/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€j
conv1d_30/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€є
conv1d_30/Conv1D/ExpandDims
ExpandDims*batch_normalization_29/batchnorm/add_1:z:0(conv1d_30/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€¶
,conv1d_30/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_30_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_30/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Њ
conv1d_30/Conv1D/ExpandDims_1
ExpandDims4conv1d_30/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_30/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 
conv1d_30/Conv1DConv2D$conv1d_30/Conv1D/ExpandDims:output:0&conv1d_30/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
Ф
conv1d_30/Conv1D/SqueezeSqueezeconv1d_30/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€Ж
 conv1d_30/BiasAdd/ReadVariableOpReadVariableOp)conv1d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv1d_30/BiasAddBiasAdd!conv1d_30/Conv1D/Squeeze:output:0(conv1d_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€h
conv1d_30/ReluReluconv1d_30/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€§
/batch_normalization_30/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_30_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_30/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Љ
$batch_normalization_30/batchnorm/addAddV27batch_normalization_30/batchnorm/ReadVariableOp:value:0/batch_normalization_30/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_30/batchnorm/RsqrtRsqrt(batch_normalization_30/batchnorm/add:z:0*
T0*
_output_shapes
:ђ
3batch_normalization_30/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_30_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0є
$batch_normalization_30/batchnorm/mulMul*batch_normalization_30/batchnorm/Rsqrt:y:0;batch_normalization_30/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ђ
&batch_normalization_30/batchnorm/mul_1Mulconv1d_30/Relu:activations:0(batch_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€®
1batch_normalization_30/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_30_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0Ј
&batch_normalization_30/batchnorm/mul_2Mul9batch_normalization_30/batchnorm/ReadVariableOp_1:value:0(batch_normalization_30/batchnorm/mul:z:0*
T0*
_output_shapes
:®
1batch_normalization_30/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_30_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0Ј
$batch_normalization_30/batchnorm/subSub9batch_normalization_30/batchnorm/ReadVariableOp_2:value:0*batch_normalization_30/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ї
&batch_normalization_30/batchnorm/add_1AddV2*batch_normalization_30/batchnorm/mul_1:z:0(batch_normalization_30/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€j
conv1d_31/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€є
conv1d_31/Conv1D/ExpandDims
ExpandDims*batch_normalization_30/batchnorm/add_1:z:0(conv1d_31/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€¶
,conv1d_31/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_31_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0c
!conv1d_31/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Њ
conv1d_31/Conv1D/ExpandDims_1
ExpandDims4conv1d_31/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_31/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 
conv1d_31/Conv1DConv2D$conv1d_31/Conv1D/ExpandDims:output:0&conv1d_31/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingSAME*
strides
Ф
conv1d_31/Conv1D/SqueezeSqueezeconv1d_31/Conv1D:output:0*
T0*+
_output_shapes
:€€€€€€€€€*
squeeze_dims

э€€€€€€€€Ж
 conv1d_31/BiasAdd/ReadVariableOpReadVariableOp)conv1d_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
conv1d_31/BiasAddBiasAdd!conv1d_31/Conv1D/Squeeze:output:0(conv1d_31/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€h
conv1d_31/ReluReluconv1d_31/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€§
/batch_normalization_31/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_31_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0k
&batch_normalization_31/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Љ
$batch_normalization_31/batchnorm/addAddV27batch_normalization_31/batchnorm/ReadVariableOp:value:0/batch_normalization_31/batchnorm/add/y:output:0*
T0*
_output_shapes
:~
&batch_normalization_31/batchnorm/RsqrtRsqrt(batch_normalization_31/batchnorm/add:z:0*
T0*
_output_shapes
:ђ
3batch_normalization_31/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_31_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0є
$batch_normalization_31/batchnorm/mulMul*batch_normalization_31/batchnorm/Rsqrt:y:0;batch_normalization_31/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Ђ
&batch_normalization_31/batchnorm/mul_1Mulconv1d_31/Relu:activations:0(batch_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€®
1batch_normalization_31/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_31_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0Ј
&batch_normalization_31/batchnorm/mul_2Mul9batch_normalization_31/batchnorm/ReadVariableOp_1:value:0(batch_normalization_31/batchnorm/mul:z:0*
T0*
_output_shapes
:®
1batch_normalization_31/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_31_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0Ј
$batch_normalization_31/batchnorm/subSub9batch_normalization_31/batchnorm/ReadVariableOp_2:value:0*batch_normalization_31/batchnorm/mul_2:z:0*
T0*
_output_shapes
:ї
&batch_normalization_31/batchnorm/add_1AddV2*batch_normalization_31/batchnorm/mul_1:z:0(batch_normalization_31/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€t
2global_average_pooling1d_14/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :√
 global_average_pooling1d_14/MeanMean*batch_normalization_31/batchnorm/add_1:z:0;global_average_pooling1d_14/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ю
dense_65/MatMulMatMul)global_average_pooling1d_14/Mean:output:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0С
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ b
dense_65/ReluReludense_65/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ n
dropout_15/IdentityIdentitydense_65/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ Ж
dense_66/MatMul/ReadVariableOpReadVariableOp'dense_66_matmul_readvariableop_resource*
_output_shapes

: <*
dtype0С
dense_66/MatMulMatMuldropout_15/Identity:output:0&dense_66/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€<Д
dense_66/BiasAdd/ReadVariableOpReadVariableOp(dense_66_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0С
dense_66/BiasAddBiasAdddense_66/MatMul:product:0'dense_66/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€<g
reshape_22/ShapeShapedense_66/BiasAdd:output:0*
T0*
_output_shapes
::нѕh
reshape_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 reshape_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 reshape_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
reshape_22/strided_sliceStridedSlicereshape_22/Shape:output:0'reshape_22/strided_slice/stack:output:0)reshape_22/strided_slice/stack_1:output:0)reshape_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_22/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_22/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :ї
reshape_22/Reshape/shapePack!reshape_22/strided_slice:output:0#reshape_22/Reshape/shape/1:output:0#reshape_22/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:С
reshape_22/ReshapeReshapedense_66/BiasAdd:output:0!reshape_22/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€n
IdentityIdentityreshape_22/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€‘

NoOpNoOp0^batch_normalization_28/batchnorm/ReadVariableOp2^batch_normalization_28/batchnorm/ReadVariableOp_12^batch_normalization_28/batchnorm/ReadVariableOp_24^batch_normalization_28/batchnorm/mul/ReadVariableOp0^batch_normalization_29/batchnorm/ReadVariableOp2^batch_normalization_29/batchnorm/ReadVariableOp_12^batch_normalization_29/batchnorm/ReadVariableOp_24^batch_normalization_29/batchnorm/mul/ReadVariableOp0^batch_normalization_30/batchnorm/ReadVariableOp2^batch_normalization_30/batchnorm/ReadVariableOp_12^batch_normalization_30/batchnorm/ReadVariableOp_24^batch_normalization_30/batchnorm/mul/ReadVariableOp0^batch_normalization_31/batchnorm/ReadVariableOp2^batch_normalization_31/batchnorm/ReadVariableOp_12^batch_normalization_31/batchnorm/ReadVariableOp_24^batch_normalization_31/batchnorm/mul/ReadVariableOp!^conv1d_28/BiasAdd/ReadVariableOp-^conv1d_28/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_29/BiasAdd/ReadVariableOp-^conv1d_29/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_30/BiasAdd/ReadVariableOp-^conv1d_30/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_31/BiasAdd/ReadVariableOp-^conv1d_31/Conv1D/ExpandDims_1/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp ^dense_66/BiasAdd/ReadVariableOp^dense_66/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1batch_normalization_28/batchnorm/ReadVariableOp_11batch_normalization_28/batchnorm/ReadVariableOp_12f
1batch_normalization_28/batchnorm/ReadVariableOp_21batch_normalization_28/batchnorm/ReadVariableOp_22b
/batch_normalization_28/batchnorm/ReadVariableOp/batch_normalization_28/batchnorm/ReadVariableOp2j
3batch_normalization_28/batchnorm/mul/ReadVariableOp3batch_normalization_28/batchnorm/mul/ReadVariableOp2f
1batch_normalization_29/batchnorm/ReadVariableOp_11batch_normalization_29/batchnorm/ReadVariableOp_12f
1batch_normalization_29/batchnorm/ReadVariableOp_21batch_normalization_29/batchnorm/ReadVariableOp_22b
/batch_normalization_29/batchnorm/ReadVariableOp/batch_normalization_29/batchnorm/ReadVariableOp2j
3batch_normalization_29/batchnorm/mul/ReadVariableOp3batch_normalization_29/batchnorm/mul/ReadVariableOp2f
1batch_normalization_30/batchnorm/ReadVariableOp_11batch_normalization_30/batchnorm/ReadVariableOp_12f
1batch_normalization_30/batchnorm/ReadVariableOp_21batch_normalization_30/batchnorm/ReadVariableOp_22b
/batch_normalization_30/batchnorm/ReadVariableOp/batch_normalization_30/batchnorm/ReadVariableOp2j
3batch_normalization_30/batchnorm/mul/ReadVariableOp3batch_normalization_30/batchnorm/mul/ReadVariableOp2f
1batch_normalization_31/batchnorm/ReadVariableOp_11batch_normalization_31/batchnorm/ReadVariableOp_12f
1batch_normalization_31/batchnorm/ReadVariableOp_21batch_normalization_31/batchnorm/ReadVariableOp_22b
/batch_normalization_31/batchnorm/ReadVariableOp/batch_normalization_31/batchnorm/ReadVariableOp2j
3batch_normalization_31/batchnorm/mul/ReadVariableOp3batch_normalization_31/batchnorm/mul/ReadVariableOp2D
 conv1d_28/BiasAdd/ReadVariableOp conv1d_28/BiasAdd/ReadVariableOp2\
,conv1d_28/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_28/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_29/BiasAdd/ReadVariableOp conv1d_29/BiasAdd/ReadVariableOp2\
,conv1d_29/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_29/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_30/BiasAdd/ReadVariableOp conv1d_30/BiasAdd/ReadVariableOp2\
,conv1d_30/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_30/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_31/BiasAdd/ReadVariableOp conv1d_31/BiasAdd/ReadVariableOp2\
,conv1d_31/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_31/Conv1D/ExpandDims_1/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2B
dense_66/BiasAdd/ReadVariableOpdense_66/BiasAdd/ReadVariableOp2@
dense_66/MatMul/ReadVariableOpdense_66/MatMul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
®

g
H__inference_dropout_15_layer_call_and_return_conditional_losses_51799090

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::нѕШ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
џ
f
H__inference_dropout_15_layer_call_and_return_conditional_losses_51800734

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€ [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Э

ч
F__inference_dense_65_layer_call_and_return_conditional_losses_51800707

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѕ
в
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799482	
input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16: 

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23: 

unknown_24: 

unknown_25: <

unknown_26:<
identityИҐStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799423s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:€€€€€€€€€

_user_specified_nameInput
‘J
ґ
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799211	
input(
conv1d_28_51799136: 
conv1d_28_51799138:-
batch_normalization_28_51799141:-
batch_normalization_28_51799143:-
batch_normalization_28_51799145:-
batch_normalization_28_51799147:(
conv1d_29_51799150: 
conv1d_29_51799152:-
batch_normalization_29_51799155:-
batch_normalization_29_51799157:-
batch_normalization_29_51799159:-
batch_normalization_29_51799161:(
conv1d_30_51799164: 
conv1d_30_51799166:-
batch_normalization_30_51799169:-
batch_normalization_30_51799171:-
batch_normalization_30_51799173:-
batch_normalization_30_51799175:(
conv1d_31_51799178: 
conv1d_31_51799180:-
batch_normalization_31_51799183:-
batch_normalization_31_51799185:-
batch_normalization_31_51799187:-
batch_normalization_31_51799189:#
dense_65_51799193: 
dense_65_51799195: #
dense_66_51799204: <
dense_66_51799206:<
identityИҐ.batch_normalization_28/StatefulPartitionedCallҐ.batch_normalization_29/StatefulPartitionedCallҐ.batch_normalization_30/StatefulPartitionedCallҐ.batch_normalization_31/StatefulPartitionedCallҐ!conv1d_28/StatefulPartitionedCallҐ!conv1d_29/StatefulPartitionedCallҐ!conv1d_30/StatefulPartitionedCallҐ!conv1d_31/StatefulPartitionedCallҐ dense_65/StatefulPartitionedCallҐ dense_66/StatefulPartitionedCallљ
lambda_7/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lambda_7_layer_call_and_return_conditional_losses_51799134Щ
!conv1d_28/StatefulPartitionedCallStatefulPartitionedCall!lambda_7/PartitionedCall:output:0conv1d_28_51799136conv1d_28_51799138*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_28_layer_call_and_return_conditional_losses_51798952Ь
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv1d_28/StatefulPartitionedCall:output:0batch_normalization_28_51799141batch_normalization_28_51799143batch_normalization_28_51799145batch_normalization_28_51799147*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51798636ѓ
!conv1d_29/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0conv1d_29_51799150conv1d_29_51799152*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_29_layer_call_and_return_conditional_losses_51798983Ь
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall*conv1d_29/StatefulPartitionedCall:output:0batch_normalization_29_51799155batch_normalization_29_51799157batch_normalization_29_51799159batch_normalization_29_51799161*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51798718ѓ
!conv1d_30/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0conv1d_30_51799164conv1d_30_51799166*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_30_layer_call_and_return_conditional_losses_51799014Ь
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall*conv1d_30/StatefulPartitionedCall:output:0batch_normalization_30_51799169batch_normalization_30_51799171batch_normalization_30_51799173batch_normalization_30_51799175*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51798800ѓ
!conv1d_31/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_30/StatefulPartitionedCall:output:0conv1d_31_51799178conv1d_31_51799180*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_31_layer_call_and_return_conditional_losses_51799045Ь
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCall*conv1d_31/StatefulPartitionedCall:output:0batch_normalization_31_51799183batch_normalization_31_51799185batch_normalization_31_51799187batch_normalization_31_51799189*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51798882С
+global_average_pooling1d_14/PartitionedCallPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *b
f]R[
Y__inference_global_average_pooling1d_14_layer_call_and_return_conditional_losses_51798916§
 dense_65/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_14/PartitionedCall:output:0dense_65_51799193dense_65_51799195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_51799072б
dropout_15/PartitionedCallPartitionedCall)dense_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_51799202У
 dense_66/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_66_51799204dense_66_51799206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_51799102е
reshape_22/PartitionedCallPartitionedCall)dense_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_reshape_22_layer_call_and_return_conditional_losses_51799121v
IdentityIdentity#reshape_22/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€а
NoOpNoOp/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall/^batch_normalization_30/StatefulPartitionedCall/^batch_normalization_31/StatefulPartitionedCall"^conv1d_28/StatefulPartitionedCall"^conv1d_29/StatefulPartitionedCall"^conv1d_30/StatefulPartitionedCall"^conv1d_31/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2F
!conv1d_28/StatefulPartitionedCall!conv1d_28/StatefulPartitionedCall2F
!conv1d_29/StatefulPartitionedCall!conv1d_29/StatefulPartitionedCall2F
!conv1d_30/StatefulPartitionedCall!conv1d_30/StatefulPartitionedCall2F
!conv1d_31/StatefulPartitionedCall!conv1d_31/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall:R N
+
_output_shapes
:€€€€€€€€€

_user_specified_nameInput
њ
b
F__inference_lambda_7_layer_call_and_return_conditional_losses_51799134

inputs
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    э€€€    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         и
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:€€€€€€€€€*

begin_mask*
end_maskb
IdentityIdentitystrided_slice:output:0*
T0*+
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
„J
Ј
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799423

inputs(
conv1d_28_51799353: 
conv1d_28_51799355:-
batch_normalization_28_51799358:-
batch_normalization_28_51799360:-
batch_normalization_28_51799362:-
batch_normalization_28_51799364:(
conv1d_29_51799367: 
conv1d_29_51799369:-
batch_normalization_29_51799372:-
batch_normalization_29_51799374:-
batch_normalization_29_51799376:-
batch_normalization_29_51799378:(
conv1d_30_51799381: 
conv1d_30_51799383:-
batch_normalization_30_51799386:-
batch_normalization_30_51799388:-
batch_normalization_30_51799390:-
batch_normalization_30_51799392:(
conv1d_31_51799395: 
conv1d_31_51799397:-
batch_normalization_31_51799400:-
batch_normalization_31_51799402:-
batch_normalization_31_51799404:-
batch_normalization_31_51799406:#
dense_65_51799410: 
dense_65_51799412: #
dense_66_51799416: <
dense_66_51799418:<
identityИҐ.batch_normalization_28/StatefulPartitionedCallҐ.batch_normalization_29/StatefulPartitionedCallҐ.batch_normalization_30/StatefulPartitionedCallҐ.batch_normalization_31/StatefulPartitionedCallҐ!conv1d_28/StatefulPartitionedCallҐ!conv1d_29/StatefulPartitionedCallҐ!conv1d_30/StatefulPartitionedCallҐ!conv1d_31/StatefulPartitionedCallҐ dense_65/StatefulPartitionedCallҐ dense_66/StatefulPartitionedCallЊ
lambda_7/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lambda_7_layer_call_and_return_conditional_losses_51799134Щ
!conv1d_28/StatefulPartitionedCallStatefulPartitionedCall!lambda_7/PartitionedCall:output:0conv1d_28_51799353conv1d_28_51799355*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_28_layer_call_and_return_conditional_losses_51798952Ь
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv1d_28/StatefulPartitionedCall:output:0batch_normalization_28_51799358batch_normalization_28_51799360batch_normalization_28_51799362batch_normalization_28_51799364*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51798636ѓ
!conv1d_29/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0conv1d_29_51799367conv1d_29_51799369*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_29_layer_call_and_return_conditional_losses_51798983Ь
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall*conv1d_29/StatefulPartitionedCall:output:0batch_normalization_29_51799372batch_normalization_29_51799374batch_normalization_29_51799376batch_normalization_29_51799378*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51798718ѓ
!conv1d_30/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0conv1d_30_51799381conv1d_30_51799383*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_30_layer_call_and_return_conditional_losses_51799014Ь
.batch_normalization_30/StatefulPartitionedCallStatefulPartitionedCall*conv1d_30/StatefulPartitionedCall:output:0batch_normalization_30_51799386batch_normalization_30_51799388batch_normalization_30_51799390batch_normalization_30_51799392*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51798800ѓ
!conv1d_31/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_30/StatefulPartitionedCall:output:0conv1d_31_51799395conv1d_31_51799397*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_conv1d_31_layer_call_and_return_conditional_losses_51799045Ь
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCall*conv1d_31/StatefulPartitionedCall:output:0batch_normalization_31_51799400batch_normalization_31_51799402batch_normalization_31_51799404batch_normalization_31_51799406*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51798882С
+global_average_pooling1d_14/PartitionedCallPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *b
f]R[
Y__inference_global_average_pooling1d_14_layer_call_and_return_conditional_losses_51798916§
 dense_65/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_14/PartitionedCall:output:0dense_65_51799410dense_65_51799412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_51799072б
dropout_15/PartitionedCallPartitionedCall)dense_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_51799202У
 dense_66/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_66_51799416dense_66_51799418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_51799102е
reshape_22/PartitionedCallPartitionedCall)dense_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_reshape_22_layer_call_and_return_conditional_losses_51799121v
IdentityIdentity#reshape_22/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€а
NoOpNoOp/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall/^batch_normalization_30/StatefulPartitionedCall/^batch_normalization_31/StatefulPartitionedCall"^conv1d_28/StatefulPartitionedCall"^conv1d_29/StatefulPartitionedCall"^conv1d_30/StatefulPartitionedCall"^conv1d_31/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall!^dense_66/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2`
.batch_normalization_30/StatefulPartitionedCall.batch_normalization_30/StatefulPartitionedCall2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2F
!conv1d_28/StatefulPartitionedCall!conv1d_28/StatefulPartitionedCall2F
!conv1d_29/StatefulPartitionedCall!conv1d_29/StatefulPartitionedCall2F
!conv1d_30/StatefulPartitionedCall!conv1d_30/StatefulPartitionedCall2F
!conv1d_31/StatefulPartitionedCall!conv1d_31/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Т
≥
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51798718

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€€€€€€€€€€: : : : 28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_224
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Е
Z
>__inference_global_average_pooling1d_14_layer_call_fn_51800681

inputs
identityЌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *b
f]R[
Y__inference_global_average_pooling1d_14_layer_call_and_return_conditional_losses_51798916i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs"у
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_defaultЭ
;
Input2
serving_default_Input:0€€€€€€€€€B

reshape_224
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:бс
†
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer-10
layer_with_weights-8
layer-11
layer-12
layer_with_weights-9
layer-13
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
•
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ё
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias
 &_jit_compiled_convolution_op"
_tf_keras_layer
к
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-axis
	.gamma
/beta
0moving_mean
1moving_variance"
_tf_keras_layer
Ё
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias
 :_jit_compiled_convolution_op"
_tf_keras_layer
к
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance"
_tf_keras_layer
Ё
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

Lkernel
Mbias
 N_jit_compiled_convolution_op"
_tf_keras_layer
к
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
Uaxis
	Vgamma
Wbeta
Xmoving_mean
Ymoving_variance"
_tf_keras_layer
Ё
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias
 b_jit_compiled_convolution_op"
_tf_keras_layer
к
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses
iaxis
	jgamma
kbeta
lmoving_mean
mmoving_variance"
_tf_keras_layer
•
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias"
_tf_keras_layer
њ
|	variables
}trainable_variables
~regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
В_random_generator"
_tf_keras_layer
√
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses
Йkernel
	Кbias"
_tf_keras_layer
Ђ
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"
_tf_keras_layer
ш
$0
%1
.2
/3
04
15
86
97
B8
C9
D10
E11
L12
M13
V14
W15
X16
Y17
`18
a19
j20
k21
l22
m23
z24
{25
Й26
К27"
trackable_list_wrapper
Є
$0
%1
.2
/3
84
95
B6
C7
L8
M9
V10
W11
`12
a13
j14
k15
z16
{17
Й18
К19"
trackable_list_wrapper
 "
trackable_list_wrapper
ѕ
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
П
Цtrace_0
Чtrace_1
Шtrace_2
Щtrace_32Ь
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799347
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799482
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799816
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799877µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЦtrace_0zЧtrace_1zШtrace_2zЩtrace_3
ы
Ъtrace_0
Ыtrace_1
Ьtrace_2
Эtrace_32И
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799124
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799211
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51800085
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51800230µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЪtrace_0zЫtrace_1zЬtrace_2zЭtrace_3
ћB…
#__inference__wrapped_model_51798581Input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
-
Юserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Яnon_trainable_variables
†layers
°metrics
 Ґlayer_regularization_losses
£layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ќ
§trace_0
•trace_12Т
+__inference_lambda_7_layer_call_fn_51800235
+__inference_lambda_7_layer_call_fn_51800240µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z§trace_0z•trace_1
Г
¶trace_0
Іtrace_12»
F__inference_lambda_7_layer_call_and_return_conditional_losses_51800248
F__inference_lambda_7_layer_call_and_return_conditional_losses_51800256µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z¶trace_0zІtrace_1
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
®non_trainable_variables
©layers
™metrics
 Ђlayer_regularization_losses
ђlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
и
≠trace_02…
,__inference_conv1d_28_layer_call_fn_51800265Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 z≠trace_0
Г
Ѓtrace_02д
G__inference_conv1d_28_layer_call_and_return_conditional_losses_51800281Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zЃtrace_0
&:$2conv1d_28/kernel
:2conv1d_28/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
<
.0
/1
02
13"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
ѓnon_trainable_variables
∞layers
±metrics
 ≤layer_regularization_losses
≥layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
й
іtrace_0
µtrace_12Ѓ
9__inference_batch_normalization_28_layer_call_fn_51800294
9__inference_batch_normalization_28_layer_call_fn_51800307µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zіtrace_0zµtrace_1
Я
ґtrace_0
Јtrace_12д
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51800341
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51800361µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zґtrace_0zЈtrace_1
 "
trackable_list_wrapper
*:(2batch_normalization_28/gamma
):'2batch_normalization_28/beta
2:0 (2"batch_normalization_28/moving_mean
6:4 (2&batch_normalization_28/moving_variance
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Љlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
и
љtrace_02…
,__inference_conv1d_29_layer_call_fn_51800370Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zљtrace_0
Г
Њtrace_02д
G__inference_conv1d_29_layer_call_and_return_conditional_losses_51800386Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zЊtrace_0
&:$2conv1d_29/kernel
:2conv1d_29/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
њnon_trainable_variables
јlayers
Ѕmetrics
 ¬layer_regularization_losses
√layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
й
ƒtrace_0
≈trace_12Ѓ
9__inference_batch_normalization_29_layer_call_fn_51800399
9__inference_batch_normalization_29_layer_call_fn_51800412µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zƒtrace_0z≈trace_1
Я
∆trace_0
«trace_12д
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51800446
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51800466µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z∆trace_0z«trace_1
 "
trackable_list_wrapper
*:(2batch_normalization_29/gamma
):'2batch_normalization_29/beta
2:0 (2"batch_normalization_29/moving_mean
6:4 (2&batch_normalization_29/moving_variance
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
и
Ќtrace_02…
,__inference_conv1d_30_layer_call_fn_51800475Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zЌtrace_0
Г
ќtrace_02д
G__inference_conv1d_30_layer_call_and_return_conditional_losses_51800491Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zќtrace_0
&:$2conv1d_30/kernel
:2conv1d_30/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
<
V0
W1
X2
Y3"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
ѕnon_trainable_variables
–layers
—metrics
 “layer_regularization_losses
”layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
й
‘trace_0
’trace_12Ѓ
9__inference_batch_normalization_30_layer_call_fn_51800504
9__inference_batch_normalization_30_layer_call_fn_51800517µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z‘trace_0z’trace_1
Я
÷trace_0
„trace_12д
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51800551
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51800571µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z÷trace_0z„trace_1
 "
trackable_list_wrapper
*:(2batch_normalization_30/gamma
):'2batch_normalization_30/beta
2:0 (2"batch_normalization_30/moving_mean
6:4 (2&batch_normalization_30/moving_variance
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Ўnon_trainable_variables
ўlayers
Џmetrics
 џlayer_regularization_losses
№layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
и
Ёtrace_02…
,__inference_conv1d_31_layer_call_fn_51800580Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zЁtrace_0
Г
ёtrace_02д
G__inference_conv1d_31_layer_call_and_return_conditional_losses_51800596Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zёtrace_0
&:$2conv1d_31/kernel
:2conv1d_31/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
<
j0
k1
l2
m3"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
й
дtrace_0
еtrace_12Ѓ
9__inference_batch_normalization_31_layer_call_fn_51800609
9__inference_batch_normalization_31_layer_call_fn_51800622µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zдtrace_0zеtrace_1
Я
жtrace_0
зtrace_12д
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51800656
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51800676µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zжtrace_0zзtrace_1
 "
trackable_list_wrapper
*:(2batch_normalization_31/gamma
):'2batch_normalization_31/beta
2:0 (2"batch_normalization_31/moving_mean
6:4 (2&batch_normalization_31/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
З
нtrace_02и
>__inference_global_average_pooling1d_14_layer_call_fn_51800681•
Ю≤Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zнtrace_0
Ґ
оtrace_02Г
Y__inference_global_average_pooling1d_14_layer_call_and_return_conditional_losses_51800687•
Ю≤Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zоtrace_0
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
з
фtrace_02»
+__inference_dense_65_layer_call_fn_51800696Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zфtrace_0
В
хtrace_02г
F__inference_dense_65_layer_call_and_return_conditional_losses_51800707Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zхtrace_0
!: 2dense_65/kernel
: 2dense_65/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
|	variables
}trainable_variables
~regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
≈
ыtrace_0
ьtrace_12К
-__inference_dropout_15_layer_call_fn_51800712
-__inference_dropout_15_layer_call_fn_51800717©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zыtrace_0zьtrace_1
ы
эtrace_0
юtrace_12ј
H__inference_dropout_15_layer_call_and_return_conditional_losses_51800729
H__inference_dropout_15_layer_call_and_return_conditional_losses_51800734©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zэtrace_0zюtrace_1
"
_generic_user_object
0
Й0
К1"
trackable_list_wrapper
0
Й0
К1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
€non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
з
Дtrace_02»
+__inference_dense_66_layer_call_fn_51800743Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zДtrace_0
В
Еtrace_02г
F__inference_dense_66_layer_call_and_return_conditional_losses_51800753Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zЕtrace_0
!: <2dense_66/kernel
:<2dense_66/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
Л	variables
Мtrainable_variables
Нregularization_losses
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
й
Лtrace_02 
-__inference_reshape_22_layer_call_fn_51800758Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zЛtrace_0
Д
Мtrace_02е
H__inference_reshape_22_layer_call_and_return_conditional_losses_51800771Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 zМtrace_0
X
00
11
D2
E3
X4
Y5
l6
m7"
trackable_list_wrapper
О
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799347Input"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
эBъ
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799482Input"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799816inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799877inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ШBХ
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799124Input"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ШBХ
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799211Input"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЩBЦ
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51800085inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЩBЦ
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51800230inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЋB»
&__inference_signature_wrapper_51799755Input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
+__inference_lambda_7_layer_call_fn_51800235inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
тBп
+__inference_lambda_7_layer_call_fn_51800240inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
НBК
F__inference_lambda_7_layer_call_and_return_conditional_losses_51800248inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
НBК
F__inference_lambda_7_layer_call_and_return_conditional_losses_51800256inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsҐ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
÷B”
,__inference_conv1d_28_layer_call_fn_51800265inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
сBо
G__inference_conv1d_28_layer_call_and_return_conditional_losses_51800281inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
АBэ
9__inference_batch_normalization_28_layer_call_fn_51800294inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
9__inference_batch_normalization_28_layer_call_fn_51800307inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЫBШ
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51800341inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЫBШ
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51800361inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
÷B”
,__inference_conv1d_29_layer_call_fn_51800370inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
сBо
G__inference_conv1d_29_layer_call_and_return_conditional_losses_51800386inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
АBэ
9__inference_batch_normalization_29_layer_call_fn_51800399inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
9__inference_batch_normalization_29_layer_call_fn_51800412inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЫBШ
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51800446inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЫBШ
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51800466inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
÷B”
,__inference_conv1d_30_layer_call_fn_51800475inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
сBо
G__inference_conv1d_30_layer_call_and_return_conditional_losses_51800491inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
АBэ
9__inference_batch_normalization_30_layer_call_fn_51800504inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
9__inference_batch_normalization_30_layer_call_fn_51800517inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЫBШ
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51800551inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЫBШ
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51800571inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
÷B”
,__inference_conv1d_31_layer_call_fn_51800580inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
сBо
G__inference_conv1d_31_layer_call_and_return_conditional_losses_51800596inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
АBэ
9__inference_batch_normalization_31_layer_call_fn_51800609inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
9__inference_batch_normalization_31_layer_call_fn_51800622inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЫBШ
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51800656inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЫBШ
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51800676inputs"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
>__inference_global_average_pooling1d_14_layer_call_fn_51800681inputs"•
Ю≤Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
РBН
Y__inference_global_average_pooling1d_14_layer_call_and_return_conditional_losses_51800687inputs"•
Ю≤Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
’B“
+__inference_dense_65_layer_call_fn_51800696inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
рBн
F__inference_dense_65_layer_call_and_return_conditional_losses_51800707inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
-__inference_dropout_15_layer_call_fn_51800712inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
иBе
-__inference_dropout_15_layer_call_fn_51800717inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ГBА
H__inference_dropout_15_layer_call_and_return_conditional_losses_51800729inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ГBА
H__inference_dropout_15_layer_call_and_return_conditional_losses_51800734inputs"©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
’B“
+__inference_dense_66_layer_call_fn_51800743inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
рBн
F__inference_dense_66_layer_call_and_return_conditional_losses_51800753inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
„B‘
-__inference_reshape_22_layer_call_fn_51800758inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 
тBп
H__inference_reshape_22_layer_call_and_return_conditional_losses_51800771inputs"Ш
С≤Н
FullArgSpec
argsЪ

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
annotations™ *
 е
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799124О$%01./89DEBCLMXYVW`almjkz{ЙК:Ґ7
0Ґ-
#К 
Input€€€€€€€€€
p

 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ е
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51799211О$%1.0/89EBDCLMYVXW`amjlkz{ЙК:Ґ7
0Ґ-
#К 
Input€€€€€€€€€
p 

 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ ж
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51800085П$%01./89DEBCLMXYVW`almjkz{ЙК;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ ж
R__inference_Federated_CNN_F5_H12_layer_call_and_return_conditional_losses_51800230П$%1.0/89EBDCLMYVXW`amjlkz{ЙК;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ њ
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799347Г$%01./89DEBCLMXYVW`almjkz{ЙК:Ґ7
0Ґ-
#К 
Input€€€€€€€€€
p

 
™ "%К"
unknown€€€€€€€€€њ
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799482Г$%1.0/89EBDCLMYVXW`amjlkz{ЙК:Ґ7
0Ґ-
#К 
Input€€€€€€€€€
p 

 
™ "%К"
unknown€€€€€€€€€ј
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799816Д$%01./89DEBCLMXYVW`almjkz{ЙК;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ "%К"
unknown€€€€€€€€€ј
7__inference_Federated_CNN_F5_H12_layer_call_fn_51799877Д$%1.0/89EBDCLMYVXW`amjlkz{ЙК;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ "%К"
unknown€€€€€€€€€є
#__inference__wrapped_model_51798581С$%1.0/89EBDCLMYVXW`amjlkz{ЙК2Ґ/
(Ґ%
#К 
Input€€€€€€€€€
™ ";™8
6

reshape_22(К%

reshape_22€€€€€€€€€а
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51800341З01./DҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ а
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_51800361З1.0/DҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p 

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ є
9__inference_batch_normalization_28_layer_call_fn_51800294|01./DҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€є
9__inference_batch_normalization_28_layer_call_fn_51800307|1.0/DҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p 

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€а
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51800446ЗDEBCDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ а
T__inference_batch_normalization_29_layer_call_and_return_conditional_losses_51800466ЗEBDCDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p 

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ є
9__inference_batch_normalization_29_layer_call_fn_51800399|DEBCDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€є
9__inference_batch_normalization_29_layer_call_fn_51800412|EBDCDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p 

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€а
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51800551ЗXYVWDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ а
T__inference_batch_normalization_30_layer_call_and_return_conditional_losses_51800571ЗYVXWDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p 

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ є
9__inference_batch_normalization_30_layer_call_fn_51800504|XYVWDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€є
9__inference_batch_normalization_30_layer_call_fn_51800517|YVXWDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p 

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€а
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51800656ЗlmjkDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ а
T__inference_batch_normalization_31_layer_call_and_return_conditional_losses_51800676ЗmjlkDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p 

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ є
9__inference_batch_normalization_31_layer_call_fn_51800609|lmjkDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€є
9__inference_batch_normalization_31_layer_call_fn_51800622|mjlkDҐA
:Ґ7
-К*
inputs€€€€€€€€€€€€€€€€€€
p 

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€ґ
G__inference_conv1d_28_layer_call_and_return_conditional_losses_51800281k$%3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ Р
,__inference_conv1d_28_layer_call_fn_51800265`$%3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "%К"
unknown€€€€€€€€€ґ
G__inference_conv1d_29_layer_call_and_return_conditional_losses_51800386k893Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ Р
,__inference_conv1d_29_layer_call_fn_51800370`893Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "%К"
unknown€€€€€€€€€ґ
G__inference_conv1d_30_layer_call_and_return_conditional_losses_51800491kLM3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ Р
,__inference_conv1d_30_layer_call_fn_51800475`LM3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "%К"
unknown€€€€€€€€€ґ
G__inference_conv1d_31_layer_call_and_return_conditional_losses_51800596k`a3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ Р
,__inference_conv1d_31_layer_call_fn_51800580``a3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "%К"
unknown€€€€€€€€€≠
F__inference_dense_65_layer_call_and_return_conditional_losses_51800707cz{/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€ 
Ъ З
+__inference_dense_65_layer_call_fn_51800696Xz{/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€ ѓ
F__inference_dense_66_layer_call_and_return_conditional_losses_51800753eЙК/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ ",Ґ)
"К
tensor_0€€€€€€€€€<
Ъ Й
+__inference_dense_66_layer_call_fn_51800743ZЙК/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "!К
unknown€€€€€€€€€<ѓ
H__inference_dropout_15_layer_call_and_return_conditional_losses_51800729c3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p
™ ",Ґ)
"К
tensor_0€€€€€€€€€ 
Ъ ѓ
H__inference_dropout_15_layer_call_and_return_conditional_losses_51800734c3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p 
™ ",Ґ)
"К
tensor_0€€€€€€€€€ 
Ъ Й
-__inference_dropout_15_layer_call_fn_51800712X3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p
™ "!К
unknown€€€€€€€€€ Й
-__inference_dropout_15_layer_call_fn_51800717X3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p 
™ "!К
unknown€€€€€€€€€ а
Y__inference_global_average_pooling1d_14_layer_call_and_return_conditional_losses_51800687ВIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ "5Ґ2
+К(
tensor_0€€€€€€€€€€€€€€€€€€
Ъ є
>__inference_global_average_pooling1d_14_layer_call_fn_51800681wIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ "*К'
unknown€€€€€€€€€€€€€€€€€€є
F__inference_lambda_7_layer_call_and_return_conditional_losses_51800248o;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€

 
p
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ є
F__inference_lambda_7_layer_call_and_return_conditional_losses_51800256o;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€

 
p 
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ У
+__inference_lambda_7_layer_call_fn_51800235d;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€

 
p
™ "%К"
unknown€€€€€€€€€У
+__inference_lambda_7_layer_call_fn_51800240d;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€

 
p 
™ "%К"
unknown€€€€€€€€€ѓ
H__inference_reshape_22_layer_call_and_return_conditional_losses_51800771c/Ґ,
%Ґ"
 К
inputs€€€€€€€€€<
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ Й
-__inference_reshape_22_layer_call_fn_51800758X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€<
™ "%К"
unknown€€€€€€€€€≈
&__inference_signature_wrapper_51799755Ъ$%1.0/89EBDCLMYVXW`amjlkz{ЙК;Ґ8
Ґ 
1™.
,
Input#К 
input€€€€€€€€€";™8
6

reshape_22(К%

reshape_22€€€€€€€€€