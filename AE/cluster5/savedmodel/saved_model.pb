 
×ˇ
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.13.12
b'unknown'8˙
d
XPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
W
random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
W
random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
y
random_uniform/RandomUniformRandomUniformrandom_uniform/shape*
T0*
dtype0*
_output_shapes

:
b
random_uniform/subSubrandom_uniform/maxrandom_uniform/min*
T0*
_output_shapes
: 
t
random_uniform/mulMulrandom_uniform/RandomUniformrandom_uniform/sub*
T0*
_output_shapes

:
f
random_uniformAddrandom_uniform/mulrandom_uniform/min*
T0*
_output_shapes

:
Q
W
VariableV2*
shape
:*
dtype0*
_output_shapes

:
d
W/AssignAssignWrandom_uniform*
T0*
_class

loc:@W*
_output_shapes

:
T
W/readIdentityW*
T0*
_class

loc:@W*
_output_shapes

:
R
zerosConst*
valueB*    *
dtype0*
_output_shapes
:
I
b
VariableV2*
shape:*
dtype0*
_output_shapes
:
W
b/AssignAssignbzeros*
T0*
_class

loc:@b*
_output_shapes
:
P
b/readIdentityb*
T0*
_class

loc:@b*
_output_shapes
:
_
transpose/permConst*
valueB"       *
dtype0*
_output_shapes
:
W
	transpose	TransposeW/readtranspose/perm*
T0*
_output_shapes

:
T
zeros_1Const*
valueB*    *
dtype0*
_output_shapes
:
O
b_prime
VariableV2*
shape:*
dtype0*
_output_shapes
:
k
b_prime/AssignAssignb_primezeros_1*
T0*
_class
loc:@b_prime*
_output_shapes
:
b
b_prime/readIdentityb_prime*
T0*
_class
loc:@b_prime*
_output_shapes
:
M
MatMulMatMulXW/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
L
addAddMatMulb/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
I
SigmoidSigmoidadd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
MatMul_1MatMulSigmoid	transpose*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
add_1AddMatMul_1b_prime/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
M
	Sigmoid_1Sigmoidadd_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
subSubX	Sigmoid_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
Pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
H
PowPowsubPow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
7
SumSumPowConst*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
]
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*
_output_shapes
: 
q
 gradients/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

gradients/Sum_grad/ReshapeReshapegradients/Fill gradients/Sum_grad/Reshape/shape*
T0*
_output_shapes

:
K
gradients/Sum_grad/ShapeShapePow*
T0*
_output_shapes
:

gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
K
gradients/Pow_grad/ShapeShapesub*
T0*
_output_shapes
:
]
gradients/Pow_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ť
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
o
gradients/Pow_grad/mulMulgradients/Sum_grad/TilePow/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
gradients/Pow_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
gradients/Pow_grad/subSubPow/ygradients/Pow_grad/sub/y*
T0*
_output_shapes
: 
l
gradients/Pow_grad/PowPowsubgradients/Pow_grad/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*
T0*
_output_shapes
:

gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients/Pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
gradients/Pow_grad/GreaterGreatersubgradients/Pow_grad/Greater/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
"gradients/Pow_grad/ones_like/ShapeShapesub*
T0*
_output_shapes
:
g
"gradients/Pow_grad/ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

gradients/Pow_grad/ones_likeFill"gradients/Pow_grad/ones_like/Shape"gradients/Pow_grad/ones_like/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatersubgradients/Pow_grad/ones_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
gradients/Pow_grad/LogLoggradients/Pow_grad/Select*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
gradients/Pow_grad/zeros_like	ZerosLikesub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
gradients/Pow_grad/Select_1Selectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
gradients/Pow_grad/mul_2Mulgradients/Sum_grad/TilePow*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
~
gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
T0*
_output_shapes
: 
g
#gradients/Pow_grad/tuple/group_depsNoOp^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
Ú
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Pow_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
-gradients/Pow_grad/tuple/control_dependency_1Identitygradients/Pow_grad/Reshape_1$^gradients/Pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_grad/Reshape_1*
_output_shapes
: 
I
gradients/sub_grad/ShapeShapeX*
T0*
_output_shapes
:
S
gradients/sub_grad/Shape_1Shape	Sigmoid_1*
T0*
_output_shapes
:
Ť
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/sub_grad/Sum_1Sum+gradients/Pow_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
Ú
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

$gradients/Sigmoid_1_grad/SigmoidGradSigmoidGrad	Sigmoid_1-gradients/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
gradients/add_1_grad/ShapeShapeMatMul_1*
T0*
_output_shapes
:
f
gradients/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ą
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/add_1_grad/SumSum$gradients/Sigmoid_1_grad/SigmoidGrad*gradients/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:

gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/add_1_grad/Sum_1Sum$gradients/Sigmoid_1_grad/SigmoidGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
_output_shapes
:
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
â
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
:
§
gradients/MatMul_1_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependency	transpose*
transpose_b(*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

 gradients/MatMul_1_grad/MatMul_1MatMulSigmoid-gradients/add_1_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
ě
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:

"gradients/Sigmoid_grad/SigmoidGradSigmoidGradSigmoid0gradients/MatMul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
*gradients/transpose_grad/InvertPermutationInvertPermutationtranspose/perm*
_output_shapes
:
¸
"gradients/transpose_grad/transpose	Transpose2gradients/MatMul_1_grad/tuple/control_dependency_1*gradients/transpose_grad/InvertPermutation*
T0*
_output_shapes

:
N
gradients/add_grad/ShapeShapeMatMul*
T0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ť
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/add_grad/SumSum"gradients/Sigmoid_grad/SigmoidGrad(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/add_grad/Sum_1Sum"gradients/Sigmoid_grad/SigmoidGrad*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
_output_shapes
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Ú
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:
 
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyW/read*
transpose_b(*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/MatMul_grad/MatMul_1MatMulX+gradients/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ä
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:
Ő
gradients/AddNAddN"gradients/transpose_grad/transpose0gradients/MatMul_grad/tuple/control_dependency_1*
T0*5
_class+
)'loc:@gradients/transpose_grad/transpose*
N*
_output_shapes

:
t
beta1_power/initial_valueConst*
_class

loc:@W*
valueB
 *fff?*
dtype0*
_output_shapes
: 
a
beta1_power
VariableV2*
shape: *
_class

loc:@W*
dtype0*
_output_shapes
: 
{
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_class

loc:@W*
_output_shapes
: 
`
beta1_power/readIdentitybeta1_power*
T0*
_class

loc:@W*
_output_shapes
: 
t
beta2_power/initial_valueConst*
_class

loc:@W*
valueB
 *wž?*
dtype0*
_output_shapes
: 
a
beta2_power
VariableV2*
shape: *
_class

loc:@W*
dtype0*
_output_shapes
: 
{
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_class

loc:@W*
_output_shapes
: 
`
beta2_power/readIdentitybeta2_power*
T0*
_class

loc:@W*
_output_shapes
: 

W/Adam/Initializer/zerosConst*
_class

loc:@W*
valueB*    *
dtype0*
_output_shapes

:
l
W/Adam
VariableV2*
shape
:*
_class

loc:@W*
dtype0*
_output_shapes

:
x
W/Adam/AssignAssignW/AdamW/Adam/Initializer/zeros*
T0*
_class

loc:@W*
_output_shapes

:
^
W/Adam/readIdentityW/Adam*
T0*
_class

loc:@W*
_output_shapes

:

W/Adam_1/Initializer/zerosConst*
_class

loc:@W*
valueB*    *
dtype0*
_output_shapes

:
n
W/Adam_1
VariableV2*
shape
:*
_class

loc:@W*
dtype0*
_output_shapes

:
~
W/Adam_1/AssignAssignW/Adam_1W/Adam_1/Initializer/zeros*
T0*
_class

loc:@W*
_output_shapes

:
b
W/Adam_1/readIdentityW/Adam_1*
T0*
_class

loc:@W*
_output_shapes

:
{
b/Adam/Initializer/zerosConst*
_class

loc:@b*
valueB*    *
dtype0*
_output_shapes
:
d
b/Adam
VariableV2*
shape:*
_class

loc:@b*
dtype0*
_output_shapes
:
t
b/Adam/AssignAssignb/Adamb/Adam/Initializer/zeros*
T0*
_class

loc:@b*
_output_shapes
:
Z
b/Adam/readIdentityb/Adam*
T0*
_class

loc:@b*
_output_shapes
:
}
b/Adam_1/Initializer/zerosConst*
_class

loc:@b*
valueB*    *
dtype0*
_output_shapes
:
f
b/Adam_1
VariableV2*
shape:*
_class

loc:@b*
dtype0*
_output_shapes
:
z
b/Adam_1/AssignAssignb/Adam_1b/Adam_1/Initializer/zeros*
T0*
_class

loc:@b*
_output_shapes
:
^
b/Adam_1/readIdentityb/Adam_1*
T0*
_class

loc:@b*
_output_shapes
:

b_prime/Adam/Initializer/zerosConst*
_class
loc:@b_prime*
valueB*    *
dtype0*
_output_shapes
:
p
b_prime/Adam
VariableV2*
shape:*
_class
loc:@b_prime*
dtype0*
_output_shapes
:

b_prime/Adam/AssignAssignb_prime/Adamb_prime/Adam/Initializer/zeros*
T0*
_class
loc:@b_prime*
_output_shapes
:
l
b_prime/Adam/readIdentityb_prime/Adam*
T0*
_class
loc:@b_prime*
_output_shapes
:

 b_prime/Adam_1/Initializer/zerosConst*
_class
loc:@b_prime*
valueB*    *
dtype0*
_output_shapes
:
r
b_prime/Adam_1
VariableV2*
shape:*
_class
loc:@b_prime*
dtype0*
_output_shapes
:

b_prime/Adam_1/AssignAssignb_prime/Adam_1 b_prime/Adam_1/Initializer/zeros*
T0*
_class
loc:@b_prime*
_output_shapes
:
p
b_prime/Adam_1/readIdentityb_prime/Adam_1*
T0*
_class
loc:@b_prime*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *Âu<*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
ć
Adam/update_W/ApplyAdam	ApplyAdamWW/AdamW/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
T0*
_class

loc:@W*
_output_shapes

:

Adam/update_b/ApplyAdam	ApplyAdambb/Adamb/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
_class

loc:@b*
_output_shapes
:
Ą
Adam/update_b_prime/ApplyAdam	ApplyAdamb_primeb_prime/Adamb_prime/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
T0*
_class
loc:@b_prime*
_output_shapes
:
¸
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_W/ApplyAdam^Adam/update_b/ApplyAdam^Adam/update_b_prime/ApplyAdam*
T0*
_class

loc:@W*
_output_shapes
: 
v
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class

loc:@W*
_output_shapes
: 
ş

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_W/ApplyAdam^Adam/update_b/ApplyAdam^Adam/update_b_prime/ApplyAdam*
T0*
_class

loc:@W*
_output_shapes
: 
z
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class

loc:@W*
_output_shapes
: 
~
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_W/ApplyAdam^Adam/update_b/ApplyAdam^Adam/update_b_prime/ApplyAdam
Ď
initNoOp^W/Adam/Assign^W/Adam_1/Assign	^W/Assign^b/Adam/Assign^b/Adam_1/Assign	^b/Assign^b_prime/Adam/Assign^b_prime/Adam_1/Assign^b_prime/Assign^beta1_power/Assign^beta2_power/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
Ę
save/SaveV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
y
save/SaveV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
ß
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesWW/AdamW/Adam_1bb/Adamb/Adam_1b_primeb_prime/Adamb_prime/Adam_1beta1_powerbeta2_power*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
Í
save/RestoreV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
|
save/RestoreV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
Â
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*@
_output_shapes.
,:::::::::::
g
save/AssignAssignWsave/RestoreV2*
T0*
_class

loc:@W*
_output_shapes

:
p
save/Assign_1AssignW/Adamsave/RestoreV2:1*
T0*
_class

loc:@W*
_output_shapes

:
r
save/Assign_2AssignW/Adam_1save/RestoreV2:2*
T0*
_class

loc:@W*
_output_shapes

:
g
save/Assign_3Assignbsave/RestoreV2:3*
T0*
_class

loc:@b*
_output_shapes
:
l
save/Assign_4Assignb/Adamsave/RestoreV2:4*
T0*
_class

loc:@b*
_output_shapes
:
n
save/Assign_5Assignb/Adam_1save/RestoreV2:5*
T0*
_class

loc:@b*
_output_shapes
:
s
save/Assign_6Assignb_primesave/RestoreV2:6*
T0*
_class
loc:@b_prime*
_output_shapes
:
x
save/Assign_7Assignb_prime/Adamsave/RestoreV2:7*
T0*
_class
loc:@b_prime*
_output_shapes
:
z
save/Assign_8Assignb_prime/Adam_1save/RestoreV2:8*
T0*
_class
loc:@b_prime*
_output_shapes
:
m
save/Assign_9Assignbeta1_powersave/RestoreV2:9*
T0*
_class

loc:@W*
_output_shapes
: 
o
save/Assign_10Assignbeta2_powersave/RestoreV2:10*
T0*
_class

loc:@W*
_output_shapes
: 
Ç
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
M
PREDIdentity	Sigmoid_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
COSTIdentitySum*
T0*
_output_shapes
: 
Ń
init_1NoOp^W/Adam/Assign^W/Adam_1/Assign	^W/Assign^b/Adam/Assign^b/Adam_1/Assign	^b/Assign^b_prime/Adam/Assign^b_prime/Adam_1/Assign^b_prime/Assign^beta1_power/Assign^beta2_power/Assign
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
shape: *
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_66c20e9f8ee24147aaf14862326d700c/part*
dtype0*
_output_shapes
: 
j
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
Ě
save_1/SaveV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
{
save_1/SaveV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
ń
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesWW/AdamW/Adam_1bb/Adamb/Adam_1b_primeb_prime/Adamb_prime/Adam_1beta1_powerbeta2_power*
dtypes
2

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 

-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*
N*
_output_shapes
:
l
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
Ď
save_1/RestoreV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
~
!save_1/RestoreV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
Ę
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
dtypes
2*@
_output_shapes.
,:::::::::::
k
save_1/AssignAssignWsave_1/RestoreV2*
T0*
_class

loc:@W*
_output_shapes

:
t
save_1/Assign_1AssignW/Adamsave_1/RestoreV2:1*
T0*
_class

loc:@W*
_output_shapes

:
v
save_1/Assign_2AssignW/Adam_1save_1/RestoreV2:2*
T0*
_class

loc:@W*
_output_shapes

:
k
save_1/Assign_3Assignbsave_1/RestoreV2:3*
T0*
_class

loc:@b*
_output_shapes
:
p
save_1/Assign_4Assignb/Adamsave_1/RestoreV2:4*
T0*
_class

loc:@b*
_output_shapes
:
r
save_1/Assign_5Assignb/Adam_1save_1/RestoreV2:5*
T0*
_class

loc:@b*
_output_shapes
:
w
save_1/Assign_6Assignb_primesave_1/RestoreV2:6*
T0*
_class
loc:@b_prime*
_output_shapes
:
|
save_1/Assign_7Assignb_prime/Adamsave_1/RestoreV2:7*
T0*
_class
loc:@b_prime*
_output_shapes
:
~
save_1/Assign_8Assignb_prime/Adam_1save_1/RestoreV2:8*
T0*
_class
loc:@b_prime*
_output_shapes
:
q
save_1/Assign_9Assignbeta1_powersave_1/RestoreV2:9*
T0*
_class

loc:@W*
_output_shapes
: 
s
save_1/Assign_10Assignbeta2_powersave_1/RestoreV2:10*
T0*
_class

loc:@W*
_output_shapes
: 
á
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
Ń
init_2NoOp^W/Adam/Assign^W/Adam_1/Assign	^W/Assign^b/Adam/Assign^b/Adam_1/Assign	^b/Assign^b_prime/Adam/Assign^b_prime/Adam_1/Assign^b_prime/Assign^beta1_power/Assign^beta2_power/Assign
[
save_2/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
shape: *
dtype0*
_output_shapes
: 

save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_9148363787424ebe8f5899dd8119c416/part*
dtype0*
_output_shapes
: 
j
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
_output_shapes
: 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
Ě
save_2/SaveV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
{
save_2/SaveV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
ń
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesWW/AdamW/Adam_1bb/Adamb/Adam_1b_primeb_prime/Adamb_prime/Adam_1beta1_powerbeta2_power*
dtypes
2

save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 

-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
T0*
N*
_output_shapes
:
l
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const

save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
Ď
save_2/RestoreV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
~
!save_2/RestoreV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
Ę
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*
dtypes
2*@
_output_shapes.
,:::::::::::
k
save_2/AssignAssignWsave_2/RestoreV2*
T0*
_class

loc:@W*
_output_shapes

:
t
save_2/Assign_1AssignW/Adamsave_2/RestoreV2:1*
T0*
_class

loc:@W*
_output_shapes

:
v
save_2/Assign_2AssignW/Adam_1save_2/RestoreV2:2*
T0*
_class

loc:@W*
_output_shapes

:
k
save_2/Assign_3Assignbsave_2/RestoreV2:3*
T0*
_class

loc:@b*
_output_shapes
:
p
save_2/Assign_4Assignb/Adamsave_2/RestoreV2:4*
T0*
_class

loc:@b*
_output_shapes
:
r
save_2/Assign_5Assignb/Adam_1save_2/RestoreV2:5*
T0*
_class

loc:@b*
_output_shapes
:
w
save_2/Assign_6Assignb_primesave_2/RestoreV2:6*
T0*
_class
loc:@b_prime*
_output_shapes
:
|
save_2/Assign_7Assignb_prime/Adamsave_2/RestoreV2:7*
T0*
_class
loc:@b_prime*
_output_shapes
:
~
save_2/Assign_8Assignb_prime/Adam_1save_2/RestoreV2:8*
T0*
_class
loc:@b_prime*
_output_shapes
:
q
save_2/Assign_9Assignbeta1_powersave_2/RestoreV2:9*
T0*
_class

loc:@W*
_output_shapes
: 
s
save_2/Assign_10Assignbeta2_powersave_2/RestoreV2:10*
T0*
_class

loc:@W*
_output_shapes
: 
á
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_2^save_2/Assign_3^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
Ń
init_3NoOp^W/Adam/Assign^W/Adam_1/Assign	^W/Assign^b/Adam/Assign^b/Adam_1/Assign	^b/Assign^b_prime/Adam/Assign^b_prime/Adam_1/Assign^b_prime/Assign^beta1_power/Assign^beta2_power/Assign
[
save_3/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
shape: *
dtype0*
_output_shapes
: 

save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_7f641820f02b44cb812318f93ff0e39d/part*
dtype0*
_output_shapes
: 
j
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
N*
_output_shapes
: 
S
save_3/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_3/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
Ě
save_3/SaveV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
{
save_3/SaveV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
ń
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesWW/AdamW/Adam_1bb/Adamb/Adam_1b_primeb_prime/Adamb_prime/Adam_1beta1_powerbeta2_power*
dtypes
2

save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: 

-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
T0*
N*
_output_shapes
:
l
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const

save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
Ď
save_3/RestoreV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
~
!save_3/RestoreV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
Ę
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*
dtypes
2*@
_output_shapes.
,:::::::::::
k
save_3/AssignAssignWsave_3/RestoreV2*
T0*
_class

loc:@W*
_output_shapes

:
t
save_3/Assign_1AssignW/Adamsave_3/RestoreV2:1*
T0*
_class

loc:@W*
_output_shapes

:
v
save_3/Assign_2AssignW/Adam_1save_3/RestoreV2:2*
T0*
_class

loc:@W*
_output_shapes

:
k
save_3/Assign_3Assignbsave_3/RestoreV2:3*
T0*
_class

loc:@b*
_output_shapes
:
p
save_3/Assign_4Assignb/Adamsave_3/RestoreV2:4*
T0*
_class

loc:@b*
_output_shapes
:
r
save_3/Assign_5Assignb/Adam_1save_3/RestoreV2:5*
T0*
_class

loc:@b*
_output_shapes
:
w
save_3/Assign_6Assignb_primesave_3/RestoreV2:6*
T0*
_class
loc:@b_prime*
_output_shapes
:
|
save_3/Assign_7Assignb_prime/Adamsave_3/RestoreV2:7*
T0*
_class
loc:@b_prime*
_output_shapes
:
~
save_3/Assign_8Assignb_prime/Adam_1save_3/RestoreV2:8*
T0*
_class
loc:@b_prime*
_output_shapes
:
q
save_3/Assign_9Assignbeta1_powersave_3/RestoreV2:9*
T0*
_class

loc:@W*
_output_shapes
: 
s
save_3/Assign_10Assignbeta2_powersave_3/RestoreV2:10*
T0*
_class

loc:@W*
_output_shapes
: 
á
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_2^save_3/Assign_3^save_3/Assign_4^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
Ń
init_4NoOp^W/Adam/Assign^W/Adam_1/Assign	^W/Assign^b/Adam/Assign^b/Adam_1/Assign	^b/Assign^b_prime/Adam/Assign^b_prime/Adam_1/Assign^b_prime/Assign^beta1_power/Assign^beta2_power/Assign
[
save_4/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
shape: *
dtype0*
_output_shapes
: 

save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_68886453c0614c7fb876c705fd081e39/part*
dtype0*
_output_shapes
: 
j
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
_output_shapes
: 
S
save_4/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_4/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
Ě
save_4/SaveV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
{
save_4/SaveV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
ń
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesWW/AdamW/Adam_1bb/Adamb/Adam_1b_primeb_prime/Adamb_prime/Adam_1beta1_powerbeta2_power*
dtypes
2

save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*)
_class
loc:@save_4/ShardedFilename*
_output_shapes
: 

-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
T0*
N*
_output_shapes
:
l
save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const

save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
T0*
_output_shapes
: 
Ď
save_4/RestoreV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
~
!save_4/RestoreV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
Ę
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*
dtypes
2*@
_output_shapes.
,:::::::::::
k
save_4/AssignAssignWsave_4/RestoreV2*
T0*
_class

loc:@W*
_output_shapes

:
t
save_4/Assign_1AssignW/Adamsave_4/RestoreV2:1*
T0*
_class

loc:@W*
_output_shapes

:
v
save_4/Assign_2AssignW/Adam_1save_4/RestoreV2:2*
T0*
_class

loc:@W*
_output_shapes

:
k
save_4/Assign_3Assignbsave_4/RestoreV2:3*
T0*
_class

loc:@b*
_output_shapes
:
p
save_4/Assign_4Assignb/Adamsave_4/RestoreV2:4*
T0*
_class

loc:@b*
_output_shapes
:
r
save_4/Assign_5Assignb/Adam_1save_4/RestoreV2:5*
T0*
_class

loc:@b*
_output_shapes
:
w
save_4/Assign_6Assignb_primesave_4/RestoreV2:6*
T0*
_class
loc:@b_prime*
_output_shapes
:
|
save_4/Assign_7Assignb_prime/Adamsave_4/RestoreV2:7*
T0*
_class
loc:@b_prime*
_output_shapes
:
~
save_4/Assign_8Assignb_prime/Adam_1save_4/RestoreV2:8*
T0*
_class
loc:@b_prime*
_output_shapes
:
q
save_4/Assign_9Assignbeta1_powersave_4/RestoreV2:9*
T0*
_class

loc:@W*
_output_shapes
: 
s
save_4/Assign_10Assignbeta2_powersave_4/RestoreV2:10*
T0*
_class

loc:@W*
_output_shapes
: 
á
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_2^save_4/Assign_3^save_4/Assign_4^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
Ń
init_5NoOp^W/Adam/Assign^W/Adam_1/Assign	^W/Assign^b/Adam/Assign^b/Adam_1/Assign	^b/Assign^b_prime/Adam/Assign^b_prime/Adam_1/Assign^b_prime/Assign^beta1_power/Assign^beta2_power/Assign
[
save_5/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
shape: *
dtype0*
_output_shapes
: 

save_5/StringJoin/inputs_1Const*<
value3B1 B+_temp_44362d2246f446a1a9575c70aa408274/part*
dtype0*
_output_shapes
: 
j
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
_output_shapes
: 
S
save_5/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_5/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
Ě
save_5/SaveV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
{
save_5/SaveV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
ń
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesWW/AdamW/Adam_1bb/Adamb/Adam_1b_primeb_prime/Adamb_prime/Adam_1beta1_powerbeta2_power*
dtypes
2

save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
T0*)
_class
loc:@save_5/ShardedFilename*
_output_shapes
: 

-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
T0*
N*
_output_shapes
:
l
save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const

save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
Ď
save_5/RestoreV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
~
!save_5/RestoreV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
Ę
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*
dtypes
2*@
_output_shapes.
,:::::::::::
k
save_5/AssignAssignWsave_5/RestoreV2*
T0*
_class

loc:@W*
_output_shapes

:
t
save_5/Assign_1AssignW/Adamsave_5/RestoreV2:1*
T0*
_class

loc:@W*
_output_shapes

:
v
save_5/Assign_2AssignW/Adam_1save_5/RestoreV2:2*
T0*
_class

loc:@W*
_output_shapes

:
k
save_5/Assign_3Assignbsave_5/RestoreV2:3*
T0*
_class

loc:@b*
_output_shapes
:
p
save_5/Assign_4Assignb/Adamsave_5/RestoreV2:4*
T0*
_class

loc:@b*
_output_shapes
:
r
save_5/Assign_5Assignb/Adam_1save_5/RestoreV2:5*
T0*
_class

loc:@b*
_output_shapes
:
w
save_5/Assign_6Assignb_primesave_5/RestoreV2:6*
T0*
_class
loc:@b_prime*
_output_shapes
:
|
save_5/Assign_7Assignb_prime/Adamsave_5/RestoreV2:7*
T0*
_class
loc:@b_prime*
_output_shapes
:
~
save_5/Assign_8Assignb_prime/Adam_1save_5/RestoreV2:8*
T0*
_class
loc:@b_prime*
_output_shapes
:
q
save_5/Assign_9Assignbeta1_powersave_5/RestoreV2:9*
T0*
_class

loc:@W*
_output_shapes
: 
s
save_5/Assign_10Assignbeta2_powersave_5/RestoreV2:10*
T0*
_class

loc:@W*
_output_shapes
: 
á
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_2^save_5/Assign_3^save_5/Assign_4^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
Ń
init_6NoOp^W/Adam/Assign^W/Adam_1/Assign	^W/Assign^b/Adam/Assign^b/Adam_1/Assign	^b/Assign^b_prime/Adam/Assign^b_prime/Adam_1/Assign^b_prime/Assign^beta1_power/Assign^beta2_power/Assign
[
save_6/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
shape: *
dtype0*
_output_shapes
: 

save_6/StringJoin/inputs_1Const*<
value3B1 B+_temp_e8dbafd9cbac454ca1e1a46a82d7f489/part*
dtype0*
_output_shapes
: 
j
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
N*
_output_shapes
: 
S
save_6/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
Ě
save_6/SaveV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
{
save_6/SaveV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
ń
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesWW/AdamW/Adam_1bb/Adamb/Adam_1b_primeb_prime/Adamb_prime/Adam_1beta1_powerbeta2_power*
dtypes
2

save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
T0*)
_class
loc:@save_6/ShardedFilename*
_output_shapes
: 

-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
T0*
N*
_output_shapes
:
l
save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const

save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
T0*
_output_shapes
: 
Ď
save_6/RestoreV2/tensor_namesConst*~
valueuBsBWBW/AdamBW/Adam_1BbBb/AdamBb/Adam_1Bb_primeBb_prime/AdamBb_prime/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
~
!save_6/RestoreV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0*
_output_shapes
:
Ę
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*
dtypes
2*@
_output_shapes.
,:::::::::::
k
save_6/AssignAssignWsave_6/RestoreV2*
T0*
_class

loc:@W*
_output_shapes

:
t
save_6/Assign_1AssignW/Adamsave_6/RestoreV2:1*
T0*
_class

loc:@W*
_output_shapes

:
v
save_6/Assign_2AssignW/Adam_1save_6/RestoreV2:2*
T0*
_class

loc:@W*
_output_shapes

:
k
save_6/Assign_3Assignbsave_6/RestoreV2:3*
T0*
_class

loc:@b*
_output_shapes
:
p
save_6/Assign_4Assignb/Adamsave_6/RestoreV2:4*
T0*
_class

loc:@b*
_output_shapes
:
r
save_6/Assign_5Assignb/Adam_1save_6/RestoreV2:5*
T0*
_class

loc:@b*
_output_shapes
:
w
save_6/Assign_6Assignb_primesave_6/RestoreV2:6*
T0*
_class
loc:@b_prime*
_output_shapes
:
|
save_6/Assign_7Assignb_prime/Adamsave_6/RestoreV2:7*
T0*
_class
loc:@b_prime*
_output_shapes
:
~
save_6/Assign_8Assignb_prime/Adam_1save_6/RestoreV2:8*
T0*
_class
loc:@b_prime*
_output_shapes
:
q
save_6/Assign_9Assignbeta1_powersave_6/RestoreV2:9*
T0*
_class

loc:@W*
_output_shapes
: 
s
save_6/Assign_10Assignbeta2_powersave_6/RestoreV2:10*
T0*
_class

loc:@W*
_output_shapes
: 
á
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_2^save_6/Assign_3^save_6/Assign_4^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard "B
save_6/Const:0save_6/Identity:0save_6/restore_all (5 @F8"Ş
trainable_variables
-
W:0W/AssignW/read:02random_uniform:08
$
b:0b/Assignb/read:02zeros:08
8
	b_prime:0b_prime/Assignb_prime/read:02	zeros_1:08"
train_op

Adam"¸
	variablesŞ§
-
W:0W/AssignW/read:02random_uniform:08
$
b:0b/Assignb/read:02zeros:08
8
	b_prime:0b_prime/Assignb_prime/read:02	zeros_1:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
D
W/Adam:0W/Adam/AssignW/Adam/read:02W/Adam/Initializer/zeros:0
L

W/Adam_1:0W/Adam_1/AssignW/Adam_1/read:02W/Adam_1/Initializer/zeros:0
D
b/Adam:0b/Adam/Assignb/Adam/read:02b/Adam/Initializer/zeros:0
L

b/Adam_1:0b/Adam_1/Assignb/Adam_1/read:02b/Adam_1/Initializer/zeros:0
\
b_prime/Adam:0b_prime/Adam/Assignb_prime/Adam/read:02 b_prime/Adam/Initializer/zeros:0
d
b_prime/Adam_1:0b_prime/Adam_1/Assignb_prime/Adam_1/read:02"b_prime/Adam_1/Initializer/zeros:0*m
serving_defaultZ
$
inputs
X:0˙˙˙˙˙˙˙˙˙
outputs
Sum:0 tensorflow/serving/predict