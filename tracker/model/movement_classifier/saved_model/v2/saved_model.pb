Ни
х(М(
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
ь
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ж
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

Ы

DecodeJpeg
contents	
image"
channelsint "
ratioint"
fancy_upscalingbool("!
try_recover_truncatedbool( "#
acceptable_fractionfloat%  ?"

dct_methodstring 
;
Elu
features"T
activations"T"
Ttype:
2
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

)
Exit	
data"T
output"T"	
Ttype
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
2
L2Loss
t"T
output"T"
Ttype:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z

!
LoopCond	
input


output

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
д
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Р
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
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
2
NextIteration	
data"T
output"T"	
Ttype
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
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
q
ResizeBilinear
images"T
size
resized_images"
Ttype:
2
	"
align_cornersbool( 
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
о
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.13.12v1.13.1-0-g6612da8951Пб

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

global_step
VariableV2*
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container 
В
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
f
PlaceholderPlaceholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
Placeholder_3Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
h
Placeholder_2Placeholder*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
h
Placeholder_5Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
Placeholder_4Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
h
Placeholder_7Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
h
Placeholder_6Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
Placeholder_9Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
Placeholder_8Placeholder*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
i
Placeholder_11Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
i
Placeholder_10Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
i
Placeholder_13Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
i
Placeholder_12Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
i
Placeholder_15Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
i
Placeholder_14Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
T
	map/ShapeShapePlaceholder*
T0*
out_type0*
_output_shapes
:
a
map/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
c
map/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
c
map/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

map/strided_sliceStridedSlice	map/Shapemap/strided_slice/stackmap/strided_slice/stack_1map/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
к
map/TensorArrayTensorArrayV3map/strided_slice*
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: 
g
map/TensorArrayUnstack/ShapeShapePlaceholder*
_output_shapes
:*
T0*
out_type0
t
*map/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
v
,map/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
v
,map/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ь
$map/TensorArrayUnstack/strided_sliceStridedSlicemap/TensorArrayUnstack/Shape*map/TensorArrayUnstack/strided_slice/stack,map/TensorArrayUnstack/strided_slice/stack_1,map/TensorArrayUnstack/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
d
"map/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"map/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ф
map/TensorArrayUnstack/rangeRange"map/TensorArrayUnstack/range/start$map/TensorArrayUnstack/strided_slice"map/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0
ц
>map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map/TensorArraymap/TensorArrayUnstack/rangePlaceholdermap/TensorArray:1*
T0*
_class
loc:@Placeholder*
_output_shapes
: 
K
	map/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
м
map/TensorArray_1TensorArrayV3map/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
]
map/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
­
map/while/EnterEntermap/while/iteration_counter*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *'

frame_namemap/while/while_context

map/while/Enter_1Enter	map/Const*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *'

frame_namemap/while/while_context
Ї
map/while/Enter_2Entermap/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *'

frame_namemap/while/while_context
n
map/while/MergeMergemap/while/Entermap/while/NextIteration*
T0*
N*
_output_shapes
: : 
t
map/while/Merge_1Mergemap/while/Enter_1map/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
t
map/while/Merge_2Mergemap/while/Enter_2map/while/NextIteration_2*
_output_shapes
: : *
T0*
N
^
map/while/LessLessmap/while/Mergemap/while/Less/Enter*
T0*
_output_shapes
: 
Ј
map/while/Less/EnterEntermap/strided_slice*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *'

frame_namemap/while/while_context
b
map/while/Less_1Lessmap/while/Merge_1map/while/Less/Enter*
_output_shapes
: *
T0
\
map/while/LogicalAnd
LogicalAndmap/while/Lessmap/while/Less_1*
_output_shapes
: 
L
map/while/LoopCondLoopCondmap/while/LogicalAnd*
_output_shapes
: 

map/while/SwitchSwitchmap/while/Mergemap/while/LoopCond*
T0*"
_class
loc:@map/while/Merge*
_output_shapes
: : 

map/while/Switch_1Switchmap/while/Merge_1map/while/LoopCond*
T0*$
_class
loc:@map/while/Merge_1*
_output_shapes
: : 

map/while/Switch_2Switchmap/while/Merge_2map/while/LoopCond*
T0*$
_class
loc:@map/while/Merge_2*
_output_shapes
: : 
S
map/while/IdentityIdentitymap/while/Switch:1*
T0*
_output_shapes
: 
W
map/while/Identity_1Identitymap/while/Switch_1:1*
T0*
_output_shapes
: 
W
map/while/Identity_2Identitymap/while/Switch_2:1*
_output_shapes
: *
T0
f
map/while/add/yConst^map/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Z
map/while/addAddmap/while/Identitymap/while/add/y*
_output_shapes
: *
T0
Г
map/while/TensorArrayReadV3TensorArrayReadV3!map/while/TensorArrayReadV3/Entermap/while/Identity_1#map/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
З
!map/while/TensorArrayReadV3/EnterEntermap/TensorArray*
_output_shapes
:*'

frame_namemap/while/while_context*
T0*
is_constant(*
parallel_iterations

ф
#map/while/TensorArrayReadV3/Enter_1Enter>map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *'

frame_namemap/while/while_context
і
map/while/DecodeJpeg
DecodeJpegmap/while/TensorArrayReadV3*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( 
v
map/while/resize/ExpandDims/dimConst^map/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
Џ
map/while/resize/ExpandDims
ExpandDimsmap/while/DecodeJpegmap/while/resize/ExpandDims/dim*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*

Tdim0*
T0
{
map/while/resize/sizeConst^map/while/Identity*
_output_shapes
:*
valueB",  ,  *
dtype0
­
map/while/resize/ResizeBilinearResizeBilinearmap/while/resize/ExpandDimsmap/while/resize/size*(
_output_shapes
:ЌЌ*
align_corners( *
T0

map/while/resize/SqueezeSqueezemap/while/resize/ResizeBilinear*
squeeze_dims
 *
T0*$
_output_shapes
:ЌЌ
i
map/while/div/yConst^map/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  C
r
map/while/divRealDivmap/while/resize/Squeezemap/while/div/y*$
_output_shapes
:ЌЌ*
T0

map/while/Reshape_Preproc/shapeConst^map/while/Identity*
dtype0*
_output_shapes
:*!
valueB",  ,     

map/while/Reshape_PreprocReshapemap/while/divmap/while/Reshape_Preproc/shape*
T0*
Tshape0*$
_output_shapes
:ЌЌ

-map/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33map/while/TensorArrayWrite/TensorArrayWriteV3/Entermap/while/Identity_1map/while/Reshape_Preprocmap/while/Identity_2*
T0*,
_class"
 loc:@map/while/Reshape_Preproc*
_output_shapes
: 
љ
3map/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap/TensorArray_1*
is_constant(*
_output_shapes
:*'

frame_namemap/while/while_context*
T0*,
_class"
 loc:@map/while/Reshape_Preproc*
parallel_iterations

h
map/while/add_1/yConst^map/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
map/while/add_1Addmap/while/Identity_1map/while/add_1/y*
_output_shapes
: *
T0
X
map/while/NextIterationNextIterationmap/while/add*
T0*
_output_shapes
: 
\
map/while/NextIteration_1NextIterationmap/while/add_1*
_output_shapes
: *
T0
z
map/while/NextIteration_2NextIteration-map/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
I
map/while/ExitExitmap/while/Switch*
T0*
_output_shapes
: 
M
map/while/Exit_1Exitmap/while/Switch_1*
_output_shapes
: *
T0
M
map/while/Exit_2Exitmap/while/Switch_2*
_output_shapes
: *
T0

&map/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map/TensorArray_1map/while/Exit_2*
_output_shapes
: *$
_class
loc:@map/TensorArray_1

 map/TensorArrayStack/range/startConst*
_output_shapes
: *
value	B : *$
_class
loc:@map/TensorArray_1*
dtype0

 map/TensorArrayStack/range/deltaConst*
value	B :*$
_class
loc:@map/TensorArray_1*
dtype0*
_output_shapes
: 
ц
map/TensorArrayStack/rangeRange map/TensorArrayStack/range/start&map/TensorArrayStack/TensorArraySizeV3 map/TensorArrayStack/range/delta*$
_class
loc:@map/TensorArray_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0

(map/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map/TensorArray_1map/TensorArrayStack/rangemap/while/Exit_2*!
element_shape:ЌЌ*$
_class
loc:@map/TensorArray_1*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ
Y
ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ


ExpandDims
ExpandDims(map/TensorArrayStack/TensorArrayGatherV3ExpandDims/dim*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ*

Tdim0
k
transpose/permConst*)
value B"                *
dtype0*
_output_shapes
:

	transpose	Transpose
ExpandDimstranspose/perm*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0*
T0
X
map_1/ShapeShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
c
map_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
e
map_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

map_1/strided_sliceStridedSlicemap_1/Shapemap_1/strided_slice/stackmap_1/strided_slice/stack_1map_1/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
о
map_1/TensorArrayTensorArrayV3map_1/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
k
map_1/TensorArrayUnstack/ShapeShapePlaceholder_1*
out_type0*
_output_shapes
:*
T0
v
,map_1/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
x
.map_1/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
x
.map_1/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
і
&map_1/TensorArrayUnstack/strided_sliceStridedSlicemap_1/TensorArrayUnstack/Shape,map_1/TensorArrayUnstack/strided_slice/stack.map_1/TensorArrayUnstack/strided_slice/stack_1.map_1/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
f
$map_1/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$map_1/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
map_1/TensorArrayUnstack/rangeRange$map_1/TensorArrayUnstack/range/start&map_1/TensorArrayUnstack/strided_slice$map_1/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0
ђ
@map_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_1/TensorArraymap_1/TensorArrayUnstack/rangePlaceholder_1map_1/TensorArray:1* 
_class
loc:@Placeholder_1*
_output_shapes
: *
T0
M
map_1/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
р
map_1/TensorArray_1TensorArrayV3map_1/strided_slice*
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: 
_
map_1/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Г
map_1/while/EnterEntermap_1/while/iteration_counter*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_1/while/while_context
Ѓ
map_1/while/Enter_1Entermap_1/Const*
_output_shapes
: *)

frame_namemap_1/while/while_context*
T0*
is_constant( *
parallel_iterations

­
map_1/while/Enter_2Entermap_1/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_1/while/while_context
t
map_1/while/MergeMergemap_1/while/Entermap_1/while/NextIteration*
T0*
N*
_output_shapes
: : 
z
map_1/while/Merge_1Mergemap_1/while/Enter_1map_1/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
z
map_1/while/Merge_2Mergemap_1/while/Enter_2map_1/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
d
map_1/while/LessLessmap_1/while/Mergemap_1/while/Less/Enter*
T0*
_output_shapes
: 
Ў
map_1/while/Less/EnterEntermap_1/strided_slice*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_1/while/while_context
h
map_1/while/Less_1Lessmap_1/while/Merge_1map_1/while/Less/Enter*
_output_shapes
: *
T0
b
map_1/while/LogicalAnd
LogicalAndmap_1/while/Lessmap_1/while/Less_1*
_output_shapes
: 
P
map_1/while/LoopCondLoopCondmap_1/while/LogicalAnd*
_output_shapes
: 

map_1/while/SwitchSwitchmap_1/while/Mergemap_1/while/LoopCond*$
_class
loc:@map_1/while/Merge*
_output_shapes
: : *
T0

map_1/while/Switch_1Switchmap_1/while/Merge_1map_1/while/LoopCond*
T0*&
_class
loc:@map_1/while/Merge_1*
_output_shapes
: : 

map_1/while/Switch_2Switchmap_1/while/Merge_2map_1/while/LoopCond*
T0*&
_class
loc:@map_1/while/Merge_2*
_output_shapes
: : 
W
map_1/while/IdentityIdentitymap_1/while/Switch:1*
T0*
_output_shapes
: 
[
map_1/while/Identity_1Identitymap_1/while/Switch_1:1*
T0*
_output_shapes
: 
[
map_1/while/Identity_2Identitymap_1/while/Switch_2:1*
_output_shapes
: *
T0
j
map_1/while/add/yConst^map_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
map_1/while/addAddmap_1/while/Identitymap_1/while/add/y*
_output_shapes
: *
T0
Л
map_1/while/TensorArrayReadV3TensorArrayReadV3#map_1/while/TensorArrayReadV3/Entermap_1/while/Identity_1%map_1/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Н
#map_1/while/TensorArrayReadV3/EnterEntermap_1/TensorArray*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
:*)

frame_namemap_1/while/while_context
ъ
%map_1/while/TensorArrayReadV3/Enter_1Enter@map_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_1/while/while_context
њ
map_1/while/DecodeJpeg
DecodeJpegmap_1/while/TensorArrayReadV3*
try_recover_truncated( *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(
z
!map_1/while/resize/ExpandDims/dimConst^map_1/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
Е
map_1/while/resize/ExpandDims
ExpandDimsmap_1/while/DecodeJpeg!map_1/while/resize/ExpandDims/dim*

Tdim0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

map_1/while/resize/sizeConst^map_1/while/Identity*
valueB",  ,  *
dtype0*
_output_shapes
:
Г
!map_1/while/resize/ResizeBilinearResizeBilinearmap_1/while/resize/ExpandDimsmap_1/while/resize/size*
T0*(
_output_shapes
:ЌЌ*
align_corners( 

map_1/while/resize/SqueezeSqueeze!map_1/while/resize/ResizeBilinear*
T0*$
_output_shapes
:ЌЌ*
squeeze_dims
 
m
map_1/while/div/yConst^map_1/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  C
x
map_1/while/divRealDivmap_1/while/resize/Squeezemap_1/while/div/y*$
_output_shapes
:ЌЌ*
T0

!map_1/while/Reshape_Preproc/shapeConst^map_1/while/Identity*!
valueB",  ,     *
dtype0*
_output_shapes
:

map_1/while/Reshape_PreprocReshapemap_1/while/div!map_1/while/Reshape_Preproc/shape*
T0*
Tshape0*$
_output_shapes
:ЌЌ

/map_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35map_1/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_1/while/Identity_1map_1/while/Reshape_Preprocmap_1/while/Identity_2*
T0*.
_class$
" loc:@map_1/while/Reshape_Preproc*
_output_shapes
: 

5map_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_1/TensorArray_1*
is_constant(*
_output_shapes
:*)

frame_namemap_1/while/while_context*
T0*.
_class$
" loc:@map_1/while/Reshape_Preproc*
parallel_iterations

l
map_1/while/add_1/yConst^map_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
f
map_1/while/add_1Addmap_1/while/Identity_1map_1/while/add_1/y*
T0*
_output_shapes
: 
\
map_1/while/NextIterationNextIterationmap_1/while/add*
T0*
_output_shapes
: 
`
map_1/while/NextIteration_1NextIterationmap_1/while/add_1*
_output_shapes
: *
T0
~
map_1/while/NextIteration_2NextIteration/map_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
M
map_1/while/ExitExitmap_1/while/Switch*
_output_shapes
: *
T0
Q
map_1/while/Exit_1Exitmap_1/while/Switch_1*
T0*
_output_shapes
: 
Q
map_1/while/Exit_2Exitmap_1/while/Switch_2*
T0*
_output_shapes
: 
І
(map_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_1/TensorArray_1map_1/while/Exit_2*
_output_shapes
: *&
_class
loc:@map_1/TensorArray_1

"map_1/TensorArrayStack/range/startConst*
value	B : *&
_class
loc:@map_1/TensorArray_1*
dtype0*
_output_shapes
: 

"map_1/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*&
_class
loc:@map_1/TensorArray_1
№
map_1/TensorArrayStack/rangeRange"map_1/TensorArrayStack/range/start(map_1/TensorArrayStack/TensorArraySizeV3"map_1/TensorArrayStack/range/delta*&
_class
loc:@map_1/TensorArray_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0

*map_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_1/TensorArray_1map_1/TensorArrayStack/rangemap_1/while/Exit_2*1
_output_shapes
:џџџџџџџџџЌЌ*!
element_shape:ЌЌ*&
_class
loc:@map_1/TensorArray_1*
dtype0
[
ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
Є
ExpandDims_1
ExpandDims*map_1/TensorArrayStack/TensorArrayGatherV3ExpandDims_1/dim*5
_output_shapes#
!:џџџџџџџџџЌЌ*

Tdim0*
T0
m
transpose_1/permConst*)
value B"                *
dtype0*
_output_shapes
:

transpose_1	TransposeExpandDims_1transpose_1/perm*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0
X
map_2/ShapeShapePlaceholder_2*
T0*
out_type0*
_output_shapes
:
c
map_2/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
e
map_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map_2/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0

map_2/strided_sliceStridedSlicemap_2/Shapemap_2/strided_slice/stackmap_2/strided_slice/stack_1map_2/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
о
map_2/TensorArrayTensorArrayV3map_2/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
k
map_2/TensorArrayUnstack/ShapeShapePlaceholder_2*
T0*
out_type0*
_output_shapes
:
v
,map_2/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
x
.map_2/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
x
.map_2/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
і
&map_2/TensorArrayUnstack/strided_sliceStridedSlicemap_2/TensorArrayUnstack/Shape,map_2/TensorArrayUnstack/strided_slice/stack.map_2/TensorArrayUnstack/strided_slice/stack_1.map_2/TensorArrayUnstack/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
f
$map_2/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$map_2/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
map_2/TensorArrayUnstack/rangeRange$map_2/TensorArrayUnstack/range/start&map_2/TensorArrayUnstack/strided_slice$map_2/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0
ђ
@map_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_2/TensorArraymap_2/TensorArrayUnstack/rangePlaceholder_2map_2/TensorArray:1*
T0* 
_class
loc:@Placeholder_2*
_output_shapes
: 
M
map_2/ConstConst*
dtype0*
_output_shapes
: *
value	B : 
р
map_2/TensorArray_1TensorArrayV3map_2/strided_slice*
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: 
_
map_2/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Г
map_2/while/EnterEntermap_2/while/iteration_counter*
_output_shapes
: *)

frame_namemap_2/while/while_context*
T0*
is_constant( *
parallel_iterations

Ѓ
map_2/while/Enter_1Entermap_2/Const*
_output_shapes
: *)

frame_namemap_2/while/while_context*
T0*
is_constant( *
parallel_iterations

­
map_2/while/Enter_2Entermap_2/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_2/while/while_context
t
map_2/while/MergeMergemap_2/while/Entermap_2/while/NextIteration*
T0*
N*
_output_shapes
: : 
z
map_2/while/Merge_1Mergemap_2/while/Enter_1map_2/while/NextIteration_1*
_output_shapes
: : *
T0*
N
z
map_2/while/Merge_2Mergemap_2/while/Enter_2map_2/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
d
map_2/while/LessLessmap_2/while/Mergemap_2/while/Less/Enter*
_output_shapes
: *
T0
Ў
map_2/while/Less/EnterEntermap_2/strided_slice*
_output_shapes
: *)

frame_namemap_2/while/while_context*
T0*
is_constant(*
parallel_iterations

h
map_2/while/Less_1Lessmap_2/while/Merge_1map_2/while/Less/Enter*
T0*
_output_shapes
: 
b
map_2/while/LogicalAnd
LogicalAndmap_2/while/Lessmap_2/while/Less_1*
_output_shapes
: 
P
map_2/while/LoopCondLoopCondmap_2/while/LogicalAnd*
_output_shapes
: 

map_2/while/SwitchSwitchmap_2/while/Mergemap_2/while/LoopCond*
T0*$
_class
loc:@map_2/while/Merge*
_output_shapes
: : 

map_2/while/Switch_1Switchmap_2/while/Merge_1map_2/while/LoopCond*
T0*&
_class
loc:@map_2/while/Merge_1*
_output_shapes
: : 

map_2/while/Switch_2Switchmap_2/while/Merge_2map_2/while/LoopCond*&
_class
loc:@map_2/while/Merge_2*
_output_shapes
: : *
T0
W
map_2/while/IdentityIdentitymap_2/while/Switch:1*
_output_shapes
: *
T0
[
map_2/while/Identity_1Identitymap_2/while/Switch_1:1*
_output_shapes
: *
T0
[
map_2/while/Identity_2Identitymap_2/while/Switch_2:1*
T0*
_output_shapes
: 
j
map_2/while/add/yConst^map_2/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
map_2/while/addAddmap_2/while/Identitymap_2/while/add/y*
T0*
_output_shapes
: 
Л
map_2/while/TensorArrayReadV3TensorArrayReadV3#map_2/while/TensorArrayReadV3/Entermap_2/while/Identity_1%map_2/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Н
#map_2/while/TensorArrayReadV3/EnterEntermap_2/TensorArray*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
:*)

frame_namemap_2/while/while_context
ъ
%map_2/while/TensorArrayReadV3/Enter_1Enter@map_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_2/while/while_context
њ
map_2/while/DecodeJpeg
DecodeJpegmap_2/while/TensorArrayReadV3*
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method 
z
!map_2/while/resize/ExpandDims/dimConst^map_2/while/Identity*
dtype0*
_output_shapes
: *
value	B : 
Е
map_2/while/resize/ExpandDims
ExpandDimsmap_2/while/DecodeJpeg!map_2/while/resize/ExpandDims/dim*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*

Tdim0*
T0

map_2/while/resize/sizeConst^map_2/while/Identity*
dtype0*
_output_shapes
:*
valueB",  ,  
Г
!map_2/while/resize/ResizeBilinearResizeBilinearmap_2/while/resize/ExpandDimsmap_2/while/resize/size*(
_output_shapes
:ЌЌ*
align_corners( *
T0

map_2/while/resize/SqueezeSqueeze!map_2/while/resize/ResizeBilinear*$
_output_shapes
:ЌЌ*
squeeze_dims
 *
T0
m
map_2/while/div/yConst^map_2/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  C
x
map_2/while/divRealDivmap_2/while/resize/Squeezemap_2/while/div/y*
T0*$
_output_shapes
:ЌЌ

!map_2/while/Reshape_Preproc/shapeConst^map_2/while/Identity*!
valueB",  ,     *
dtype0*
_output_shapes
:

map_2/while/Reshape_PreprocReshapemap_2/while/div!map_2/while/Reshape_Preproc/shape*$
_output_shapes
:ЌЌ*
T0*
Tshape0

/map_2/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35map_2/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_2/while/Identity_1map_2/while/Reshape_Preprocmap_2/while/Identity_2*.
_class$
" loc:@map_2/while/Reshape_Preproc*
_output_shapes
: *
T0

5map_2/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_2/TensorArray_1*
_output_shapes
:*)

frame_namemap_2/while/while_context*
T0*.
_class$
" loc:@map_2/while/Reshape_Preproc*
parallel_iterations
*
is_constant(
l
map_2/while/add_1/yConst^map_2/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
f
map_2/while/add_1Addmap_2/while/Identity_1map_2/while/add_1/y*
T0*
_output_shapes
: 
\
map_2/while/NextIterationNextIterationmap_2/while/add*
T0*
_output_shapes
: 
`
map_2/while/NextIteration_1NextIterationmap_2/while/add_1*
T0*
_output_shapes
: 
~
map_2/while/NextIteration_2NextIteration/map_2/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
M
map_2/while/ExitExitmap_2/while/Switch*
_output_shapes
: *
T0
Q
map_2/while/Exit_1Exitmap_2/while/Switch_1*
T0*
_output_shapes
: 
Q
map_2/while/Exit_2Exitmap_2/while/Switch_2*
_output_shapes
: *
T0
І
(map_2/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_2/TensorArray_1map_2/while/Exit_2*&
_class
loc:@map_2/TensorArray_1*
_output_shapes
: 

"map_2/TensorArrayStack/range/startConst*
value	B : *&
_class
loc:@map_2/TensorArray_1*
dtype0*
_output_shapes
: 

"map_2/TensorArrayStack/range/deltaConst*
value	B :*&
_class
loc:@map_2/TensorArray_1*
dtype0*
_output_shapes
: 
№
map_2/TensorArrayStack/rangeRange"map_2/TensorArrayStack/range/start(map_2/TensorArrayStack/TensorArraySizeV3"map_2/TensorArrayStack/range/delta*

Tidx0*&
_class
loc:@map_2/TensorArray_1*#
_output_shapes
:џџџџџџџџџ

*map_2/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_2/TensorArray_1map_2/TensorArrayStack/rangemap_2/while/Exit_2*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ*!
element_shape:ЌЌ*&
_class
loc:@map_2/TensorArray_1
[
ExpandDims_2/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Є
ExpandDims_2
ExpandDims*map_2/TensorArrayStack/TensorArrayGatherV3ExpandDims_2/dim*

Tdim0*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ
m
transpose_2/permConst*
dtype0*
_output_shapes
:*)
value B"                

transpose_2	TransposeExpandDims_2transpose_2/perm*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0
X
map_3/ShapeShapePlaceholder_3*
T0*
out_type0*
_output_shapes
:
c
map_3/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
e
map_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map_3/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0

map_3/strided_sliceStridedSlicemap_3/Shapemap_3/strided_slice/stackmap_3/strided_slice/stack_1map_3/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
о
map_3/TensorArrayTensorArrayV3map_3/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
k
map_3/TensorArrayUnstack/ShapeShapePlaceholder_3*
T0*
out_type0*
_output_shapes
:
v
,map_3/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
x
.map_3/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
x
.map_3/TensorArrayUnstack/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
і
&map_3/TensorArrayUnstack/strided_sliceStridedSlicemap_3/TensorArrayUnstack/Shape,map_3/TensorArrayUnstack/strided_slice/stack.map_3/TensorArrayUnstack/strided_slice/stack_1.map_3/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
f
$map_3/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$map_3/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
Ь
map_3/TensorArrayUnstack/rangeRange$map_3/TensorArrayUnstack/range/start&map_3/TensorArrayUnstack/strided_slice$map_3/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0
ђ
@map_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_3/TensorArraymap_3/TensorArrayUnstack/rangePlaceholder_3map_3/TensorArray:1*
T0* 
_class
loc:@Placeholder_3*
_output_shapes
: 
M
map_3/ConstConst*
_output_shapes
: *
value	B : *
dtype0
р
map_3/TensorArray_1TensorArrayV3map_3/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
_
map_3/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Г
map_3/while/EnterEntermap_3/while/iteration_counter*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_3/while/while_context*
T0*
is_constant( 
Ѓ
map_3/while/Enter_1Entermap_3/Const*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_3/while/while_context
­
map_3/while/Enter_2Entermap_3/TensorArray_1:1*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_3/while/while_context*
T0
t
map_3/while/MergeMergemap_3/while/Entermap_3/while/NextIteration*
T0*
N*
_output_shapes
: : 
z
map_3/while/Merge_1Mergemap_3/while/Enter_1map_3/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
z
map_3/while/Merge_2Mergemap_3/while/Enter_2map_3/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
d
map_3/while/LessLessmap_3/while/Mergemap_3/while/Less/Enter*
T0*
_output_shapes
: 
Ў
map_3/while/Less/EnterEntermap_3/strided_slice*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_3/while/while_context*
T0
h
map_3/while/Less_1Lessmap_3/while/Merge_1map_3/while/Less/Enter*
_output_shapes
: *
T0
b
map_3/while/LogicalAnd
LogicalAndmap_3/while/Lessmap_3/while/Less_1*
_output_shapes
: 
P
map_3/while/LoopCondLoopCondmap_3/while/LogicalAnd*
_output_shapes
: 

map_3/while/SwitchSwitchmap_3/while/Mergemap_3/while/LoopCond*
_output_shapes
: : *
T0*$
_class
loc:@map_3/while/Merge

map_3/while/Switch_1Switchmap_3/while/Merge_1map_3/while/LoopCond*
T0*&
_class
loc:@map_3/while/Merge_1*
_output_shapes
: : 

map_3/while/Switch_2Switchmap_3/while/Merge_2map_3/while/LoopCond*
T0*&
_class
loc:@map_3/while/Merge_2*
_output_shapes
: : 
W
map_3/while/IdentityIdentitymap_3/while/Switch:1*
_output_shapes
: *
T0
[
map_3/while/Identity_1Identitymap_3/while/Switch_1:1*
_output_shapes
: *
T0
[
map_3/while/Identity_2Identitymap_3/while/Switch_2:1*
T0*
_output_shapes
: 
j
map_3/while/add/yConst^map_3/while/Identity*
_output_shapes
: *
value	B :*
dtype0
`
map_3/while/addAddmap_3/while/Identitymap_3/while/add/y*
_output_shapes
: *
T0
Л
map_3/while/TensorArrayReadV3TensorArrayReadV3#map_3/while/TensorArrayReadV3/Entermap_3/while/Identity_1%map_3/while/TensorArrayReadV3/Enter_1*
_output_shapes
: *
dtype0
Н
#map_3/while/TensorArrayReadV3/EnterEntermap_3/TensorArray*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
:*)

frame_namemap_3/while/while_context
ъ
%map_3/while/TensorArrayReadV3/Enter_1Enter@map_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_3/while/while_context
њ
map_3/while/DecodeJpeg
DecodeJpegmap_3/while/TensorArrayReadV3*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method *
channels
z
!map_3/while/resize/ExpandDims/dimConst^map_3/while/Identity*
dtype0*
_output_shapes
: *
value	B : 
Е
map_3/while/resize/ExpandDims
ExpandDimsmap_3/while/DecodeJpeg!map_3/while/resize/ExpandDims/dim*

Tdim0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

map_3/while/resize/sizeConst^map_3/while/Identity*
valueB",  ,  *
dtype0*
_output_shapes
:
Г
!map_3/while/resize/ResizeBilinearResizeBilinearmap_3/while/resize/ExpandDimsmap_3/while/resize/size*
T0*(
_output_shapes
:ЌЌ*
align_corners( 

map_3/while/resize/SqueezeSqueeze!map_3/while/resize/ResizeBilinear*$
_output_shapes
:ЌЌ*
squeeze_dims
 *
T0
m
map_3/while/div/yConst^map_3/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  C
x
map_3/while/divRealDivmap_3/while/resize/Squeezemap_3/while/div/y*
T0*$
_output_shapes
:ЌЌ

!map_3/while/Reshape_Preproc/shapeConst^map_3/while/Identity*
dtype0*
_output_shapes
:*!
valueB",  ,     

map_3/while/Reshape_PreprocReshapemap_3/while/div!map_3/while/Reshape_Preproc/shape*$
_output_shapes
:ЌЌ*
T0*
Tshape0

/map_3/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35map_3/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_3/while/Identity_1map_3/while/Reshape_Preprocmap_3/while/Identity_2*
_output_shapes
: *
T0*.
_class$
" loc:@map_3/while/Reshape_Preproc

5map_3/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_3/TensorArray_1*
parallel_iterations
*
is_constant(*
_output_shapes
:*)

frame_namemap_3/while/while_context*
T0*.
_class$
" loc:@map_3/while/Reshape_Preproc
l
map_3/while/add_1/yConst^map_3/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
f
map_3/while/add_1Addmap_3/while/Identity_1map_3/while/add_1/y*
_output_shapes
: *
T0
\
map_3/while/NextIterationNextIterationmap_3/while/add*
T0*
_output_shapes
: 
`
map_3/while/NextIteration_1NextIterationmap_3/while/add_1*
T0*
_output_shapes
: 
~
map_3/while/NextIteration_2NextIteration/map_3/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
M
map_3/while/ExitExitmap_3/while/Switch*
_output_shapes
: *
T0
Q
map_3/while/Exit_1Exitmap_3/while/Switch_1*
_output_shapes
: *
T0
Q
map_3/while/Exit_2Exitmap_3/while/Switch_2*
T0*
_output_shapes
: 
І
(map_3/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_3/TensorArray_1map_3/while/Exit_2*&
_class
loc:@map_3/TensorArray_1*
_output_shapes
: 

"map_3/TensorArrayStack/range/startConst*
value	B : *&
_class
loc:@map_3/TensorArray_1*
dtype0*
_output_shapes
: 

"map_3/TensorArrayStack/range/deltaConst*
value	B :*&
_class
loc:@map_3/TensorArray_1*
dtype0*
_output_shapes
: 
№
map_3/TensorArrayStack/rangeRange"map_3/TensorArrayStack/range/start(map_3/TensorArrayStack/TensorArraySizeV3"map_3/TensorArrayStack/range/delta*&
_class
loc:@map_3/TensorArray_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0

*map_3/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_3/TensorArray_1map_3/TensorArrayStack/rangemap_3/while/Exit_2*&
_class
loc:@map_3/TensorArray_1*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ*!
element_shape:ЌЌ
[
ExpandDims_3/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
Є
ExpandDims_3
ExpandDims*map_3/TensorArrayStack/TensorArrayGatherV3ExpandDims_3/dim*

Tdim0*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ
m
transpose_3/permConst*
_output_shapes
:*)
value B"                *
dtype0

transpose_3	TransposeExpandDims_3transpose_3/perm*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0
X
map_4/ShapeShapePlaceholder_4*
T0*
out_type0*
_output_shapes
:
c
map_4/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
e
map_4/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
e
map_4/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

map_4/strided_sliceStridedSlicemap_4/Shapemap_4/strided_slice/stackmap_4/strided_slice/stack_1map_4/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
о
map_4/TensorArrayTensorArrayV3map_4/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
k
map_4/TensorArrayUnstack/ShapeShapePlaceholder_4*
T0*
out_type0*
_output_shapes
:
v
,map_4/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
x
.map_4/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
x
.map_4/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
і
&map_4/TensorArrayUnstack/strided_sliceStridedSlicemap_4/TensorArrayUnstack/Shape,map_4/TensorArrayUnstack/strided_slice/stack.map_4/TensorArrayUnstack/strided_slice/stack_1.map_4/TensorArrayUnstack/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
f
$map_4/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$map_4/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
map_4/TensorArrayUnstack/rangeRange$map_4/TensorArrayUnstack/range/start&map_4/TensorArrayUnstack/strided_slice$map_4/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0
ђ
@map_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_4/TensorArraymap_4/TensorArrayUnstack/rangePlaceholder_4map_4/TensorArray:1*
T0* 
_class
loc:@Placeholder_4*
_output_shapes
: 
M
map_4/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
р
map_4/TensorArray_1TensorArrayV3map_4/strided_slice*
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: 
_
map_4/while/iteration_counterConst*
dtype0*
_output_shapes
: *
value	B : 
Г
map_4/while/EnterEntermap_4/while/iteration_counter*
_output_shapes
: *)

frame_namemap_4/while/while_context*
T0*
is_constant( *
parallel_iterations

Ѓ
map_4/while/Enter_1Entermap_4/Const*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_4/while/while_context*
T0*
is_constant( 
­
map_4/while/Enter_2Entermap_4/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_4/while/while_context
t
map_4/while/MergeMergemap_4/while/Entermap_4/while/NextIteration*
N*
_output_shapes
: : *
T0
z
map_4/while/Merge_1Mergemap_4/while/Enter_1map_4/while/NextIteration_1*
N*
_output_shapes
: : *
T0
z
map_4/while/Merge_2Mergemap_4/while/Enter_2map_4/while/NextIteration_2*
N*
_output_shapes
: : *
T0
d
map_4/while/LessLessmap_4/while/Mergemap_4/while/Less/Enter*
T0*
_output_shapes
: 
Ў
map_4/while/Less/EnterEntermap_4/strided_slice*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_4/while/while_context*
T0
h
map_4/while/Less_1Lessmap_4/while/Merge_1map_4/while/Less/Enter*
_output_shapes
: *
T0
b
map_4/while/LogicalAnd
LogicalAndmap_4/while/Lessmap_4/while/Less_1*
_output_shapes
: 
P
map_4/while/LoopCondLoopCondmap_4/while/LogicalAnd*
_output_shapes
: 

map_4/while/SwitchSwitchmap_4/while/Mergemap_4/while/LoopCond*
T0*$
_class
loc:@map_4/while/Merge*
_output_shapes
: : 

map_4/while/Switch_1Switchmap_4/while/Merge_1map_4/while/LoopCond*
T0*&
_class
loc:@map_4/while/Merge_1*
_output_shapes
: : 

map_4/while/Switch_2Switchmap_4/while/Merge_2map_4/while/LoopCond*
T0*&
_class
loc:@map_4/while/Merge_2*
_output_shapes
: : 
W
map_4/while/IdentityIdentitymap_4/while/Switch:1*
_output_shapes
: *
T0
[
map_4/while/Identity_1Identitymap_4/while/Switch_1:1*
T0*
_output_shapes
: 
[
map_4/while/Identity_2Identitymap_4/while/Switch_2:1*
_output_shapes
: *
T0
j
map_4/while/add/yConst^map_4/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
map_4/while/addAddmap_4/while/Identitymap_4/while/add/y*
T0*
_output_shapes
: 
Л
map_4/while/TensorArrayReadV3TensorArrayReadV3#map_4/while/TensorArrayReadV3/Entermap_4/while/Identity_1%map_4/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Н
#map_4/while/TensorArrayReadV3/EnterEntermap_4/TensorArray*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
:*)

frame_namemap_4/while/while_context
ъ
%map_4/while/TensorArrayReadV3/Enter_1Enter@map_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_4/while/while_context*
T0*
is_constant(
њ
map_4/while/DecodeJpeg
DecodeJpegmap_4/while/TensorArrayReadV3*
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method 
z
!map_4/while/resize/ExpandDims/dimConst^map_4/while/Identity*
dtype0*
_output_shapes
: *
value	B : 
Е
map_4/while/resize/ExpandDims
ExpandDimsmap_4/while/DecodeJpeg!map_4/while/resize/ExpandDims/dim*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*

Tdim0*
T0

map_4/while/resize/sizeConst^map_4/while/Identity*
valueB",  ,  *
dtype0*
_output_shapes
:
Г
!map_4/while/resize/ResizeBilinearResizeBilinearmap_4/while/resize/ExpandDimsmap_4/while/resize/size*(
_output_shapes
:ЌЌ*
align_corners( *
T0

map_4/while/resize/SqueezeSqueeze!map_4/while/resize/ResizeBilinear*$
_output_shapes
:ЌЌ*
squeeze_dims
 *
T0
m
map_4/while/div/yConst^map_4/while/Identity*
_output_shapes
: *
valueB
 *  C*
dtype0
x
map_4/while/divRealDivmap_4/while/resize/Squeezemap_4/while/div/y*$
_output_shapes
:ЌЌ*
T0

!map_4/while/Reshape_Preproc/shapeConst^map_4/while/Identity*!
valueB",  ,     *
dtype0*
_output_shapes
:

map_4/while/Reshape_PreprocReshapemap_4/while/div!map_4/while/Reshape_Preproc/shape*
T0*
Tshape0*$
_output_shapes
:ЌЌ

/map_4/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35map_4/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_4/while/Identity_1map_4/while/Reshape_Preprocmap_4/while/Identity_2*
T0*.
_class$
" loc:@map_4/while/Reshape_Preproc*
_output_shapes
: 

5map_4/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_4/TensorArray_1*
T0*.
_class$
" loc:@map_4/while/Reshape_Preproc*
parallel_iterations
*
is_constant(*
_output_shapes
:*)

frame_namemap_4/while/while_context
l
map_4/while/add_1/yConst^map_4/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
f
map_4/while/add_1Addmap_4/while/Identity_1map_4/while/add_1/y*
_output_shapes
: *
T0
\
map_4/while/NextIterationNextIterationmap_4/while/add*
T0*
_output_shapes
: 
`
map_4/while/NextIteration_1NextIterationmap_4/while/add_1*
T0*
_output_shapes
: 
~
map_4/while/NextIteration_2NextIteration/map_4/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
M
map_4/while/ExitExitmap_4/while/Switch*
T0*
_output_shapes
: 
Q
map_4/while/Exit_1Exitmap_4/while/Switch_1*
_output_shapes
: *
T0
Q
map_4/while/Exit_2Exitmap_4/while/Switch_2*
T0*
_output_shapes
: 
І
(map_4/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_4/TensorArray_1map_4/while/Exit_2*&
_class
loc:@map_4/TensorArray_1*
_output_shapes
: 

"map_4/TensorArrayStack/range/startConst*
value	B : *&
_class
loc:@map_4/TensorArray_1*
dtype0*
_output_shapes
: 

"map_4/TensorArrayStack/range/deltaConst*
value	B :*&
_class
loc:@map_4/TensorArray_1*
dtype0*
_output_shapes
: 
№
map_4/TensorArrayStack/rangeRange"map_4/TensorArrayStack/range/start(map_4/TensorArrayStack/TensorArraySizeV3"map_4/TensorArrayStack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0*&
_class
loc:@map_4/TensorArray_1

*map_4/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_4/TensorArray_1map_4/TensorArrayStack/rangemap_4/while/Exit_2*&
_class
loc:@map_4/TensorArray_1*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ*!
element_shape:ЌЌ
[
ExpandDims_4/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Є
ExpandDims_4
ExpandDims*map_4/TensorArrayStack/TensorArrayGatherV3ExpandDims_4/dim*5
_output_shapes#
!:џџџџџџџџџЌЌ*

Tdim0*
T0
m
transpose_4/permConst*)
value B"                *
dtype0*
_output_shapes
:

transpose_4	TransposeExpandDims_4transpose_4/perm*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0
X
map_5/ShapeShapePlaceholder_5*
T0*
out_type0*
_output_shapes
:
c
map_5/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
e
map_5/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map_5/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

map_5/strided_sliceStridedSlicemap_5/Shapemap_5/strided_slice/stackmap_5/strided_slice/stack_1map_5/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
о
map_5/TensorArrayTensorArrayV3map_5/strided_slice*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*
tensor_array_name 
k
map_5/TensorArrayUnstack/ShapeShapePlaceholder_5*
T0*
out_type0*
_output_shapes
:
v
,map_5/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
x
.map_5/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
x
.map_5/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
і
&map_5/TensorArrayUnstack/strided_sliceStridedSlicemap_5/TensorArrayUnstack/Shape,map_5/TensorArrayUnstack/strided_slice/stack.map_5/TensorArrayUnstack/strided_slice/stack_1.map_5/TensorArrayUnstack/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
f
$map_5/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$map_5/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
map_5/TensorArrayUnstack/rangeRange$map_5/TensorArrayUnstack/range/start&map_5/TensorArrayUnstack/strided_slice$map_5/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0
ђ
@map_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_5/TensorArraymap_5/TensorArrayUnstack/rangePlaceholder_5map_5/TensorArray:1*
T0* 
_class
loc:@Placeholder_5*
_output_shapes
: 
M
map_5/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
р
map_5/TensorArray_1TensorArrayV3map_5/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
_
map_5/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Г
map_5/while/EnterEntermap_5/while/iteration_counter*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_5/while/while_context
Ѓ
map_5/while/Enter_1Entermap_5/Const*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_5/while/while_context
­
map_5/while/Enter_2Entermap_5/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_5/while/while_context
t
map_5/while/MergeMergemap_5/while/Entermap_5/while/NextIteration*
T0*
N*
_output_shapes
: : 
z
map_5/while/Merge_1Mergemap_5/while/Enter_1map_5/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
z
map_5/while/Merge_2Mergemap_5/while/Enter_2map_5/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
d
map_5/while/LessLessmap_5/while/Mergemap_5/while/Less/Enter*
T0*
_output_shapes
: 
Ў
map_5/while/Less/EnterEntermap_5/strided_slice*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_5/while/while_context
h
map_5/while/Less_1Lessmap_5/while/Merge_1map_5/while/Less/Enter*
T0*
_output_shapes
: 
b
map_5/while/LogicalAnd
LogicalAndmap_5/while/Lessmap_5/while/Less_1*
_output_shapes
: 
P
map_5/while/LoopCondLoopCondmap_5/while/LogicalAnd*
_output_shapes
: 

map_5/while/SwitchSwitchmap_5/while/Mergemap_5/while/LoopCond*$
_class
loc:@map_5/while/Merge*
_output_shapes
: : *
T0

map_5/while/Switch_1Switchmap_5/while/Merge_1map_5/while/LoopCond*
_output_shapes
: : *
T0*&
_class
loc:@map_5/while/Merge_1

map_5/while/Switch_2Switchmap_5/while/Merge_2map_5/while/LoopCond*&
_class
loc:@map_5/while/Merge_2*
_output_shapes
: : *
T0
W
map_5/while/IdentityIdentitymap_5/while/Switch:1*
T0*
_output_shapes
: 
[
map_5/while/Identity_1Identitymap_5/while/Switch_1:1*
_output_shapes
: *
T0
[
map_5/while/Identity_2Identitymap_5/while/Switch_2:1*
T0*
_output_shapes
: 
j
map_5/while/add/yConst^map_5/while/Identity*
_output_shapes
: *
value	B :*
dtype0
`
map_5/while/addAddmap_5/while/Identitymap_5/while/add/y*
T0*
_output_shapes
: 
Л
map_5/while/TensorArrayReadV3TensorArrayReadV3#map_5/while/TensorArrayReadV3/Entermap_5/while/Identity_1%map_5/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Н
#map_5/while/TensorArrayReadV3/EnterEntermap_5/TensorArray*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
:*)

frame_namemap_5/while/while_context
ъ
%map_5/while/TensorArrayReadV3/Enter_1Enter@map_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_5/while/while_context
њ
map_5/while/DecodeJpeg
DecodeJpegmap_5/while/TensorArrayReadV3*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( 
z
!map_5/while/resize/ExpandDims/dimConst^map_5/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
Е
map_5/while/resize/ExpandDims
ExpandDimsmap_5/while/DecodeJpeg!map_5/while/resize/ExpandDims/dim*

Tdim0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

map_5/while/resize/sizeConst^map_5/while/Identity*
dtype0*
_output_shapes
:*
valueB",  ,  
Г
!map_5/while/resize/ResizeBilinearResizeBilinearmap_5/while/resize/ExpandDimsmap_5/while/resize/size*
align_corners( *
T0*(
_output_shapes
:ЌЌ

map_5/while/resize/SqueezeSqueeze!map_5/while/resize/ResizeBilinear*$
_output_shapes
:ЌЌ*
squeeze_dims
 *
T0
m
map_5/while/div/yConst^map_5/while/Identity*
valueB
 *  C*
dtype0*
_output_shapes
: 
x
map_5/while/divRealDivmap_5/while/resize/Squeezemap_5/while/div/y*
T0*$
_output_shapes
:ЌЌ

!map_5/while/Reshape_Preproc/shapeConst^map_5/while/Identity*
_output_shapes
:*!
valueB",  ,     *
dtype0

map_5/while/Reshape_PreprocReshapemap_5/while/div!map_5/while/Reshape_Preproc/shape*
Tshape0*$
_output_shapes
:ЌЌ*
T0

/map_5/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35map_5/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_5/while/Identity_1map_5/while/Reshape_Preprocmap_5/while/Identity_2*
T0*.
_class$
" loc:@map_5/while/Reshape_Preproc*
_output_shapes
: 

5map_5/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_5/TensorArray_1*
is_constant(*
_output_shapes
:*)

frame_namemap_5/while/while_context*
T0*.
_class$
" loc:@map_5/while/Reshape_Preproc*
parallel_iterations

l
map_5/while/add_1/yConst^map_5/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
f
map_5/while/add_1Addmap_5/while/Identity_1map_5/while/add_1/y*
T0*
_output_shapes
: 
\
map_5/while/NextIterationNextIterationmap_5/while/add*
_output_shapes
: *
T0
`
map_5/while/NextIteration_1NextIterationmap_5/while/add_1*
T0*
_output_shapes
: 
~
map_5/while/NextIteration_2NextIteration/map_5/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
M
map_5/while/ExitExitmap_5/while/Switch*
T0*
_output_shapes
: 
Q
map_5/while/Exit_1Exitmap_5/while/Switch_1*
T0*
_output_shapes
: 
Q
map_5/while/Exit_2Exitmap_5/while/Switch_2*
T0*
_output_shapes
: 
І
(map_5/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_5/TensorArray_1map_5/while/Exit_2*&
_class
loc:@map_5/TensorArray_1*
_output_shapes
: 

"map_5/TensorArrayStack/range/startConst*
_output_shapes
: *
value	B : *&
_class
loc:@map_5/TensorArray_1*
dtype0

"map_5/TensorArrayStack/range/deltaConst*
_output_shapes
: *
value	B :*&
_class
loc:@map_5/TensorArray_1*
dtype0
№
map_5/TensorArrayStack/rangeRange"map_5/TensorArrayStack/range/start(map_5/TensorArrayStack/TensorArraySizeV3"map_5/TensorArrayStack/range/delta*&
_class
loc:@map_5/TensorArray_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0

*map_5/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_5/TensorArray_1map_5/TensorArrayStack/rangemap_5/while/Exit_2*&
_class
loc:@map_5/TensorArray_1*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ*!
element_shape:ЌЌ
[
ExpandDims_5/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Є
ExpandDims_5
ExpandDims*map_5/TensorArrayStack/TensorArrayGatherV3ExpandDims_5/dim*5
_output_shapes#
!:џџџџџџџџџЌЌ*

Tdim0*
T0
m
transpose_5/permConst*)
value B"                *
dtype0*
_output_shapes
:

transpose_5	TransposeExpandDims_5transpose_5/perm*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0*
T0
X
map_6/ShapeShapePlaceholder_6*
_output_shapes
:*
T0*
out_type0
c
map_6/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
e
map_6/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map_6/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0

map_6/strided_sliceStridedSlicemap_6/Shapemap_6/strided_slice/stackmap_6/strided_slice/stack_1map_6/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
о
map_6/TensorArrayTensorArrayV3map_6/strided_slice*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( 
k
map_6/TensorArrayUnstack/ShapeShapePlaceholder_6*
T0*
out_type0*
_output_shapes
:
v
,map_6/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
x
.map_6/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
x
.map_6/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
і
&map_6/TensorArrayUnstack/strided_sliceStridedSlicemap_6/TensorArrayUnstack/Shape,map_6/TensorArrayUnstack/strided_slice/stack.map_6/TensorArrayUnstack/strided_slice/stack_1.map_6/TensorArrayUnstack/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
f
$map_6/TensorArrayUnstack/range/startConst*
_output_shapes
: *
value	B : *
dtype0
f
$map_6/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
map_6/TensorArrayUnstack/rangeRange$map_6/TensorArrayUnstack/range/start&map_6/TensorArrayUnstack/strided_slice$map_6/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0
ђ
@map_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_6/TensorArraymap_6/TensorArrayUnstack/rangePlaceholder_6map_6/TensorArray:1*
T0* 
_class
loc:@Placeholder_6*
_output_shapes
: 
M
map_6/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
р
map_6/TensorArray_1TensorArrayV3map_6/strided_slice*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
_
map_6/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Г
map_6/while/EnterEntermap_6/while/iteration_counter*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_6/while/while_context
Ѓ
map_6/while/Enter_1Entermap_6/Const*
_output_shapes
: *)

frame_namemap_6/while/while_context*
T0*
is_constant( *
parallel_iterations

­
map_6/while/Enter_2Entermap_6/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_6/while/while_context
t
map_6/while/MergeMergemap_6/while/Entermap_6/while/NextIteration*
T0*
N*
_output_shapes
: : 
z
map_6/while/Merge_1Mergemap_6/while/Enter_1map_6/while/NextIteration_1*
N*
_output_shapes
: : *
T0
z
map_6/while/Merge_2Mergemap_6/while/Enter_2map_6/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
d
map_6/while/LessLessmap_6/while/Mergemap_6/while/Less/Enter*
_output_shapes
: *
T0
Ў
map_6/while/Less/EnterEntermap_6/strided_slice*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_6/while/while_context*
T0
h
map_6/while/Less_1Lessmap_6/while/Merge_1map_6/while/Less/Enter*
_output_shapes
: *
T0
b
map_6/while/LogicalAnd
LogicalAndmap_6/while/Lessmap_6/while/Less_1*
_output_shapes
: 
P
map_6/while/LoopCondLoopCondmap_6/while/LogicalAnd*
_output_shapes
: 

map_6/while/SwitchSwitchmap_6/while/Mergemap_6/while/LoopCond*
T0*$
_class
loc:@map_6/while/Merge*
_output_shapes
: : 

map_6/while/Switch_1Switchmap_6/while/Merge_1map_6/while/LoopCond*
_output_shapes
: : *
T0*&
_class
loc:@map_6/while/Merge_1

map_6/while/Switch_2Switchmap_6/while/Merge_2map_6/while/LoopCond*
_output_shapes
: : *
T0*&
_class
loc:@map_6/while/Merge_2
W
map_6/while/IdentityIdentitymap_6/while/Switch:1*
_output_shapes
: *
T0
[
map_6/while/Identity_1Identitymap_6/while/Switch_1:1*
T0*
_output_shapes
: 
[
map_6/while/Identity_2Identitymap_6/while/Switch_2:1*
T0*
_output_shapes
: 
j
map_6/while/add/yConst^map_6/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
map_6/while/addAddmap_6/while/Identitymap_6/while/add/y*
_output_shapes
: *
T0
Л
map_6/while/TensorArrayReadV3TensorArrayReadV3#map_6/while/TensorArrayReadV3/Entermap_6/while/Identity_1%map_6/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Н
#map_6/while/TensorArrayReadV3/EnterEntermap_6/TensorArray*
parallel_iterations
*
_output_shapes
:*)

frame_namemap_6/while/while_context*
T0*
is_constant(
ъ
%map_6/while/TensorArrayReadV3/Enter_1Enter@map_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_6/while/while_context
њ
map_6/while/DecodeJpeg
DecodeJpegmap_6/while/TensorArrayReadV3*
try_recover_truncated( *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(
z
!map_6/while/resize/ExpandDims/dimConst^map_6/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
Е
map_6/while/resize/ExpandDims
ExpandDimsmap_6/while/DecodeJpeg!map_6/while/resize/ExpandDims/dim*

Tdim0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

map_6/while/resize/sizeConst^map_6/while/Identity*
valueB",  ,  *
dtype0*
_output_shapes
:
Г
!map_6/while/resize/ResizeBilinearResizeBilinearmap_6/while/resize/ExpandDimsmap_6/while/resize/size*
T0*(
_output_shapes
:ЌЌ*
align_corners( 

map_6/while/resize/SqueezeSqueeze!map_6/while/resize/ResizeBilinear*
T0*$
_output_shapes
:ЌЌ*
squeeze_dims
 
m
map_6/while/div/yConst^map_6/while/Identity*
valueB
 *  C*
dtype0*
_output_shapes
: 
x
map_6/while/divRealDivmap_6/while/resize/Squeezemap_6/while/div/y*
T0*$
_output_shapes
:ЌЌ

!map_6/while/Reshape_Preproc/shapeConst^map_6/while/Identity*!
valueB",  ,     *
dtype0*
_output_shapes
:

map_6/while/Reshape_PreprocReshapemap_6/while/div!map_6/while/Reshape_Preproc/shape*
T0*
Tshape0*$
_output_shapes
:ЌЌ

/map_6/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35map_6/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_6/while/Identity_1map_6/while/Reshape_Preprocmap_6/while/Identity_2*
T0*.
_class$
" loc:@map_6/while/Reshape_Preproc*
_output_shapes
: 

5map_6/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_6/TensorArray_1*
T0*.
_class$
" loc:@map_6/while/Reshape_Preproc*
parallel_iterations
*
is_constant(*
_output_shapes
:*)

frame_namemap_6/while/while_context
l
map_6/while/add_1/yConst^map_6/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
f
map_6/while/add_1Addmap_6/while/Identity_1map_6/while/add_1/y*
T0*
_output_shapes
: 
\
map_6/while/NextIterationNextIterationmap_6/while/add*
T0*
_output_shapes
: 
`
map_6/while/NextIteration_1NextIterationmap_6/while/add_1*
_output_shapes
: *
T0
~
map_6/while/NextIteration_2NextIteration/map_6/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
M
map_6/while/ExitExitmap_6/while/Switch*
T0*
_output_shapes
: 
Q
map_6/while/Exit_1Exitmap_6/while/Switch_1*
_output_shapes
: *
T0
Q
map_6/while/Exit_2Exitmap_6/while/Switch_2*
T0*
_output_shapes
: 
І
(map_6/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_6/TensorArray_1map_6/while/Exit_2*&
_class
loc:@map_6/TensorArray_1*
_output_shapes
: 

"map_6/TensorArrayStack/range/startConst*
value	B : *&
_class
loc:@map_6/TensorArray_1*
dtype0*
_output_shapes
: 

"map_6/TensorArrayStack/range/deltaConst*
value	B :*&
_class
loc:@map_6/TensorArray_1*
dtype0*
_output_shapes
: 
№
map_6/TensorArrayStack/rangeRange"map_6/TensorArrayStack/range/start(map_6/TensorArrayStack/TensorArraySizeV3"map_6/TensorArrayStack/range/delta*&
_class
loc:@map_6/TensorArray_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0

*map_6/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_6/TensorArray_1map_6/TensorArrayStack/rangemap_6/while/Exit_2*&
_class
loc:@map_6/TensorArray_1*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ*!
element_shape:ЌЌ
[
ExpandDims_6/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Є
ExpandDims_6
ExpandDims*map_6/TensorArrayStack/TensorArrayGatherV3ExpandDims_6/dim*

Tdim0*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ
m
transpose_6/permConst*)
value B"                *
dtype0*
_output_shapes
:

transpose_6	TransposeExpandDims_6transpose_6/perm*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0
X
map_7/ShapeShapePlaceholder_7*
T0*
out_type0*
_output_shapes
:
c
map_7/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
e
map_7/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map_7/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

map_7/strided_sliceStridedSlicemap_7/Shapemap_7/strided_slice/stackmap_7/strided_slice/stack_1map_7/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
о
map_7/TensorArrayTensorArrayV3map_7/strided_slice*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
k
map_7/TensorArrayUnstack/ShapeShapePlaceholder_7*
_output_shapes
:*
T0*
out_type0
v
,map_7/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
x
.map_7/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
x
.map_7/TensorArrayUnstack/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
і
&map_7/TensorArrayUnstack/strided_sliceStridedSlicemap_7/TensorArrayUnstack/Shape,map_7/TensorArrayUnstack/strided_slice/stack.map_7/TensorArrayUnstack/strided_slice/stack_1.map_7/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
f
$map_7/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$map_7/TensorArrayUnstack/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
Ь
map_7/TensorArrayUnstack/rangeRange$map_7/TensorArrayUnstack/range/start&map_7/TensorArrayUnstack/strided_slice$map_7/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0
ђ
@map_7/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_7/TensorArraymap_7/TensorArrayUnstack/rangePlaceholder_7map_7/TensorArray:1*
T0* 
_class
loc:@Placeholder_7*
_output_shapes
: 
M
map_7/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
р
map_7/TensorArray_1TensorArrayV3map_7/strided_slice*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:
_
map_7/while/iteration_counterConst*
_output_shapes
: *
value	B : *
dtype0
Г
map_7/while/EnterEntermap_7/while/iteration_counter*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_7/while/while_context*
T0
Ѓ
map_7/while/Enter_1Entermap_7/Const*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_7/while/while_context
­
map_7/while/Enter_2Entermap_7/TensorArray_1:1*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_7/while/while_context*
T0*
is_constant( 
t
map_7/while/MergeMergemap_7/while/Entermap_7/while/NextIteration*
T0*
N*
_output_shapes
: : 
z
map_7/while/Merge_1Mergemap_7/while/Enter_1map_7/while/NextIteration_1*
N*
_output_shapes
: : *
T0
z
map_7/while/Merge_2Mergemap_7/while/Enter_2map_7/while/NextIteration_2*
N*
_output_shapes
: : *
T0
d
map_7/while/LessLessmap_7/while/Mergemap_7/while/Less/Enter*
T0*
_output_shapes
: 
Ў
map_7/while/Less/EnterEntermap_7/strided_slice*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_7/while/while_context*
T0
h
map_7/while/Less_1Lessmap_7/while/Merge_1map_7/while/Less/Enter*
T0*
_output_shapes
: 
b
map_7/while/LogicalAnd
LogicalAndmap_7/while/Lessmap_7/while/Less_1*
_output_shapes
: 
P
map_7/while/LoopCondLoopCondmap_7/while/LogicalAnd*
_output_shapes
: 

map_7/while/SwitchSwitchmap_7/while/Mergemap_7/while/LoopCond*
_output_shapes
: : *
T0*$
_class
loc:@map_7/while/Merge

map_7/while/Switch_1Switchmap_7/while/Merge_1map_7/while/LoopCond*
T0*&
_class
loc:@map_7/while/Merge_1*
_output_shapes
: : 

map_7/while/Switch_2Switchmap_7/while/Merge_2map_7/while/LoopCond*
T0*&
_class
loc:@map_7/while/Merge_2*
_output_shapes
: : 
W
map_7/while/IdentityIdentitymap_7/while/Switch:1*
T0*
_output_shapes
: 
[
map_7/while/Identity_1Identitymap_7/while/Switch_1:1*
_output_shapes
: *
T0
[
map_7/while/Identity_2Identitymap_7/while/Switch_2:1*
_output_shapes
: *
T0
j
map_7/while/add/yConst^map_7/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
map_7/while/addAddmap_7/while/Identitymap_7/while/add/y*
T0*
_output_shapes
: 
Л
map_7/while/TensorArrayReadV3TensorArrayReadV3#map_7/while/TensorArrayReadV3/Entermap_7/while/Identity_1%map_7/while/TensorArrayReadV3/Enter_1*
_output_shapes
: *
dtype0
Н
#map_7/while/TensorArrayReadV3/EnterEntermap_7/TensorArray*
is_constant(*
parallel_iterations
*
_output_shapes
:*)

frame_namemap_7/while/while_context*
T0
ъ
%map_7/while/TensorArrayReadV3/Enter_1Enter@map_7/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_7/while/while_context*
T0*
is_constant(
њ
map_7/while/DecodeJpeg
DecodeJpegmap_7/while/TensorArrayReadV3*
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method 
z
!map_7/while/resize/ExpandDims/dimConst^map_7/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
Е
map_7/while/resize/ExpandDims
ExpandDimsmap_7/while/DecodeJpeg!map_7/while/resize/ExpandDims/dim*

Tdim0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

map_7/while/resize/sizeConst^map_7/while/Identity*
_output_shapes
:*
valueB",  ,  *
dtype0
Г
!map_7/while/resize/ResizeBilinearResizeBilinearmap_7/while/resize/ExpandDimsmap_7/while/resize/size*
T0*(
_output_shapes
:ЌЌ*
align_corners( 

map_7/while/resize/SqueezeSqueeze!map_7/while/resize/ResizeBilinear*
T0*$
_output_shapes
:ЌЌ*
squeeze_dims
 
m
map_7/while/div/yConst^map_7/while/Identity*
_output_shapes
: *
valueB
 *  C*
dtype0
x
map_7/while/divRealDivmap_7/while/resize/Squeezemap_7/while/div/y*
T0*$
_output_shapes
:ЌЌ

!map_7/while/Reshape_Preproc/shapeConst^map_7/while/Identity*!
valueB",  ,     *
dtype0*
_output_shapes
:

map_7/while/Reshape_PreprocReshapemap_7/while/div!map_7/while/Reshape_Preproc/shape*
Tshape0*$
_output_shapes
:ЌЌ*
T0

/map_7/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35map_7/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_7/while/Identity_1map_7/while/Reshape_Preprocmap_7/while/Identity_2*.
_class$
" loc:@map_7/while/Reshape_Preproc*
_output_shapes
: *
T0

5map_7/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_7/TensorArray_1*
T0*.
_class$
" loc:@map_7/while/Reshape_Preproc*
parallel_iterations
*
is_constant(*
_output_shapes
:*)

frame_namemap_7/while/while_context
l
map_7/while/add_1/yConst^map_7/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
f
map_7/while/add_1Addmap_7/while/Identity_1map_7/while/add_1/y*
_output_shapes
: *
T0
\
map_7/while/NextIterationNextIterationmap_7/while/add*
T0*
_output_shapes
: 
`
map_7/while/NextIteration_1NextIterationmap_7/while/add_1*
T0*
_output_shapes
: 
~
map_7/while/NextIteration_2NextIteration/map_7/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
M
map_7/while/ExitExitmap_7/while/Switch*
T0*
_output_shapes
: 
Q
map_7/while/Exit_1Exitmap_7/while/Switch_1*
T0*
_output_shapes
: 
Q
map_7/while/Exit_2Exitmap_7/while/Switch_2*
_output_shapes
: *
T0
І
(map_7/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_7/TensorArray_1map_7/while/Exit_2*
_output_shapes
: *&
_class
loc:@map_7/TensorArray_1

"map_7/TensorArrayStack/range/startConst*
value	B : *&
_class
loc:@map_7/TensorArray_1*
dtype0*
_output_shapes
: 

"map_7/TensorArrayStack/range/deltaConst*
value	B :*&
_class
loc:@map_7/TensorArray_1*
dtype0*
_output_shapes
: 
№
map_7/TensorArrayStack/rangeRange"map_7/TensorArrayStack/range/start(map_7/TensorArrayStack/TensorArraySizeV3"map_7/TensorArrayStack/range/delta*&
_class
loc:@map_7/TensorArray_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0

*map_7/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_7/TensorArray_1map_7/TensorArrayStack/rangemap_7/while/Exit_2*!
element_shape:ЌЌ*&
_class
loc:@map_7/TensorArray_1*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ
[
ExpandDims_7/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
Є
ExpandDims_7
ExpandDims*map_7/TensorArrayStack/TensorArrayGatherV3ExpandDims_7/dim*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ*

Tdim0
m
transpose_7/permConst*)
value B"                *
dtype0*
_output_shapes
:

transpose_7	TransposeExpandDims_7transpose_7/perm*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0
X
map_8/ShapeShapePlaceholder_8*
_output_shapes
:*
T0*
out_type0
c
map_8/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
e
map_8/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map_8/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

map_8/strided_sliceStridedSlicemap_8/Shapemap_8/strided_slice/stackmap_8/strided_slice/stack_1map_8/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
о
map_8/TensorArrayTensorArrayV3map_8/strided_slice*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*
tensor_array_name 
k
map_8/TensorArrayUnstack/ShapeShapePlaceholder_8*
T0*
out_type0*
_output_shapes
:
v
,map_8/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
x
.map_8/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
x
.map_8/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
і
&map_8/TensorArrayUnstack/strided_sliceStridedSlicemap_8/TensorArrayUnstack/Shape,map_8/TensorArrayUnstack/strided_slice/stack.map_8/TensorArrayUnstack/strided_slice/stack_1.map_8/TensorArrayUnstack/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
f
$map_8/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$map_8/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
map_8/TensorArrayUnstack/rangeRange$map_8/TensorArrayUnstack/range/start&map_8/TensorArrayUnstack/strided_slice$map_8/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ
ђ
@map_8/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_8/TensorArraymap_8/TensorArrayUnstack/rangePlaceholder_8map_8/TensorArray:1*
_output_shapes
: *
T0* 
_class
loc:@Placeholder_8
M
map_8/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
р
map_8/TensorArray_1TensorArrayV3map_8/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
_
map_8/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Г
map_8/while/EnterEntermap_8/while/iteration_counter*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_8/while/while_context
Ѓ
map_8/while/Enter_1Entermap_8/Const*
_output_shapes
: *)

frame_namemap_8/while/while_context*
T0*
is_constant( *
parallel_iterations

­
map_8/while/Enter_2Entermap_8/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_8/while/while_context
t
map_8/while/MergeMergemap_8/while/Entermap_8/while/NextIteration*
_output_shapes
: : *
T0*
N
z
map_8/while/Merge_1Mergemap_8/while/Enter_1map_8/while/NextIteration_1*
N*
_output_shapes
: : *
T0
z
map_8/while/Merge_2Mergemap_8/while/Enter_2map_8/while/NextIteration_2*
N*
_output_shapes
: : *
T0
d
map_8/while/LessLessmap_8/while/Mergemap_8/while/Less/Enter*
_output_shapes
: *
T0
Ў
map_8/while/Less/EnterEntermap_8/strided_slice*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_8/while/while_context
h
map_8/while/Less_1Lessmap_8/while/Merge_1map_8/while/Less/Enter*
_output_shapes
: *
T0
b
map_8/while/LogicalAnd
LogicalAndmap_8/while/Lessmap_8/while/Less_1*
_output_shapes
: 
P
map_8/while/LoopCondLoopCondmap_8/while/LogicalAnd*
_output_shapes
: 

map_8/while/SwitchSwitchmap_8/while/Mergemap_8/while/LoopCond*
_output_shapes
: : *
T0*$
_class
loc:@map_8/while/Merge

map_8/while/Switch_1Switchmap_8/while/Merge_1map_8/while/LoopCond*
T0*&
_class
loc:@map_8/while/Merge_1*
_output_shapes
: : 

map_8/while/Switch_2Switchmap_8/while/Merge_2map_8/while/LoopCond*
T0*&
_class
loc:@map_8/while/Merge_2*
_output_shapes
: : 
W
map_8/while/IdentityIdentitymap_8/while/Switch:1*
_output_shapes
: *
T0
[
map_8/while/Identity_1Identitymap_8/while/Switch_1:1*
T0*
_output_shapes
: 
[
map_8/while/Identity_2Identitymap_8/while/Switch_2:1*
_output_shapes
: *
T0
j
map_8/while/add/yConst^map_8/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
map_8/while/addAddmap_8/while/Identitymap_8/while/add/y*
T0*
_output_shapes
: 
Л
map_8/while/TensorArrayReadV3TensorArrayReadV3#map_8/while/TensorArrayReadV3/Entermap_8/while/Identity_1%map_8/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Н
#map_8/while/TensorArrayReadV3/EnterEntermap_8/TensorArray*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
:*)

frame_namemap_8/while/while_context
ъ
%map_8/while/TensorArrayReadV3/Enter_1Enter@map_8/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_8/while/while_context
њ
map_8/while/DecodeJpeg
DecodeJpegmap_8/while/TensorArrayReadV3*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( 
z
!map_8/while/resize/ExpandDims/dimConst^map_8/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
Е
map_8/while/resize/ExpandDims
ExpandDimsmap_8/while/DecodeJpeg!map_8/while/resize/ExpandDims/dim*

Tdim0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

map_8/while/resize/sizeConst^map_8/while/Identity*
valueB",  ,  *
dtype0*
_output_shapes
:
Г
!map_8/while/resize/ResizeBilinearResizeBilinearmap_8/while/resize/ExpandDimsmap_8/while/resize/size*
align_corners( *
T0*(
_output_shapes
:ЌЌ

map_8/while/resize/SqueezeSqueeze!map_8/while/resize/ResizeBilinear*
squeeze_dims
 *
T0*$
_output_shapes
:ЌЌ
m
map_8/while/div/yConst^map_8/while/Identity*
valueB
 *  C*
dtype0*
_output_shapes
: 
x
map_8/while/divRealDivmap_8/while/resize/Squeezemap_8/while/div/y*$
_output_shapes
:ЌЌ*
T0

!map_8/while/Reshape_Preproc/shapeConst^map_8/while/Identity*
dtype0*
_output_shapes
:*!
valueB",  ,     

map_8/while/Reshape_PreprocReshapemap_8/while/div!map_8/while/Reshape_Preproc/shape*
T0*
Tshape0*$
_output_shapes
:ЌЌ

/map_8/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35map_8/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_8/while/Identity_1map_8/while/Reshape_Preprocmap_8/while/Identity_2*
_output_shapes
: *
T0*.
_class$
" loc:@map_8/while/Reshape_Preproc

5map_8/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_8/TensorArray_1*
is_constant(*
_output_shapes
:*)

frame_namemap_8/while/while_context*
T0*.
_class$
" loc:@map_8/while/Reshape_Preproc*
parallel_iterations

l
map_8/while/add_1/yConst^map_8/while/Identity*
_output_shapes
: *
value	B :*
dtype0
f
map_8/while/add_1Addmap_8/while/Identity_1map_8/while/add_1/y*
T0*
_output_shapes
: 
\
map_8/while/NextIterationNextIterationmap_8/while/add*
_output_shapes
: *
T0
`
map_8/while/NextIteration_1NextIterationmap_8/while/add_1*
T0*
_output_shapes
: 
~
map_8/while/NextIteration_2NextIteration/map_8/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
M
map_8/while/ExitExitmap_8/while/Switch*
_output_shapes
: *
T0
Q
map_8/while/Exit_1Exitmap_8/while/Switch_1*
T0*
_output_shapes
: 
Q
map_8/while/Exit_2Exitmap_8/while/Switch_2*
T0*
_output_shapes
: 
І
(map_8/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_8/TensorArray_1map_8/while/Exit_2*&
_class
loc:@map_8/TensorArray_1*
_output_shapes
: 

"map_8/TensorArrayStack/range/startConst*
value	B : *&
_class
loc:@map_8/TensorArray_1*
dtype0*
_output_shapes
: 

"map_8/TensorArrayStack/range/deltaConst*
value	B :*&
_class
loc:@map_8/TensorArray_1*
dtype0*
_output_shapes
: 
№
map_8/TensorArrayStack/rangeRange"map_8/TensorArrayStack/range/start(map_8/TensorArrayStack/TensorArraySizeV3"map_8/TensorArrayStack/range/delta*&
_class
loc:@map_8/TensorArray_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0

*map_8/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_8/TensorArray_1map_8/TensorArrayStack/rangemap_8/while/Exit_2*&
_class
loc:@map_8/TensorArray_1*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ*!
element_shape:ЌЌ
[
ExpandDims_8/dimConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0
Є
ExpandDims_8
ExpandDims*map_8/TensorArrayStack/TensorArrayGatherV3ExpandDims_8/dim*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ*

Tdim0
m
transpose_8/permConst*)
value B"                *
dtype0*
_output_shapes
:

transpose_8	TransposeExpandDims_8transpose_8/perm*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0
X
map_9/ShapeShapePlaceholder_9*
_output_shapes
:*
T0*
out_type0
c
map_9/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
e
map_9/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map_9/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

map_9/strided_sliceStridedSlicemap_9/Shapemap_9/strided_slice/stackmap_9/strided_slice/stack_1map_9/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
о
map_9/TensorArrayTensorArrayV3map_9/strided_slice*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*
tensor_array_name *
dtype0
k
map_9/TensorArrayUnstack/ShapeShapePlaceholder_9*
T0*
out_type0*
_output_shapes
:
v
,map_9/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
x
.map_9/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
x
.map_9/TensorArrayUnstack/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
і
&map_9/TensorArrayUnstack/strided_sliceStridedSlicemap_9/TensorArrayUnstack/Shape,map_9/TensorArrayUnstack/strided_slice/stack.map_9/TensorArrayUnstack/strided_slice/stack_1.map_9/TensorArrayUnstack/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask 
f
$map_9/TensorArrayUnstack/range/startConst*
_output_shapes
: *
value	B : *
dtype0
f
$map_9/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
map_9/TensorArrayUnstack/rangeRange$map_9/TensorArrayUnstack/range/start&map_9/TensorArrayUnstack/strided_slice$map_9/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0
ђ
@map_9/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_9/TensorArraymap_9/TensorArrayUnstack/rangePlaceholder_9map_9/TensorArray:1*
_output_shapes
: *
T0* 
_class
loc:@Placeholder_9
M
map_9/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
р
map_9/TensorArray_1TensorArrayV3map_9/strided_slice*
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: 
_
map_9/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Г
map_9/while/EnterEntermap_9/while/iteration_counter*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_9/while/while_context*
T0
Ѓ
map_9/while/Enter_1Entermap_9/Const*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_9/while/while_context*
T0
­
map_9/while/Enter_2Entermap_9/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *)

frame_namemap_9/while/while_context
t
map_9/while/MergeMergemap_9/while/Entermap_9/while/NextIteration*
T0*
N*
_output_shapes
: : 
z
map_9/while/Merge_1Mergemap_9/while/Enter_1map_9/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
z
map_9/while/Merge_2Mergemap_9/while/Enter_2map_9/while/NextIteration_2*
_output_shapes
: : *
T0*
N
d
map_9/while/LessLessmap_9/while/Mergemap_9/while/Less/Enter*
T0*
_output_shapes
: 
Ў
map_9/while/Less/EnterEntermap_9/strided_slice*
_output_shapes
: *)

frame_namemap_9/while/while_context*
T0*
is_constant(*
parallel_iterations

h
map_9/while/Less_1Lessmap_9/while/Merge_1map_9/while/Less/Enter*
T0*
_output_shapes
: 
b
map_9/while/LogicalAnd
LogicalAndmap_9/while/Lessmap_9/while/Less_1*
_output_shapes
: 
P
map_9/while/LoopCondLoopCondmap_9/while/LogicalAnd*
_output_shapes
: 

map_9/while/SwitchSwitchmap_9/while/Mergemap_9/while/LoopCond*
_output_shapes
: : *
T0*$
_class
loc:@map_9/while/Merge

map_9/while/Switch_1Switchmap_9/while/Merge_1map_9/while/LoopCond*
_output_shapes
: : *
T0*&
_class
loc:@map_9/while/Merge_1

map_9/while/Switch_2Switchmap_9/while/Merge_2map_9/while/LoopCond*
T0*&
_class
loc:@map_9/while/Merge_2*
_output_shapes
: : 
W
map_9/while/IdentityIdentitymap_9/while/Switch:1*
T0*
_output_shapes
: 
[
map_9/while/Identity_1Identitymap_9/while/Switch_1:1*
T0*
_output_shapes
: 
[
map_9/while/Identity_2Identitymap_9/while/Switch_2:1*
T0*
_output_shapes
: 
j
map_9/while/add/yConst^map_9/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
map_9/while/addAddmap_9/while/Identitymap_9/while/add/y*
_output_shapes
: *
T0
Л
map_9/while/TensorArrayReadV3TensorArrayReadV3#map_9/while/TensorArrayReadV3/Entermap_9/while/Identity_1%map_9/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Н
#map_9/while/TensorArrayReadV3/EnterEntermap_9/TensorArray*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
:*)

frame_namemap_9/while/while_context
ъ
%map_9/while/TensorArrayReadV3/Enter_1Enter@map_9/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *)

frame_namemap_9/while/while_context
њ
map_9/while/DecodeJpeg
DecodeJpegmap_9/while/TensorArrayReadV3*
fancy_upscaling(*
try_recover_truncated( *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method *
channels*
acceptable_fraction%  ?
z
!map_9/while/resize/ExpandDims/dimConst^map_9/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
Е
map_9/while/resize/ExpandDims
ExpandDimsmap_9/while/DecodeJpeg!map_9/while/resize/ExpandDims/dim*

Tdim0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

map_9/while/resize/sizeConst^map_9/while/Identity*
_output_shapes
:*
valueB",  ,  *
dtype0
Г
!map_9/while/resize/ResizeBilinearResizeBilinearmap_9/while/resize/ExpandDimsmap_9/while/resize/size*
align_corners( *
T0*(
_output_shapes
:ЌЌ

map_9/while/resize/SqueezeSqueeze!map_9/while/resize/ResizeBilinear*$
_output_shapes
:ЌЌ*
squeeze_dims
 *
T0
m
map_9/while/div/yConst^map_9/while/Identity*
valueB
 *  C*
dtype0*
_output_shapes
: 
x
map_9/while/divRealDivmap_9/while/resize/Squeezemap_9/while/div/y*$
_output_shapes
:ЌЌ*
T0

!map_9/while/Reshape_Preproc/shapeConst^map_9/while/Identity*!
valueB",  ,     *
dtype0*
_output_shapes
:

map_9/while/Reshape_PreprocReshapemap_9/while/div!map_9/while/Reshape_Preproc/shape*$
_output_shapes
:ЌЌ*
T0*
Tshape0

/map_9/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35map_9/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_9/while/Identity_1map_9/while/Reshape_Preprocmap_9/while/Identity_2*
T0*.
_class$
" loc:@map_9/while/Reshape_Preproc*
_output_shapes
: 

5map_9/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_9/TensorArray_1*
is_constant(*
_output_shapes
:*)

frame_namemap_9/while/while_context*
T0*.
_class$
" loc:@map_9/while/Reshape_Preproc*
parallel_iterations

l
map_9/while/add_1/yConst^map_9/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
f
map_9/while/add_1Addmap_9/while/Identity_1map_9/while/add_1/y*
T0*
_output_shapes
: 
\
map_9/while/NextIterationNextIterationmap_9/while/add*
_output_shapes
: *
T0
`
map_9/while/NextIteration_1NextIterationmap_9/while/add_1*
_output_shapes
: *
T0
~
map_9/while/NextIteration_2NextIteration/map_9/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
M
map_9/while/ExitExitmap_9/while/Switch*
T0*
_output_shapes
: 
Q
map_9/while/Exit_1Exitmap_9/while/Switch_1*
T0*
_output_shapes
: 
Q
map_9/while/Exit_2Exitmap_9/while/Switch_2*
_output_shapes
: *
T0
І
(map_9/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_9/TensorArray_1map_9/while/Exit_2*
_output_shapes
: *&
_class
loc:@map_9/TensorArray_1

"map_9/TensorArrayStack/range/startConst*
value	B : *&
_class
loc:@map_9/TensorArray_1*
dtype0*
_output_shapes
: 

"map_9/TensorArrayStack/range/deltaConst*
_output_shapes
: *
value	B :*&
_class
loc:@map_9/TensorArray_1*
dtype0
№
map_9/TensorArrayStack/rangeRange"map_9/TensorArrayStack/range/start(map_9/TensorArrayStack/TensorArraySizeV3"map_9/TensorArrayStack/range/delta*&
_class
loc:@map_9/TensorArray_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0

*map_9/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_9/TensorArray_1map_9/TensorArrayStack/rangemap_9/while/Exit_2*!
element_shape:ЌЌ*&
_class
loc:@map_9/TensorArray_1*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ
[
ExpandDims_9/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
Є
ExpandDims_9
ExpandDims*map_9/TensorArrayStack/TensorArrayGatherV3ExpandDims_9/dim*5
_output_shapes#
!:џџџџџџџџџЌЌ*

Tdim0*
T0
m
transpose_9/permConst*
dtype0*
_output_shapes
:*)
value B"                

transpose_9	TransposeExpandDims_9transpose_9/perm*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0*
T0
Z
map_10/ShapeShapePlaceholder_10*
_output_shapes
:*
T0*
out_type0
d
map_10/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
map_10/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
map_10/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

map_10/strided_sliceStridedSlicemap_10/Shapemap_10/strided_slice/stackmap_10/strided_slice/stack_1map_10/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
р
map_10/TensorArrayTensorArrayV3map_10/strided_slice*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:
m
map_10/TensorArrayUnstack/ShapeShapePlaceholder_10*
T0*
out_type0*
_output_shapes
:
w
-map_10/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/map_10/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
y
/map_10/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ћ
'map_10/TensorArrayUnstack/strided_sliceStridedSlicemap_10/TensorArrayUnstack/Shape-map_10/TensorArrayUnstack/strided_slice/stack/map_10/TensorArrayUnstack/strided_slice/stack_1/map_10/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
g
%map_10/TensorArrayUnstack/range/startConst*
_output_shapes
: *
value	B : *
dtype0
g
%map_10/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
а
map_10/TensorArrayUnstack/rangeRange%map_10/TensorArrayUnstack/range/start'map_10/TensorArrayUnstack/strided_slice%map_10/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0
ј
Amap_10/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_10/TensorArraymap_10/TensorArrayUnstack/rangePlaceholder_10map_10/TensorArray:1*
T0*!
_class
loc:@Placeholder_10*
_output_shapes
: 
N
map_10/ConstConst*
_output_shapes
: *
value	B : *
dtype0
т
map_10/TensorArray_1TensorArrayV3map_10/strided_slice*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( 
`
map_10/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Ж
map_10/while/EnterEntermap_10/while/iteration_counter*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_10/while/while_context
І
map_10/while/Enter_1Entermap_10/Const*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_10/while/while_context
А
map_10/while/Enter_2Entermap_10/TensorArray_1:1*
parallel_iterations
*
_output_shapes
: **

frame_namemap_10/while/while_context*
T0*
is_constant( 
w
map_10/while/MergeMergemap_10/while/Entermap_10/while/NextIteration*
N*
_output_shapes
: : *
T0
}
map_10/while/Merge_1Mergemap_10/while/Enter_1map_10/while/NextIteration_1*
N*
_output_shapes
: : *
T0
}
map_10/while/Merge_2Mergemap_10/while/Enter_2map_10/while/NextIteration_2*
N*
_output_shapes
: : *
T0
g
map_10/while/LessLessmap_10/while/Mergemap_10/while/Less/Enter*
T0*
_output_shapes
: 
Б
map_10/while/Less/EnterEntermap_10/strided_slice*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: **

frame_namemap_10/while/while_context
k
map_10/while/Less_1Lessmap_10/while/Merge_1map_10/while/Less/Enter*
T0*
_output_shapes
: 
e
map_10/while/LogicalAnd
LogicalAndmap_10/while/Lessmap_10/while/Less_1*
_output_shapes
: 
R
map_10/while/LoopCondLoopCondmap_10/while/LogicalAnd*
_output_shapes
: 

map_10/while/SwitchSwitchmap_10/while/Mergemap_10/while/LoopCond*
T0*%
_class
loc:@map_10/while/Merge*
_output_shapes
: : 

map_10/while/Switch_1Switchmap_10/while/Merge_1map_10/while/LoopCond*
_output_shapes
: : *
T0*'
_class
loc:@map_10/while/Merge_1

map_10/while/Switch_2Switchmap_10/while/Merge_2map_10/while/LoopCond*'
_class
loc:@map_10/while/Merge_2*
_output_shapes
: : *
T0
Y
map_10/while/IdentityIdentitymap_10/while/Switch:1*
T0*
_output_shapes
: 
]
map_10/while/Identity_1Identitymap_10/while/Switch_1:1*
_output_shapes
: *
T0
]
map_10/while/Identity_2Identitymap_10/while/Switch_2:1*
T0*
_output_shapes
: 
l
map_10/while/add/yConst^map_10/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
map_10/while/addAddmap_10/while/Identitymap_10/while/add/y*
_output_shapes
: *
T0
П
map_10/while/TensorArrayReadV3TensorArrayReadV3$map_10/while/TensorArrayReadV3/Entermap_10/while/Identity_1&map_10/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Р
$map_10/while/TensorArrayReadV3/EnterEntermap_10/TensorArray*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
:**

frame_namemap_10/while/while_context
э
&map_10/while/TensorArrayReadV3/Enter_1EnterAmap_10/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: **

frame_namemap_10/while/while_context
ќ
map_10/while/DecodeJpeg
DecodeJpegmap_10/while/TensorArrayReadV3*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( 
|
"map_10/while/resize/ExpandDims/dimConst^map_10/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
И
map_10/while/resize/ExpandDims
ExpandDimsmap_10/while/DecodeJpeg"map_10/while/resize/ExpandDims/dim*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*

Tdim0*
T0

map_10/while/resize/sizeConst^map_10/while/Identity*
valueB",  ,  *
dtype0*
_output_shapes
:
Ж
"map_10/while/resize/ResizeBilinearResizeBilinearmap_10/while/resize/ExpandDimsmap_10/while/resize/size*
align_corners( *
T0*(
_output_shapes
:ЌЌ

map_10/while/resize/SqueezeSqueeze"map_10/while/resize/ResizeBilinear*
T0*$
_output_shapes
:ЌЌ*
squeeze_dims
 
o
map_10/while/div/yConst^map_10/while/Identity*
valueB
 *  C*
dtype0*
_output_shapes
: 
{
map_10/while/divRealDivmap_10/while/resize/Squeezemap_10/while/div/y*$
_output_shapes
:ЌЌ*
T0

"map_10/while/Reshape_Preproc/shapeConst^map_10/while/Identity*
dtype0*
_output_shapes
:*!
valueB",  ,     

map_10/while/Reshape_PreprocReshapemap_10/while/div"map_10/while/Reshape_Preproc/shape*
T0*
Tshape0*$
_output_shapes
:ЌЌ
 
0map_10/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36map_10/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_10/while/Identity_1map_10/while/Reshape_Preprocmap_10/while/Identity_2*
_output_shapes
: *
T0*/
_class%
#!loc:@map_10/while/Reshape_Preproc

6map_10/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_10/TensorArray_1*
T0*/
_class%
#!loc:@map_10/while/Reshape_Preproc*
parallel_iterations
*
is_constant(*
_output_shapes
:**

frame_namemap_10/while/while_context
n
map_10/while/add_1/yConst^map_10/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
map_10/while/add_1Addmap_10/while/Identity_1map_10/while/add_1/y*
T0*
_output_shapes
: 
^
map_10/while/NextIterationNextIterationmap_10/while/add*
_output_shapes
: *
T0
b
map_10/while/NextIteration_1NextIterationmap_10/while/add_1*
_output_shapes
: *
T0

map_10/while/NextIteration_2NextIteration0map_10/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
O
map_10/while/ExitExitmap_10/while/Switch*
T0*
_output_shapes
: 
S
map_10/while/Exit_1Exitmap_10/while/Switch_1*
T0*
_output_shapes
: 
S
map_10/while/Exit_2Exitmap_10/while/Switch_2*
_output_shapes
: *
T0
Њ
)map_10/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_10/TensorArray_1map_10/while/Exit_2*'
_class
loc:@map_10/TensorArray_1*
_output_shapes
: 

#map_10/TensorArrayStack/range/startConst*
value	B : *'
_class
loc:@map_10/TensorArray_1*
dtype0*
_output_shapes
: 

#map_10/TensorArrayStack/range/deltaConst*
value	B :*'
_class
loc:@map_10/TensorArray_1*
dtype0*
_output_shapes
: 
ѕ
map_10/TensorArrayStack/rangeRange#map_10/TensorArrayStack/range/start)map_10/TensorArrayStack/TensorArraySizeV3#map_10/TensorArrayStack/range/delta*'
_class
loc:@map_10/TensorArray_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0

+map_10/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_10/TensorArray_1map_10/TensorArrayStack/rangemap_10/while/Exit_2*!
element_shape:ЌЌ*'
_class
loc:@map_10/TensorArray_1*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ
\
ExpandDims_10/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ї
ExpandDims_10
ExpandDims+map_10/TensorArrayStack/TensorArrayGatherV3ExpandDims_10/dim*

Tdim0*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ
n
transpose_10/permConst*)
value B"                *
dtype0*
_output_shapes
:

transpose_10	TransposeExpandDims_10transpose_10/perm*
Tperm0*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ
Z
map_11/ShapeShapePlaceholder_11*
T0*
out_type0*
_output_shapes
:
d
map_11/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
f
map_11/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
map_11/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

map_11/strided_sliceStridedSlicemap_11/Shapemap_11/strided_slice/stackmap_11/strided_slice/stack_1map_11/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
р
map_11/TensorArrayTensorArrayV3map_11/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
m
map_11/TensorArrayUnstack/ShapeShapePlaceholder_11*
T0*
out_type0*
_output_shapes
:
w
-map_11/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
y
/map_11/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
y
/map_11/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ћ
'map_11/TensorArrayUnstack/strided_sliceStridedSlicemap_11/TensorArrayUnstack/Shape-map_11/TensorArrayUnstack/strided_slice/stack/map_11/TensorArrayUnstack/strided_slice/stack_1/map_11/TensorArrayUnstack/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
g
%map_11/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%map_11/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
а
map_11/TensorArrayUnstack/rangeRange%map_11/TensorArrayUnstack/range/start'map_11/TensorArrayUnstack/strided_slice%map_11/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ
ј
Amap_11/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_11/TensorArraymap_11/TensorArrayUnstack/rangePlaceholder_11map_11/TensorArray:1*
T0*!
_class
loc:@Placeholder_11*
_output_shapes
: 
N
map_11/ConstConst*
_output_shapes
: *
value	B : *
dtype0
т
map_11/TensorArray_1TensorArrayV3map_11/strided_slice*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( 
`
map_11/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Ж
map_11/while/EnterEntermap_11/while/iteration_counter*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_11/while/while_context
І
map_11/while/Enter_1Entermap_11/Const*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_11/while/while_context
А
map_11/while/Enter_2Entermap_11/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_11/while/while_context
w
map_11/while/MergeMergemap_11/while/Entermap_11/while/NextIteration*
N*
_output_shapes
: : *
T0
}
map_11/while/Merge_1Mergemap_11/while/Enter_1map_11/while/NextIteration_1*
N*
_output_shapes
: : *
T0
}
map_11/while/Merge_2Mergemap_11/while/Enter_2map_11/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
g
map_11/while/LessLessmap_11/while/Mergemap_11/while/Less/Enter*
T0*
_output_shapes
: 
Б
map_11/while/Less/EnterEntermap_11/strided_slice*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: **

frame_namemap_11/while/while_context
k
map_11/while/Less_1Lessmap_11/while/Merge_1map_11/while/Less/Enter*
T0*
_output_shapes
: 
e
map_11/while/LogicalAnd
LogicalAndmap_11/while/Lessmap_11/while/Less_1*
_output_shapes
: 
R
map_11/while/LoopCondLoopCondmap_11/while/LogicalAnd*
_output_shapes
: 

map_11/while/SwitchSwitchmap_11/while/Mergemap_11/while/LoopCond*
T0*%
_class
loc:@map_11/while/Merge*
_output_shapes
: : 

map_11/while/Switch_1Switchmap_11/while/Merge_1map_11/while/LoopCond*
_output_shapes
: : *
T0*'
_class
loc:@map_11/while/Merge_1

map_11/while/Switch_2Switchmap_11/while/Merge_2map_11/while/LoopCond*
_output_shapes
: : *
T0*'
_class
loc:@map_11/while/Merge_2
Y
map_11/while/IdentityIdentitymap_11/while/Switch:1*
T0*
_output_shapes
: 
]
map_11/while/Identity_1Identitymap_11/while/Switch_1:1*
T0*
_output_shapes
: 
]
map_11/while/Identity_2Identitymap_11/while/Switch_2:1*
T0*
_output_shapes
: 
l
map_11/while/add/yConst^map_11/while/Identity*
dtype0*
_output_shapes
: *
value	B :
c
map_11/while/addAddmap_11/while/Identitymap_11/while/add/y*
T0*
_output_shapes
: 
П
map_11/while/TensorArrayReadV3TensorArrayReadV3$map_11/while/TensorArrayReadV3/Entermap_11/while/Identity_1&map_11/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Р
$map_11/while/TensorArrayReadV3/EnterEntermap_11/TensorArray*
is_constant(*
parallel_iterations
*
_output_shapes
:**

frame_namemap_11/while/while_context*
T0
э
&map_11/while/TensorArrayReadV3/Enter_1EnterAmap_11/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: **

frame_namemap_11/while/while_context*
T0*
is_constant(*
parallel_iterations

ќ
map_11/while/DecodeJpeg
DecodeJpegmap_11/while/TensorArrayReadV3*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( 
|
"map_11/while/resize/ExpandDims/dimConst^map_11/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
И
map_11/while/resize/ExpandDims
ExpandDimsmap_11/while/DecodeJpeg"map_11/while/resize/ExpandDims/dim*

Tdim0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

map_11/while/resize/sizeConst^map_11/while/Identity*
valueB",  ,  *
dtype0*
_output_shapes
:
Ж
"map_11/while/resize/ResizeBilinearResizeBilinearmap_11/while/resize/ExpandDimsmap_11/while/resize/size*(
_output_shapes
:ЌЌ*
align_corners( *
T0

map_11/while/resize/SqueezeSqueeze"map_11/while/resize/ResizeBilinear*$
_output_shapes
:ЌЌ*
squeeze_dims
 *
T0
o
map_11/while/div/yConst^map_11/while/Identity*
valueB
 *  C*
dtype0*
_output_shapes
: 
{
map_11/while/divRealDivmap_11/while/resize/Squeezemap_11/while/div/y*
T0*$
_output_shapes
:ЌЌ

"map_11/while/Reshape_Preproc/shapeConst^map_11/while/Identity*!
valueB",  ,     *
dtype0*
_output_shapes
:

map_11/while/Reshape_PreprocReshapemap_11/while/div"map_11/while/Reshape_Preproc/shape*
T0*
Tshape0*$
_output_shapes
:ЌЌ
 
0map_11/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36map_11/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_11/while/Identity_1map_11/while/Reshape_Preprocmap_11/while/Identity_2*
T0*/
_class%
#!loc:@map_11/while/Reshape_Preproc*
_output_shapes
: 

6map_11/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_11/TensorArray_1*
_output_shapes
:**

frame_namemap_11/while/while_context*
T0*/
_class%
#!loc:@map_11/while/Reshape_Preproc*
parallel_iterations
*
is_constant(
n
map_11/while/add_1/yConst^map_11/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
map_11/while/add_1Addmap_11/while/Identity_1map_11/while/add_1/y*
T0*
_output_shapes
: 
^
map_11/while/NextIterationNextIterationmap_11/while/add*
T0*
_output_shapes
: 
b
map_11/while/NextIteration_1NextIterationmap_11/while/add_1*
T0*
_output_shapes
: 

map_11/while/NextIteration_2NextIteration0map_11/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
O
map_11/while/ExitExitmap_11/while/Switch*
_output_shapes
: *
T0
S
map_11/while/Exit_1Exitmap_11/while/Switch_1*
_output_shapes
: *
T0
S
map_11/while/Exit_2Exitmap_11/while/Switch_2*
_output_shapes
: *
T0
Њ
)map_11/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_11/TensorArray_1map_11/while/Exit_2*'
_class
loc:@map_11/TensorArray_1*
_output_shapes
: 

#map_11/TensorArrayStack/range/startConst*
_output_shapes
: *
value	B : *'
_class
loc:@map_11/TensorArray_1*
dtype0

#map_11/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*'
_class
loc:@map_11/TensorArray_1
ѕ
map_11/TensorArrayStack/rangeRange#map_11/TensorArrayStack/range/start)map_11/TensorArrayStack/TensorArraySizeV3#map_11/TensorArrayStack/range/delta*'
_class
loc:@map_11/TensorArray_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0

+map_11/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_11/TensorArray_1map_11/TensorArrayStack/rangemap_11/while/Exit_2*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ*!
element_shape:ЌЌ*'
_class
loc:@map_11/TensorArray_1
\
ExpandDims_11/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ї
ExpandDims_11
ExpandDims+map_11/TensorArrayStack/TensorArrayGatherV3ExpandDims_11/dim*5
_output_shapes#
!:џџџџџџџџџЌЌ*

Tdim0*
T0
n
transpose_11/permConst*)
value B"                *
dtype0*
_output_shapes
:

transpose_11	TransposeExpandDims_11transpose_11/perm*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0*
T0
Z
map_12/ShapeShapePlaceholder_12*
T0*
out_type0*
_output_shapes
:
d
map_12/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
map_12/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
map_12/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

map_12/strided_sliceStridedSlicemap_12/Shapemap_12/strided_slice/stackmap_12/strided_slice/stack_1map_12/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
р
map_12/TensorArrayTensorArrayV3map_12/strided_slice*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(
m
map_12/TensorArrayUnstack/ShapeShapePlaceholder_12*
T0*
out_type0*
_output_shapes
:
w
-map_12/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
y
/map_12/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/map_12/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ћ
'map_12/TensorArrayUnstack/strided_sliceStridedSlicemap_12/TensorArrayUnstack/Shape-map_12/TensorArrayUnstack/strided_slice/stack/map_12/TensorArrayUnstack/strided_slice/stack_1/map_12/TensorArrayUnstack/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
g
%map_12/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%map_12/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
а
map_12/TensorArrayUnstack/rangeRange%map_12/TensorArrayUnstack/range/start'map_12/TensorArrayUnstack/strided_slice%map_12/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ
ј
Amap_12/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_12/TensorArraymap_12/TensorArrayUnstack/rangePlaceholder_12map_12/TensorArray:1*
T0*!
_class
loc:@Placeholder_12*
_output_shapes
: 
N
map_12/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
т
map_12/TensorArray_1TensorArrayV3map_12/strided_slice*
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: 
`
map_12/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Ж
map_12/while/EnterEntermap_12/while/iteration_counter*
_output_shapes
: **

frame_namemap_12/while/while_context*
T0*
is_constant( *
parallel_iterations

І
map_12/while/Enter_1Entermap_12/Const*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_12/while/while_context
А
map_12/while/Enter_2Entermap_12/TensorArray_1:1*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_12/while/while_context*
T0
w
map_12/while/MergeMergemap_12/while/Entermap_12/while/NextIteration*
N*
_output_shapes
: : *
T0
}
map_12/while/Merge_1Mergemap_12/while/Enter_1map_12/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
}
map_12/while/Merge_2Mergemap_12/while/Enter_2map_12/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
g
map_12/while/LessLessmap_12/while/Mergemap_12/while/Less/Enter*
T0*
_output_shapes
: 
Б
map_12/while/Less/EnterEntermap_12/strided_slice*
_output_shapes
: **

frame_namemap_12/while/while_context*
T0*
is_constant(*
parallel_iterations

k
map_12/while/Less_1Lessmap_12/while/Merge_1map_12/while/Less/Enter*
T0*
_output_shapes
: 
e
map_12/while/LogicalAnd
LogicalAndmap_12/while/Lessmap_12/while/Less_1*
_output_shapes
: 
R
map_12/while/LoopCondLoopCondmap_12/while/LogicalAnd*
_output_shapes
: 

map_12/while/SwitchSwitchmap_12/while/Mergemap_12/while/LoopCond*
T0*%
_class
loc:@map_12/while/Merge*
_output_shapes
: : 

map_12/while/Switch_1Switchmap_12/while/Merge_1map_12/while/LoopCond*
T0*'
_class
loc:@map_12/while/Merge_1*
_output_shapes
: : 

map_12/while/Switch_2Switchmap_12/while/Merge_2map_12/while/LoopCond*
T0*'
_class
loc:@map_12/while/Merge_2*
_output_shapes
: : 
Y
map_12/while/IdentityIdentitymap_12/while/Switch:1*
_output_shapes
: *
T0
]
map_12/while/Identity_1Identitymap_12/while/Switch_1:1*
_output_shapes
: *
T0
]
map_12/while/Identity_2Identitymap_12/while/Switch_2:1*
_output_shapes
: *
T0
l
map_12/while/add/yConst^map_12/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
map_12/while/addAddmap_12/while/Identitymap_12/while/add/y*
_output_shapes
: *
T0
П
map_12/while/TensorArrayReadV3TensorArrayReadV3$map_12/while/TensorArrayReadV3/Entermap_12/while/Identity_1&map_12/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Р
$map_12/while/TensorArrayReadV3/EnterEntermap_12/TensorArray*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
:**

frame_namemap_12/while/while_context
э
&map_12/while/TensorArrayReadV3/Enter_1EnterAmap_12/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations
*
_output_shapes
: **

frame_namemap_12/while/while_context*
T0*
is_constant(
ќ
map_12/while/DecodeJpeg
DecodeJpegmap_12/while/TensorArrayReadV3*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio
|
"map_12/while/resize/ExpandDims/dimConst^map_12/while/Identity*
_output_shapes
: *
value	B : *
dtype0
И
map_12/while/resize/ExpandDims
ExpandDimsmap_12/while/DecodeJpeg"map_12/while/resize/ExpandDims/dim*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*

Tdim0*
T0

map_12/while/resize/sizeConst^map_12/while/Identity*
valueB",  ,  *
dtype0*
_output_shapes
:
Ж
"map_12/while/resize/ResizeBilinearResizeBilinearmap_12/while/resize/ExpandDimsmap_12/while/resize/size*
align_corners( *
T0*(
_output_shapes
:ЌЌ

map_12/while/resize/SqueezeSqueeze"map_12/while/resize/ResizeBilinear*$
_output_shapes
:ЌЌ*
squeeze_dims
 *
T0
o
map_12/while/div/yConst^map_12/while/Identity*
_output_shapes
: *
valueB
 *  C*
dtype0
{
map_12/while/divRealDivmap_12/while/resize/Squeezemap_12/while/div/y*$
_output_shapes
:ЌЌ*
T0

"map_12/while/Reshape_Preproc/shapeConst^map_12/while/Identity*!
valueB",  ,     *
dtype0*
_output_shapes
:

map_12/while/Reshape_PreprocReshapemap_12/while/div"map_12/while/Reshape_Preproc/shape*
T0*
Tshape0*$
_output_shapes
:ЌЌ
 
0map_12/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36map_12/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_12/while/Identity_1map_12/while/Reshape_Preprocmap_12/while/Identity_2*
T0*/
_class%
#!loc:@map_12/while/Reshape_Preproc*
_output_shapes
: 

6map_12/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_12/TensorArray_1*
is_constant(*
_output_shapes
:**

frame_namemap_12/while/while_context*
T0*/
_class%
#!loc:@map_12/while/Reshape_Preproc*
parallel_iterations

n
map_12/while/add_1/yConst^map_12/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
map_12/while/add_1Addmap_12/while/Identity_1map_12/while/add_1/y*
T0*
_output_shapes
: 
^
map_12/while/NextIterationNextIterationmap_12/while/add*
_output_shapes
: *
T0
b
map_12/while/NextIteration_1NextIterationmap_12/while/add_1*
T0*
_output_shapes
: 

map_12/while/NextIteration_2NextIteration0map_12/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
O
map_12/while/ExitExitmap_12/while/Switch*
_output_shapes
: *
T0
S
map_12/while/Exit_1Exitmap_12/while/Switch_1*
T0*
_output_shapes
: 
S
map_12/while/Exit_2Exitmap_12/while/Switch_2*
_output_shapes
: *
T0
Њ
)map_12/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_12/TensorArray_1map_12/while/Exit_2*'
_class
loc:@map_12/TensorArray_1*
_output_shapes
: 

#map_12/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *
value	B : *'
_class
loc:@map_12/TensorArray_1

#map_12/TensorArrayStack/range/deltaConst*
value	B :*'
_class
loc:@map_12/TensorArray_1*
dtype0*
_output_shapes
: 
ѕ
map_12/TensorArrayStack/rangeRange#map_12/TensorArrayStack/range/start)map_12/TensorArrayStack/TensorArraySizeV3#map_12/TensorArrayStack/range/delta*

Tidx0*'
_class
loc:@map_12/TensorArray_1*#
_output_shapes
:џџџџџџџџџ

+map_12/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_12/TensorArray_1map_12/TensorArrayStack/rangemap_12/while/Exit_2*1
_output_shapes
:џџџџџџџџџЌЌ*!
element_shape:ЌЌ*'
_class
loc:@map_12/TensorArray_1*
dtype0
\
ExpandDims_12/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ї
ExpandDims_12
ExpandDims+map_12/TensorArrayStack/TensorArrayGatherV3ExpandDims_12/dim*

Tdim0*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ
n
transpose_12/permConst*)
value B"                *
dtype0*
_output_shapes
:

transpose_12	TransposeExpandDims_12transpose_12/perm*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0
Z
map_13/ShapeShapePlaceholder_13*
_output_shapes
:*
T0*
out_type0
d
map_13/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
map_13/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
map_13/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

map_13/strided_sliceStridedSlicemap_13/Shapemap_13/strided_slice/stackmap_13/strided_slice/stack_1map_13/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
р
map_13/TensorArrayTensorArrayV3map_13/strided_slice*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( 
m
map_13/TensorArrayUnstack/ShapeShapePlaceholder_13*
_output_shapes
:*
T0*
out_type0
w
-map_13/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/map_13/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/map_13/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ћ
'map_13/TensorArrayUnstack/strided_sliceStridedSlicemap_13/TensorArrayUnstack/Shape-map_13/TensorArrayUnstack/strided_slice/stack/map_13/TensorArrayUnstack/strided_slice/stack_1/map_13/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
g
%map_13/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%map_13/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
а
map_13/TensorArrayUnstack/rangeRange%map_13/TensorArrayUnstack/range/start'map_13/TensorArrayUnstack/strided_slice%map_13/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ
ј
Amap_13/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_13/TensorArraymap_13/TensorArrayUnstack/rangePlaceholder_13map_13/TensorArray:1*
T0*!
_class
loc:@Placeholder_13*
_output_shapes
: 
N
map_13/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
т
map_13/TensorArray_1TensorArrayV3map_13/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
`
map_13/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Ж
map_13/while/EnterEntermap_13/while/iteration_counter*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_13/while/while_context
І
map_13/while/Enter_1Entermap_13/Const*
parallel_iterations
*
_output_shapes
: **

frame_namemap_13/while/while_context*
T0*
is_constant( 
А
map_13/while/Enter_2Entermap_13/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_13/while/while_context
w
map_13/while/MergeMergemap_13/while/Entermap_13/while/NextIteration*
T0*
N*
_output_shapes
: : 
}
map_13/while/Merge_1Mergemap_13/while/Enter_1map_13/while/NextIteration_1*
N*
_output_shapes
: : *
T0
}
map_13/while/Merge_2Mergemap_13/while/Enter_2map_13/while/NextIteration_2*
N*
_output_shapes
: : *
T0
g
map_13/while/LessLessmap_13/while/Mergemap_13/while/Less/Enter*
T0*
_output_shapes
: 
Б
map_13/while/Less/EnterEntermap_13/strided_slice*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: **

frame_namemap_13/while/while_context
k
map_13/while/Less_1Lessmap_13/while/Merge_1map_13/while/Less/Enter*
_output_shapes
: *
T0
e
map_13/while/LogicalAnd
LogicalAndmap_13/while/Lessmap_13/while/Less_1*
_output_shapes
: 
R
map_13/while/LoopCondLoopCondmap_13/while/LogicalAnd*
_output_shapes
: 

map_13/while/SwitchSwitchmap_13/while/Mergemap_13/while/LoopCond*
T0*%
_class
loc:@map_13/while/Merge*
_output_shapes
: : 

map_13/while/Switch_1Switchmap_13/while/Merge_1map_13/while/LoopCond*'
_class
loc:@map_13/while/Merge_1*
_output_shapes
: : *
T0

map_13/while/Switch_2Switchmap_13/while/Merge_2map_13/while/LoopCond*
T0*'
_class
loc:@map_13/while/Merge_2*
_output_shapes
: : 
Y
map_13/while/IdentityIdentitymap_13/while/Switch:1*
_output_shapes
: *
T0
]
map_13/while/Identity_1Identitymap_13/while/Switch_1:1*
_output_shapes
: *
T0
]
map_13/while/Identity_2Identitymap_13/while/Switch_2:1*
T0*
_output_shapes
: 
l
map_13/while/add/yConst^map_13/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
map_13/while/addAddmap_13/while/Identitymap_13/while/add/y*
T0*
_output_shapes
: 
П
map_13/while/TensorArrayReadV3TensorArrayReadV3$map_13/while/TensorArrayReadV3/Entermap_13/while/Identity_1&map_13/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Р
$map_13/while/TensorArrayReadV3/EnterEntermap_13/TensorArray*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
:**

frame_namemap_13/while/while_context
э
&map_13/while/TensorArrayReadV3/Enter_1EnterAmap_13/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: **

frame_namemap_13/while/while_context
ќ
map_13/while/DecodeJpeg
DecodeJpegmap_13/while/TensorArrayReadV3*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( 
|
"map_13/while/resize/ExpandDims/dimConst^map_13/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
И
map_13/while/resize/ExpandDims
ExpandDimsmap_13/while/DecodeJpeg"map_13/while/resize/ExpandDims/dim*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*

Tdim0*
T0

map_13/while/resize/sizeConst^map_13/while/Identity*
valueB",  ,  *
dtype0*
_output_shapes
:
Ж
"map_13/while/resize/ResizeBilinearResizeBilinearmap_13/while/resize/ExpandDimsmap_13/while/resize/size*
align_corners( *
T0*(
_output_shapes
:ЌЌ

map_13/while/resize/SqueezeSqueeze"map_13/while/resize/ResizeBilinear*
T0*$
_output_shapes
:ЌЌ*
squeeze_dims
 
o
map_13/while/div/yConst^map_13/while/Identity*
valueB
 *  C*
dtype0*
_output_shapes
: 
{
map_13/while/divRealDivmap_13/while/resize/Squeezemap_13/while/div/y*$
_output_shapes
:ЌЌ*
T0

"map_13/while/Reshape_Preproc/shapeConst^map_13/while/Identity*!
valueB",  ,     *
dtype0*
_output_shapes
:

map_13/while/Reshape_PreprocReshapemap_13/while/div"map_13/while/Reshape_Preproc/shape*$
_output_shapes
:ЌЌ*
T0*
Tshape0
 
0map_13/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36map_13/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_13/while/Identity_1map_13/while/Reshape_Preprocmap_13/while/Identity_2*
T0*/
_class%
#!loc:@map_13/while/Reshape_Preproc*
_output_shapes
: 

6map_13/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_13/TensorArray_1*/
_class%
#!loc:@map_13/while/Reshape_Preproc*
parallel_iterations
*
is_constant(*
_output_shapes
:**

frame_namemap_13/while/while_context*
T0
n
map_13/while/add_1/yConst^map_13/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
map_13/while/add_1Addmap_13/while/Identity_1map_13/while/add_1/y*
_output_shapes
: *
T0
^
map_13/while/NextIterationNextIterationmap_13/while/add*
T0*
_output_shapes
: 
b
map_13/while/NextIteration_1NextIterationmap_13/while/add_1*
_output_shapes
: *
T0

map_13/while/NextIteration_2NextIteration0map_13/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
O
map_13/while/ExitExitmap_13/while/Switch*
_output_shapes
: *
T0
S
map_13/while/Exit_1Exitmap_13/while/Switch_1*
T0*
_output_shapes
: 
S
map_13/while/Exit_2Exitmap_13/while/Switch_2*
_output_shapes
: *
T0
Њ
)map_13/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_13/TensorArray_1map_13/while/Exit_2*'
_class
loc:@map_13/TensorArray_1*
_output_shapes
: 

#map_13/TensorArrayStack/range/startConst*
value	B : *'
_class
loc:@map_13/TensorArray_1*
dtype0*
_output_shapes
: 

#map_13/TensorArrayStack/range/deltaConst*
_output_shapes
: *
value	B :*'
_class
loc:@map_13/TensorArray_1*
dtype0
ѕ
map_13/TensorArrayStack/rangeRange#map_13/TensorArrayStack/range/start)map_13/TensorArrayStack/TensorArraySizeV3#map_13/TensorArrayStack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0*'
_class
loc:@map_13/TensorArray_1

+map_13/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_13/TensorArray_1map_13/TensorArrayStack/rangemap_13/while/Exit_2*!
element_shape:ЌЌ*'
_class
loc:@map_13/TensorArray_1*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ
\
ExpandDims_13/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ї
ExpandDims_13
ExpandDims+map_13/TensorArrayStack/TensorArrayGatherV3ExpandDims_13/dim*5
_output_shapes#
!:џџџџџџџџџЌЌ*

Tdim0*
T0
n
transpose_13/permConst*)
value B"                *
dtype0*
_output_shapes
:

transpose_13	TransposeExpandDims_13transpose_13/perm*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0
Z
map_14/ShapeShapePlaceholder_14*
T0*
out_type0*
_output_shapes
:
d
map_14/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
map_14/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
f
map_14/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

map_14/strided_sliceStridedSlicemap_14/Shapemap_14/strided_slice/stackmap_14/strided_slice/stack_1map_14/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
р
map_14/TensorArrayTensorArrayV3map_14/strided_slice*
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: 
m
map_14/TensorArrayUnstack/ShapeShapePlaceholder_14*
T0*
out_type0*
_output_shapes
:
w
-map_14/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/map_14/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/map_14/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ћ
'map_14/TensorArrayUnstack/strided_sliceStridedSlicemap_14/TensorArrayUnstack/Shape-map_14/TensorArrayUnstack/strided_slice/stack/map_14/TensorArrayUnstack/strided_slice/stack_1/map_14/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
g
%map_14/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%map_14/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
а
map_14/TensorArrayUnstack/rangeRange%map_14/TensorArrayUnstack/range/start'map_14/TensorArrayUnstack/strided_slice%map_14/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0
ј
Amap_14/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_14/TensorArraymap_14/TensorArrayUnstack/rangePlaceholder_14map_14/TensorArray:1*!
_class
loc:@Placeholder_14*
_output_shapes
: *
T0
N
map_14/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
т
map_14/TensorArray_1TensorArrayV3map_14/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
`
map_14/while/iteration_counterConst*
_output_shapes
: *
value	B : *
dtype0
Ж
map_14/while/EnterEntermap_14/while/iteration_counter*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_14/while/while_context
І
map_14/while/Enter_1Entermap_14/Const*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_14/while/while_context
А
map_14/while/Enter_2Entermap_14/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_14/while/while_context
w
map_14/while/MergeMergemap_14/while/Entermap_14/while/NextIteration*
N*
_output_shapes
: : *
T0
}
map_14/while/Merge_1Mergemap_14/while/Enter_1map_14/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
}
map_14/while/Merge_2Mergemap_14/while/Enter_2map_14/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
g
map_14/while/LessLessmap_14/while/Mergemap_14/while/Less/Enter*
_output_shapes
: *
T0
Б
map_14/while/Less/EnterEntermap_14/strided_slice*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: **

frame_namemap_14/while/while_context
k
map_14/while/Less_1Lessmap_14/while/Merge_1map_14/while/Less/Enter*
T0*
_output_shapes
: 
e
map_14/while/LogicalAnd
LogicalAndmap_14/while/Lessmap_14/while/Less_1*
_output_shapes
: 
R
map_14/while/LoopCondLoopCondmap_14/while/LogicalAnd*
_output_shapes
: 

map_14/while/SwitchSwitchmap_14/while/Mergemap_14/while/LoopCond*
T0*%
_class
loc:@map_14/while/Merge*
_output_shapes
: : 

map_14/while/Switch_1Switchmap_14/while/Merge_1map_14/while/LoopCond*
T0*'
_class
loc:@map_14/while/Merge_1*
_output_shapes
: : 

map_14/while/Switch_2Switchmap_14/while/Merge_2map_14/while/LoopCond*
T0*'
_class
loc:@map_14/while/Merge_2*
_output_shapes
: : 
Y
map_14/while/IdentityIdentitymap_14/while/Switch:1*
T0*
_output_shapes
: 
]
map_14/while/Identity_1Identitymap_14/while/Switch_1:1*
_output_shapes
: *
T0
]
map_14/while/Identity_2Identitymap_14/while/Switch_2:1*
_output_shapes
: *
T0
l
map_14/while/add/yConst^map_14/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
c
map_14/while/addAddmap_14/while/Identitymap_14/while/add/y*
T0*
_output_shapes
: 
П
map_14/while/TensorArrayReadV3TensorArrayReadV3$map_14/while/TensorArrayReadV3/Entermap_14/while/Identity_1&map_14/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Р
$map_14/while/TensorArrayReadV3/EnterEntermap_14/TensorArray*
parallel_iterations
*
_output_shapes
:**

frame_namemap_14/while/while_context*
T0*
is_constant(
э
&map_14/while/TensorArrayReadV3/Enter_1EnterAmap_14/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: **

frame_namemap_14/while/while_context
ќ
map_14/while/DecodeJpeg
DecodeJpegmap_14/while/TensorArrayReadV3*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( 
|
"map_14/while/resize/ExpandDims/dimConst^map_14/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
И
map_14/while/resize/ExpandDims
ExpandDimsmap_14/while/DecodeJpeg"map_14/while/resize/ExpandDims/dim*

Tdim0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ

map_14/while/resize/sizeConst^map_14/while/Identity*
valueB",  ,  *
dtype0*
_output_shapes
:
Ж
"map_14/while/resize/ResizeBilinearResizeBilinearmap_14/while/resize/ExpandDimsmap_14/while/resize/size*
T0*(
_output_shapes
:ЌЌ*
align_corners( 

map_14/while/resize/SqueezeSqueeze"map_14/while/resize/ResizeBilinear*
squeeze_dims
 *
T0*$
_output_shapes
:ЌЌ
o
map_14/while/div/yConst^map_14/while/Identity*
_output_shapes
: *
valueB
 *  C*
dtype0
{
map_14/while/divRealDivmap_14/while/resize/Squeezemap_14/while/div/y*$
_output_shapes
:ЌЌ*
T0

"map_14/while/Reshape_Preproc/shapeConst^map_14/while/Identity*!
valueB",  ,     *
dtype0*
_output_shapes
:

map_14/while/Reshape_PreprocReshapemap_14/while/div"map_14/while/Reshape_Preproc/shape*
T0*
Tshape0*$
_output_shapes
:ЌЌ
 
0map_14/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36map_14/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_14/while/Identity_1map_14/while/Reshape_Preprocmap_14/while/Identity_2*
_output_shapes
: *
T0*/
_class%
#!loc:@map_14/while/Reshape_Preproc

6map_14/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_14/TensorArray_1*
is_constant(*
_output_shapes
:**

frame_namemap_14/while/while_context*
T0*/
_class%
#!loc:@map_14/while/Reshape_Preproc*
parallel_iterations

n
map_14/while/add_1/yConst^map_14/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
map_14/while/add_1Addmap_14/while/Identity_1map_14/while/add_1/y*
_output_shapes
: *
T0
^
map_14/while/NextIterationNextIterationmap_14/while/add*
T0*
_output_shapes
: 
b
map_14/while/NextIteration_1NextIterationmap_14/while/add_1*
T0*
_output_shapes
: 

map_14/while/NextIteration_2NextIteration0map_14/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
O
map_14/while/ExitExitmap_14/while/Switch*
T0*
_output_shapes
: 
S
map_14/while/Exit_1Exitmap_14/while/Switch_1*
T0*
_output_shapes
: 
S
map_14/while/Exit_2Exitmap_14/while/Switch_2*
T0*
_output_shapes
: 
Њ
)map_14/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_14/TensorArray_1map_14/while/Exit_2*'
_class
loc:@map_14/TensorArray_1*
_output_shapes
: 

#map_14/TensorArrayStack/range/startConst*
value	B : *'
_class
loc:@map_14/TensorArray_1*
dtype0*
_output_shapes
: 

#map_14/TensorArrayStack/range/deltaConst*
value	B :*'
_class
loc:@map_14/TensorArray_1*
dtype0*
_output_shapes
: 
ѕ
map_14/TensorArrayStack/rangeRange#map_14/TensorArrayStack/range/start)map_14/TensorArrayStack/TensorArraySizeV3#map_14/TensorArrayStack/range/delta*

Tidx0*'
_class
loc:@map_14/TensorArray_1*#
_output_shapes
:џџџџџџџџџ

+map_14/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_14/TensorArray_1map_14/TensorArrayStack/rangemap_14/while/Exit_2*!
element_shape:ЌЌ*'
_class
loc:@map_14/TensorArray_1*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ
\
ExpandDims_14/dimConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0
Ї
ExpandDims_14
ExpandDims+map_14/TensorArrayStack/TensorArrayGatherV3ExpandDims_14/dim*

Tdim0*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ
n
transpose_14/permConst*)
value B"                *
dtype0*
_output_shapes
:

transpose_14	TransposeExpandDims_14transpose_14/perm*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0*
T0
Z
map_15/ShapeShapePlaceholder_15*
T0*
out_type0*
_output_shapes
:
d
map_15/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
map_15/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
f
map_15/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

map_15/strided_sliceStridedSlicemap_15/Shapemap_15/strided_slice/stackmap_15/strided_slice/stack_1map_15/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
р
map_15/TensorArrayTensorArrayV3map_15/strided_slice*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:
m
map_15/TensorArrayUnstack/ShapeShapePlaceholder_15*
_output_shapes
:*
T0*
out_type0
w
-map_15/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/map_15/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/map_15/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ћ
'map_15/TensorArrayUnstack/strided_sliceStridedSlicemap_15/TensorArrayUnstack/Shape-map_15/TensorArrayUnstack/strided_slice/stack/map_15/TensorArrayUnstack/strided_slice/stack_1/map_15/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
g
%map_15/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
g
%map_15/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
а
map_15/TensorArrayUnstack/rangeRange%map_15/TensorArrayUnstack/range/start'map_15/TensorArrayUnstack/strided_slice%map_15/TensorArrayUnstack/range/delta*#
_output_shapes
:џџџџџџџџџ*

Tidx0
ј
Amap_15/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map_15/TensorArraymap_15/TensorArrayUnstack/rangePlaceholder_15map_15/TensorArray:1*
T0*!
_class
loc:@Placeholder_15*
_output_shapes
: 
N
map_15/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
т
map_15/TensorArray_1TensorArrayV3map_15/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
`
map_15/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
Ж
map_15/while/EnterEntermap_15/while/iteration_counter*
_output_shapes
: **

frame_namemap_15/while/while_context*
T0*
is_constant( *
parallel_iterations

І
map_15/while/Enter_1Entermap_15/Const*
parallel_iterations
*
_output_shapes
: **

frame_namemap_15/while/while_context*
T0*
is_constant( 
А
map_15/while/Enter_2Entermap_15/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: **

frame_namemap_15/while/while_context
w
map_15/while/MergeMergemap_15/while/Entermap_15/while/NextIteration*
T0*
N*
_output_shapes
: : 
}
map_15/while/Merge_1Mergemap_15/while/Enter_1map_15/while/NextIteration_1*
_output_shapes
: : *
T0*
N
}
map_15/while/Merge_2Mergemap_15/while/Enter_2map_15/while/NextIteration_2*
_output_shapes
: : *
T0*
N
g
map_15/while/LessLessmap_15/while/Mergemap_15/while/Less/Enter*
T0*
_output_shapes
: 
Б
map_15/while/Less/EnterEntermap_15/strided_slice*
parallel_iterations
*
_output_shapes
: **

frame_namemap_15/while/while_context*
T0*
is_constant(
k
map_15/while/Less_1Lessmap_15/while/Merge_1map_15/while/Less/Enter*
_output_shapes
: *
T0
e
map_15/while/LogicalAnd
LogicalAndmap_15/while/Lessmap_15/while/Less_1*
_output_shapes
: 
R
map_15/while/LoopCondLoopCondmap_15/while/LogicalAnd*
_output_shapes
: 

map_15/while/SwitchSwitchmap_15/while/Mergemap_15/while/LoopCond*%
_class
loc:@map_15/while/Merge*
_output_shapes
: : *
T0

map_15/while/Switch_1Switchmap_15/while/Merge_1map_15/while/LoopCond*
T0*'
_class
loc:@map_15/while/Merge_1*
_output_shapes
: : 

map_15/while/Switch_2Switchmap_15/while/Merge_2map_15/while/LoopCond*
T0*'
_class
loc:@map_15/while/Merge_2*
_output_shapes
: : 
Y
map_15/while/IdentityIdentitymap_15/while/Switch:1*
T0*
_output_shapes
: 
]
map_15/while/Identity_1Identitymap_15/while/Switch_1:1*
_output_shapes
: *
T0
]
map_15/while/Identity_2Identitymap_15/while/Switch_2:1*
T0*
_output_shapes
: 
l
map_15/while/add/yConst^map_15/while/Identity*
dtype0*
_output_shapes
: *
value	B :
c
map_15/while/addAddmap_15/while/Identitymap_15/while/add/y*
T0*
_output_shapes
: 
П
map_15/while/TensorArrayReadV3TensorArrayReadV3$map_15/while/TensorArrayReadV3/Entermap_15/while/Identity_1&map_15/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
Р
$map_15/while/TensorArrayReadV3/EnterEntermap_15/TensorArray*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
:**

frame_namemap_15/while/while_context
э
&map_15/while/TensorArrayReadV3/Enter_1EnterAmap_15/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations
*
_output_shapes
: **

frame_namemap_15/while/while_context*
T0*
is_constant(
ќ
map_15/while/DecodeJpeg
DecodeJpegmap_15/while/TensorArrayReadV3*
fancy_upscaling(*
try_recover_truncated( *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
ratio*

dct_method *
channels*
acceptable_fraction%  ?
|
"map_15/while/resize/ExpandDims/dimConst^map_15/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
И
map_15/while/resize/ExpandDims
ExpandDimsmap_15/while/DecodeJpeg"map_15/while/resize/ExpandDims/dim*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*

Tdim0*
T0

map_15/while/resize/sizeConst^map_15/while/Identity*
valueB",  ,  *
dtype0*
_output_shapes
:
Ж
"map_15/while/resize/ResizeBilinearResizeBilinearmap_15/while/resize/ExpandDimsmap_15/while/resize/size*(
_output_shapes
:ЌЌ*
align_corners( *
T0

map_15/while/resize/SqueezeSqueeze"map_15/while/resize/ResizeBilinear*
T0*$
_output_shapes
:ЌЌ*
squeeze_dims
 
o
map_15/while/div/yConst^map_15/while/Identity*
valueB
 *  C*
dtype0*
_output_shapes
: 
{
map_15/while/divRealDivmap_15/while/resize/Squeezemap_15/while/div/y*$
_output_shapes
:ЌЌ*
T0

"map_15/while/Reshape_Preproc/shapeConst^map_15/while/Identity*!
valueB",  ,     *
dtype0*
_output_shapes
:

map_15/while/Reshape_PreprocReshapemap_15/while/div"map_15/while/Reshape_Preproc/shape*$
_output_shapes
:ЌЌ*
T0*
Tshape0
 
0map_15/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36map_15/while/TensorArrayWrite/TensorArrayWriteV3/Entermap_15/while/Identity_1map_15/while/Reshape_Preprocmap_15/while/Identity_2*
_output_shapes
: *
T0*/
_class%
#!loc:@map_15/while/Reshape_Preproc

6map_15/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap_15/TensorArray_1*
T0*/
_class%
#!loc:@map_15/while/Reshape_Preproc*
parallel_iterations
*
is_constant(*
_output_shapes
:**

frame_namemap_15/while/while_context
n
map_15/while/add_1/yConst^map_15/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
map_15/while/add_1Addmap_15/while/Identity_1map_15/while/add_1/y*
T0*
_output_shapes
: 
^
map_15/while/NextIterationNextIterationmap_15/while/add*
T0*
_output_shapes
: 
b
map_15/while/NextIteration_1NextIterationmap_15/while/add_1*
_output_shapes
: *
T0

map_15/while/NextIteration_2NextIteration0map_15/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
O
map_15/while/ExitExitmap_15/while/Switch*
T0*
_output_shapes
: 
S
map_15/while/Exit_1Exitmap_15/while/Switch_1*
T0*
_output_shapes
: 
S
map_15/while/Exit_2Exitmap_15/while/Switch_2*
_output_shapes
: *
T0
Њ
)map_15/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map_15/TensorArray_1map_15/while/Exit_2*'
_class
loc:@map_15/TensorArray_1*
_output_shapes
: 

#map_15/TensorArrayStack/range/startConst*
value	B : *'
_class
loc:@map_15/TensorArray_1*
dtype0*
_output_shapes
: 

#map_15/TensorArrayStack/range/deltaConst*
value	B :*'
_class
loc:@map_15/TensorArray_1*
dtype0*
_output_shapes
: 
ѕ
map_15/TensorArrayStack/rangeRange#map_15/TensorArrayStack/range/start)map_15/TensorArrayStack/TensorArraySizeV3#map_15/TensorArrayStack/range/delta*'
_class
loc:@map_15/TensorArray_1*#
_output_shapes
:џџџџџџџџџ*

Tidx0

+map_15/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map_15/TensorArray_1map_15/TensorArrayStack/rangemap_15/while/Exit_2*'
_class
loc:@map_15/TensorArray_1*
dtype0*1
_output_shapes
:џџџџџџџџџЌЌ*!
element_shape:ЌЌ
\
ExpandDims_15/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ї
ExpandDims_15
ExpandDims+map_15/TensorArrayStack/TensorArrayGatherV3ExpandDims_15/dim*

Tdim0*
T0*5
_output_shapes#
!:џџџџџџџџџЌЌ
n
transpose_15/permConst*
dtype0*
_output_shapes
:*)
value B"                

transpose_15	TransposeExpandDims_15transpose_15/perm*5
_output_shapes#
!:џџџџџџџџџЌЌ*
Tperm0*
T0
M
concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
Ш
concatConcatV2	transposetranspose_1transpose_2transpose_3transpose_4transpose_5transpose_6transpose_7transpose_8transpose_9transpose_10transpose_11transpose_12transpose_13transpose_14transpose_15concat/axis*5
_output_shapes#
!:џџџџџџџџџЌЌ*

Tidx0*
T0*
N
Я
@AquamanNet/netconv_0_0/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*)
value B"                *0
_class&
$"loc:@AquamanNet/netconv_0_0/kernel*
dtype0
Ж
?AquamanNet/netconv_0_0/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@AquamanNet/netconv_0_0/kernel*
dtype0*
_output_shapes
: 
И
AAquamanNet/netconv_0_0/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *v7>*0
_class&
$"loc:@AquamanNet/netconv_0_0/kernel*
dtype0
Ќ
JAquamanNet/netconv_0_0/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/netconv_0_0/kernel/Initializer/truncated_normal/shape*

seed *
T0*0
_class&
$"loc:@AquamanNet/netconv_0_0/kernel*
seed2 *
dtype0**
_output_shapes
: 
Л
>AquamanNet/netconv_0_0/kernel/Initializer/truncated_normal/mulMulJAquamanNet/netconv_0_0/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/netconv_0_0/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@AquamanNet/netconv_0_0/kernel**
_output_shapes
: 
Љ
:AquamanNet/netconv_0_0/kernel/Initializer/truncated_normalAdd>AquamanNet/netconv_0_0/kernel/Initializer/truncated_normal/mul?AquamanNet/netconv_0_0/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@AquamanNet/netconv_0_0/kernel**
_output_shapes
: 
л
AquamanNet/netconv_0_0/kernel
VariableV2*
	container *
shape: *
dtype0**
_output_shapes
: *
shared_name *0
_class&
$"loc:@AquamanNet/netconv_0_0/kernel

$AquamanNet/netconv_0_0/kernel/AssignAssignAquamanNet/netconv_0_0/kernel:AquamanNet/netconv_0_0/kernel/Initializer/truncated_normal*
validate_shape(**
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_0_0/kernel
Д
"AquamanNet/netconv_0_0/kernel/readIdentityAquamanNet/netconv_0_0/kernel*
T0*0
_class&
$"loc:@AquamanNet/netconv_0_0/kernel**
_output_shapes
: 

>AquamanNet/netconv_0_0/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

?AquamanNet/netconv_0_0/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/netconv_0_0/kernel/read*
_output_shapes
: *
T0
с
8AquamanNet/netconv_0_0/kernel/Regularizer/l2_regularizerMul>AquamanNet/netconv_0_0/kernel/Regularizer/l2_regularizer/scale?AquamanNet/netconv_0_0/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
И
>AquamanNet/netconv_0_0/bias/Initializer/truncated_normal/shapeConst*
valueB: *.
_class$
" loc:@AquamanNet/netconv_0_0/bias*
dtype0*
_output_shapes
:
В
=AquamanNet/netconv_0_0/bias/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@AquamanNet/netconv_0_0/bias*
dtype0
Д
?AquamanNet/netconv_0_0/bias/Initializer/truncated_normal/stddevConst*
valueB
 *Eё>*.
_class$
" loc:@AquamanNet/netconv_0_0/bias*
dtype0*
_output_shapes
: 

HAquamanNet/netconv_0_0/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/netconv_0_0/bias/Initializer/truncated_normal/shape*
T0*.
_class$
" loc:@AquamanNet/netconv_0_0/bias*
seed2 *
dtype0*
_output_shapes
: *

seed 
Ѓ
<AquamanNet/netconv_0_0/bias/Initializer/truncated_normal/mulMulHAquamanNet/netconv_0_0/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/netconv_0_0/bias/Initializer/truncated_normal/stddev*
_output_shapes
: *
T0*.
_class$
" loc:@AquamanNet/netconv_0_0/bias

8AquamanNet/netconv_0_0/bias/Initializer/truncated_normalAdd<AquamanNet/netconv_0_0/bias/Initializer/truncated_normal/mul=AquamanNet/netconv_0_0/bias/Initializer/truncated_normal/mean*.
_class$
" loc:@AquamanNet/netconv_0_0/bias*
_output_shapes
: *
T0
З
AquamanNet/netconv_0_0/bias
VariableV2*
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@AquamanNet/netconv_0_0/bias*
	container *
shape: 

"AquamanNet/netconv_0_0/bias/AssignAssignAquamanNet/netconv_0_0/bias8AquamanNet/netconv_0_0/bias/Initializer/truncated_normal*.
_class$
" loc:@AquamanNet/netconv_0_0/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

 AquamanNet/netconv_0_0/bias/readIdentityAquamanNet/netconv_0_0/bias*
_output_shapes
: *
T0*.
_class$
" loc:@AquamanNet/netconv_0_0/bias

<AquamanNet/netconv_0_0/bias/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=AquamanNet/netconv_0_0/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/netconv_0_0/bias/read*
T0*
_output_shapes
: 
л
6AquamanNet/netconv_0_0/bias/Regularizer/l2_regularizerMul<AquamanNet/netconv_0_0/bias/Regularizer/l2_regularizer/scale=AquamanNet/netconv_0_0/bias/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
y
$AquamanNet/netconv_0_0/dilation_rateConst*!
valueB"         *
dtype0*
_output_shapes
:
э
AquamanNet/netconv_0_0/Conv3DConv3Dconcat"AquamanNet/netconv_0_0/kernel/read*
	dilations	
*
T0*
data_formatNDHWC*
strides	
*
paddingVALID*5
_output_shapes#
!:џџџџџџџџџЊЊ 
С
AquamanNet/netconv_0_0/BiasAddBiasAddAquamanNet/netconv_0_0/Conv3D AquamanNet/netconv_0_0/bias/read*
data_formatNHWC*5
_output_shapes#
!:џџџџџџџџџЊЊ *
T0

AquamanNet/netconv_0_0/EluEluAquamanNet/netconv_0_0/BiasAdd*
T0*5
_output_shapes#
!:џџџџџџџџџЊЊ 
Я
@AquamanNet/netconv_0_1/kernel/Initializer/truncated_normal/shapeConst*)
value B"                *0
_class&
$"loc:@AquamanNet/netconv_0_1/kernel*
dtype0*
_output_shapes
:
Ж
?AquamanNet/netconv_0_1/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@AquamanNet/netconv_0_1/kernel*
dtype0*
_output_shapes
: 
И
AAquamanNet/netconv_0_1/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *v7>*0
_class&
$"loc:@AquamanNet/netconv_0_1/kernel
Ќ
JAquamanNet/netconv_0_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/netconv_0_1/kernel/Initializer/truncated_normal/shape*
dtype0**
_output_shapes
: *

seed *
T0*0
_class&
$"loc:@AquamanNet/netconv_0_1/kernel*
seed2 
Л
>AquamanNet/netconv_0_1/kernel/Initializer/truncated_normal/mulMulJAquamanNet/netconv_0_1/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/netconv_0_1/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@AquamanNet/netconv_0_1/kernel**
_output_shapes
: 
Љ
:AquamanNet/netconv_0_1/kernel/Initializer/truncated_normalAdd>AquamanNet/netconv_0_1/kernel/Initializer/truncated_normal/mul?AquamanNet/netconv_0_1/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@AquamanNet/netconv_0_1/kernel**
_output_shapes
: 
л
AquamanNet/netconv_0_1/kernel
VariableV2*
	container *
shape: *
dtype0**
_output_shapes
: *
shared_name *0
_class&
$"loc:@AquamanNet/netconv_0_1/kernel

$AquamanNet/netconv_0_1/kernel/AssignAssignAquamanNet/netconv_0_1/kernel:AquamanNet/netconv_0_1/kernel/Initializer/truncated_normal*0
_class&
$"loc:@AquamanNet/netconv_0_1/kernel*
validate_shape(**
_output_shapes
: *
use_locking(*
T0
Д
"AquamanNet/netconv_0_1/kernel/readIdentityAquamanNet/netconv_0_1/kernel*0
_class&
$"loc:@AquamanNet/netconv_0_1/kernel**
_output_shapes
: *
T0

>AquamanNet/netconv_0_1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o:

?AquamanNet/netconv_0_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/netconv_0_1/kernel/read*
_output_shapes
: *
T0
с
8AquamanNet/netconv_0_1/kernel/Regularizer/l2_regularizerMul>AquamanNet/netconv_0_1/kernel/Regularizer/l2_regularizer/scale?AquamanNet/netconv_0_1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
И
>AquamanNet/netconv_0_1/bias/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB: *.
_class$
" loc:@AquamanNet/netconv_0_1/bias
В
=AquamanNet/netconv_0_1/bias/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@AquamanNet/netconv_0_1/bias*
dtype0
Д
?AquamanNet/netconv_0_1/bias/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *Eё>*.
_class$
" loc:@AquamanNet/netconv_0_1/bias*
dtype0

HAquamanNet/netconv_0_1/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/netconv_0_1/bias/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
: *

seed *
T0*.
_class$
" loc:@AquamanNet/netconv_0_1/bias*
seed2 
Ѓ
<AquamanNet/netconv_0_1/bias/Initializer/truncated_normal/mulMulHAquamanNet/netconv_0_1/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/netconv_0_1/bias/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@AquamanNet/netconv_0_1/bias*
_output_shapes
: 

8AquamanNet/netconv_0_1/bias/Initializer/truncated_normalAdd<AquamanNet/netconv_0_1/bias/Initializer/truncated_normal/mul=AquamanNet/netconv_0_1/bias/Initializer/truncated_normal/mean*.
_class$
" loc:@AquamanNet/netconv_0_1/bias*
_output_shapes
: *
T0
З
AquamanNet/netconv_0_1/bias
VariableV2*
shared_name *.
_class$
" loc:@AquamanNet/netconv_0_1/bias*
	container *
shape: *
dtype0*
_output_shapes
: 

"AquamanNet/netconv_0_1/bias/AssignAssignAquamanNet/netconv_0_1/bias8AquamanNet/netconv_0_1/bias/Initializer/truncated_normal*.
_class$
" loc:@AquamanNet/netconv_0_1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

 AquamanNet/netconv_0_1/bias/readIdentityAquamanNet/netconv_0_1/bias*
T0*.
_class$
" loc:@AquamanNet/netconv_0_1/bias*
_output_shapes
: 

<AquamanNet/netconv_0_1/bias/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o:

=AquamanNet/netconv_0_1/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/netconv_0_1/bias/read*
T0*
_output_shapes
: 
л
6AquamanNet/netconv_0_1/bias/Regularizer/l2_regularizerMul<AquamanNet/netconv_0_1/bias/Regularizer/l2_regularizer/scale=AquamanNet/netconv_0_1/bias/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
y
$AquamanNet/netconv_0_1/dilation_rateConst*!
valueB"         *
dtype0*
_output_shapes
:
э
AquamanNet/netconv_0_1/Conv3DConv3Dconcat"AquamanNet/netconv_0_1/kernel/read*
paddingVALID*5
_output_shapes#
!:џџџџџџџџџЊЊ *
	dilations	
*
T0*
data_formatNDHWC*
strides	

С
AquamanNet/netconv_0_1/BiasAddBiasAddAquamanNet/netconv_0_1/Conv3D AquamanNet/netconv_0_1/bias/read*
T0*
data_formatNHWC*5
_output_shapes#
!:џџџџџџџџџЊЊ 

AquamanNet/netconv_0_1/EluEluAquamanNet/netconv_0_1/BiasAdd*5
_output_shapes#
!:џџџџџџџџџЊЊ *
T0
п
!AquamanNet/netMaxPool_0/MaxPool3D	MaxPool3DAquamanNet/netconv_0_1/Elu*
data_formatNDHWC*
strides	
*
ksize	
*
paddingSAME*5
_output_shapes#
!:џџџџџџџџџ *
T0
Я
@AquamanNet/netconv_1_1/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*)
value B"                 *0
_class&
$"loc:@AquamanNet/netconv_1_1/kernel
Ж
?AquamanNet/netconv_1_1/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@AquamanNet/netconv_1_1/kernel*
dtype0*
_output_shapes
: 
И
AAquamanNet/netconv_1_1/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *=*0
_class&
$"loc:@AquamanNet/netconv_1_1/kernel*
dtype0*
_output_shapes
: 
Ќ
JAquamanNet/netconv_1_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/netconv_1_1/kernel/Initializer/truncated_normal/shape*
dtype0**
_output_shapes
:  *

seed *
T0*0
_class&
$"loc:@AquamanNet/netconv_1_1/kernel*
seed2 
Л
>AquamanNet/netconv_1_1/kernel/Initializer/truncated_normal/mulMulJAquamanNet/netconv_1_1/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/netconv_1_1/kernel/Initializer/truncated_normal/stddev**
_output_shapes
:  *
T0*0
_class&
$"loc:@AquamanNet/netconv_1_1/kernel
Љ
:AquamanNet/netconv_1_1/kernel/Initializer/truncated_normalAdd>AquamanNet/netconv_1_1/kernel/Initializer/truncated_normal/mul?AquamanNet/netconv_1_1/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@AquamanNet/netconv_1_1/kernel**
_output_shapes
:  
л
AquamanNet/netconv_1_1/kernel
VariableV2*
dtype0**
_output_shapes
:  *
shared_name *0
_class&
$"loc:@AquamanNet/netconv_1_1/kernel*
	container *
shape:  

$AquamanNet/netconv_1_1/kernel/AssignAssignAquamanNet/netconv_1_1/kernel:AquamanNet/netconv_1_1/kernel/Initializer/truncated_normal**
_output_shapes
:  *
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_1_1/kernel*
validate_shape(
Д
"AquamanNet/netconv_1_1/kernel/readIdentityAquamanNet/netconv_1_1/kernel*
T0*0
_class&
$"loc:@AquamanNet/netconv_1_1/kernel**
_output_shapes
:  

>AquamanNet/netconv_1_1/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
valueB
 *o:*
dtype0

?AquamanNet/netconv_1_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/netconv_1_1/kernel/read*
_output_shapes
: *
T0
с
8AquamanNet/netconv_1_1/kernel/Regularizer/l2_regularizerMul>AquamanNet/netconv_1_1/kernel/Regularizer/l2_regularizer/scale?AquamanNet/netconv_1_1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
И
>AquamanNet/netconv_1_1/bias/Initializer/truncated_normal/shapeConst*
valueB: *.
_class$
" loc:@AquamanNet/netconv_1_1/bias*
dtype0*
_output_shapes
:
В
=AquamanNet/netconv_1_1/bias/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@AquamanNet/netconv_1_1/bias*
dtype0
Д
?AquamanNet/netconv_1_1/bias/Initializer/truncated_normal/stddevConst*
valueB
 *Eё>*.
_class$
" loc:@AquamanNet/netconv_1_1/bias*
dtype0*
_output_shapes
: 

HAquamanNet/netconv_1_1/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/netconv_1_1/bias/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
: *

seed *
T0*.
_class$
" loc:@AquamanNet/netconv_1_1/bias*
seed2 
Ѓ
<AquamanNet/netconv_1_1/bias/Initializer/truncated_normal/mulMulHAquamanNet/netconv_1_1/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/netconv_1_1/bias/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@AquamanNet/netconv_1_1/bias*
_output_shapes
: 

8AquamanNet/netconv_1_1/bias/Initializer/truncated_normalAdd<AquamanNet/netconv_1_1/bias/Initializer/truncated_normal/mul=AquamanNet/netconv_1_1/bias/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@AquamanNet/netconv_1_1/bias*
_output_shapes
: 
З
AquamanNet/netconv_1_1/bias
VariableV2*
shared_name *.
_class$
" loc:@AquamanNet/netconv_1_1/bias*
	container *
shape: *
dtype0*
_output_shapes
: 

"AquamanNet/netconv_1_1/bias/AssignAssignAquamanNet/netconv_1_1/bias8AquamanNet/netconv_1_1/bias/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@AquamanNet/netconv_1_1/bias*
validate_shape(*
_output_shapes
: 

 AquamanNet/netconv_1_1/bias/readIdentityAquamanNet/netconv_1_1/bias*
T0*.
_class$
" loc:@AquamanNet/netconv_1_1/bias*
_output_shapes
: 

<AquamanNet/netconv_1_1/bias/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=AquamanNet/netconv_1_1/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/netconv_1_1/bias/read*
T0*
_output_shapes
: 
л
6AquamanNet/netconv_1_1/bias/Regularizer/l2_regularizerMul<AquamanNet/netconv_1_1/bias/Regularizer/l2_regularizer/scale=AquamanNet/netconv_1_1/bias/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
y
$AquamanNet/netconv_1_1/dilation_rateConst*!
valueB"         *
dtype0*
_output_shapes
:

AquamanNet/netconv_1_1/Conv3DConv3D!AquamanNet/netMaxPool_0/MaxPool3D"AquamanNet/netconv_1_1/kernel/read*
paddingVALID*5
_output_shapes#
!:џџџџџџџџџ *
	dilations	
*
T0*
data_formatNDHWC*
strides	

С
AquamanNet/netconv_1_1/BiasAddBiasAddAquamanNet/netconv_1_1/Conv3D AquamanNet/netconv_1_1/bias/read*
data_formatNHWC*5
_output_shapes#
!:џџџџџџџџџ *
T0

AquamanNet/netconv_1_1/EluEluAquamanNet/netconv_1_1/BiasAdd*
T0*5
_output_shapes#
!:џџџџџџџџџ 
Я
@AquamanNet/netconv_1_2/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*)
value B"                 *0
_class&
$"loc:@AquamanNet/netconv_1_2/kernel
Ж
?AquamanNet/netconv_1_2/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@AquamanNet/netconv_1_2/kernel*
dtype0*
_output_shapes
: 
И
AAquamanNet/netconv_1_2/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *=*0
_class&
$"loc:@AquamanNet/netconv_1_2/kernel
Ќ
JAquamanNet/netconv_1_2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/netconv_1_2/kernel/Initializer/truncated_normal/shape*

seed *
T0*0
_class&
$"loc:@AquamanNet/netconv_1_2/kernel*
seed2 *
dtype0**
_output_shapes
:  
Л
>AquamanNet/netconv_1_2/kernel/Initializer/truncated_normal/mulMulJAquamanNet/netconv_1_2/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/netconv_1_2/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@AquamanNet/netconv_1_2/kernel**
_output_shapes
:  
Љ
:AquamanNet/netconv_1_2/kernel/Initializer/truncated_normalAdd>AquamanNet/netconv_1_2/kernel/Initializer/truncated_normal/mul?AquamanNet/netconv_1_2/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@AquamanNet/netconv_1_2/kernel**
_output_shapes
:  
л
AquamanNet/netconv_1_2/kernel
VariableV2*
dtype0**
_output_shapes
:  *
shared_name *0
_class&
$"loc:@AquamanNet/netconv_1_2/kernel*
	container *
shape:  

$AquamanNet/netconv_1_2/kernel/AssignAssignAquamanNet/netconv_1_2/kernel:AquamanNet/netconv_1_2/kernel/Initializer/truncated_normal*
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_1_2/kernel*
validate_shape(**
_output_shapes
:  
Д
"AquamanNet/netconv_1_2/kernel/readIdentityAquamanNet/netconv_1_2/kernel**
_output_shapes
:  *
T0*0
_class&
$"loc:@AquamanNet/netconv_1_2/kernel

>AquamanNet/netconv_1_2/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

?AquamanNet/netconv_1_2/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/netconv_1_2/kernel/read*
_output_shapes
: *
T0
с
8AquamanNet/netconv_1_2/kernel/Regularizer/l2_regularizerMul>AquamanNet/netconv_1_2/kernel/Regularizer/l2_regularizer/scale?AquamanNet/netconv_1_2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
И
>AquamanNet/netconv_1_2/bias/Initializer/truncated_normal/shapeConst*
valueB: *.
_class$
" loc:@AquamanNet/netconv_1_2/bias*
dtype0*
_output_shapes
:
В
=AquamanNet/netconv_1_2/bias/Initializer/truncated_normal/meanConst*
valueB
 *    *.
_class$
" loc:@AquamanNet/netconv_1_2/bias*
dtype0*
_output_shapes
: 
Д
?AquamanNet/netconv_1_2/bias/Initializer/truncated_normal/stddevConst*
valueB
 *Eё>*.
_class$
" loc:@AquamanNet/netconv_1_2/bias*
dtype0*
_output_shapes
: 

HAquamanNet/netconv_1_2/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/netconv_1_2/bias/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
: *

seed *
T0*.
_class$
" loc:@AquamanNet/netconv_1_2/bias*
seed2 
Ѓ
<AquamanNet/netconv_1_2/bias/Initializer/truncated_normal/mulMulHAquamanNet/netconv_1_2/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/netconv_1_2/bias/Initializer/truncated_normal/stddev*.
_class$
" loc:@AquamanNet/netconv_1_2/bias*
_output_shapes
: *
T0

8AquamanNet/netconv_1_2/bias/Initializer/truncated_normalAdd<AquamanNet/netconv_1_2/bias/Initializer/truncated_normal/mul=AquamanNet/netconv_1_2/bias/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@AquamanNet/netconv_1_2/bias*
_output_shapes
: 
З
AquamanNet/netconv_1_2/bias
VariableV2*
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@AquamanNet/netconv_1_2/bias*
	container *
shape: 

"AquamanNet/netconv_1_2/bias/AssignAssignAquamanNet/netconv_1_2/bias8AquamanNet/netconv_1_2/bias/Initializer/truncated_normal*.
_class$
" loc:@AquamanNet/netconv_1_2/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

 AquamanNet/netconv_1_2/bias/readIdentityAquamanNet/netconv_1_2/bias*
T0*.
_class$
" loc:@AquamanNet/netconv_1_2/bias*
_output_shapes
: 

<AquamanNet/netconv_1_2/bias/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=AquamanNet/netconv_1_2/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/netconv_1_2/bias/read*
T0*
_output_shapes
: 
л
6AquamanNet/netconv_1_2/bias/Regularizer/l2_regularizerMul<AquamanNet/netconv_1_2/bias/Regularizer/l2_regularizer/scale=AquamanNet/netconv_1_2/bias/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
y
$AquamanNet/netconv_1_2/dilation_rateConst*!
valueB"         *
dtype0*
_output_shapes
:

AquamanNet/netconv_1_2/Conv3DConv3DAquamanNet/netconv_1_1/Elu"AquamanNet/netconv_1_2/kernel/read*
paddingVALID*5
_output_shapes#
!:џџџџџџџџџ *
	dilations	
*
T0*
data_formatNDHWC*
strides	

С
AquamanNet/netconv_1_2/BiasAddBiasAddAquamanNet/netconv_1_2/Conv3D AquamanNet/netconv_1_2/bias/read*
T0*
data_formatNHWC*5
_output_shapes#
!:џџџџџџџџџ 

AquamanNet/netconv_1_2/EluEluAquamanNet/netconv_1_2/BiasAdd*5
_output_shapes#
!:џџџџџџџџџ *
T0
н
!AquamanNet/netMaxPool_1/MaxPool3D	MaxPool3DAquamanNet/netconv_1_2/Elu*
ksize	
*
paddingSAME*3
_output_shapes!
:џџџџџџџџџII *
T0*
data_formatNDHWC*
strides	

Я
@AquamanNet/netconv_2_1/kernel/Initializer/truncated_normal/shapeConst*)
value B"                 *0
_class&
$"loc:@AquamanNet/netconv_2_1/kernel*
dtype0*
_output_shapes
:
Ж
?AquamanNet/netconv_2_1/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@AquamanNet/netconv_2_1/kernel*
dtype0*
_output_shapes
: 
И
AAquamanNet/netconv_2_1/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *=*0
_class&
$"loc:@AquamanNet/netconv_2_1/kernel*
dtype0*
_output_shapes
: 
Ќ
JAquamanNet/netconv_2_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/netconv_2_1/kernel/Initializer/truncated_normal/shape*
dtype0**
_output_shapes
:  *

seed *
T0*0
_class&
$"loc:@AquamanNet/netconv_2_1/kernel*
seed2 
Л
>AquamanNet/netconv_2_1/kernel/Initializer/truncated_normal/mulMulJAquamanNet/netconv_2_1/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/netconv_2_1/kernel/Initializer/truncated_normal/stddev*0
_class&
$"loc:@AquamanNet/netconv_2_1/kernel**
_output_shapes
:  *
T0
Љ
:AquamanNet/netconv_2_1/kernel/Initializer/truncated_normalAdd>AquamanNet/netconv_2_1/kernel/Initializer/truncated_normal/mul?AquamanNet/netconv_2_1/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@AquamanNet/netconv_2_1/kernel**
_output_shapes
:  
л
AquamanNet/netconv_2_1/kernel
VariableV2*0
_class&
$"loc:@AquamanNet/netconv_2_1/kernel*
	container *
shape:  *
dtype0**
_output_shapes
:  *
shared_name 

$AquamanNet/netconv_2_1/kernel/AssignAssignAquamanNet/netconv_2_1/kernel:AquamanNet/netconv_2_1/kernel/Initializer/truncated_normal*
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_2_1/kernel*
validate_shape(**
_output_shapes
:  
Д
"AquamanNet/netconv_2_1/kernel/readIdentityAquamanNet/netconv_2_1/kernel*
T0*0
_class&
$"loc:@AquamanNet/netconv_2_1/kernel**
_output_shapes
:  

>AquamanNet/netconv_2_1/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

?AquamanNet/netconv_2_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/netconv_2_1/kernel/read*
_output_shapes
: *
T0
с
8AquamanNet/netconv_2_1/kernel/Regularizer/l2_regularizerMul>AquamanNet/netconv_2_1/kernel/Regularizer/l2_regularizer/scale?AquamanNet/netconv_2_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
И
>AquamanNet/netconv_2_1/bias/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB: *.
_class$
" loc:@AquamanNet/netconv_2_1/bias
В
=AquamanNet/netconv_2_1/bias/Initializer/truncated_normal/meanConst*
valueB
 *    *.
_class$
" loc:@AquamanNet/netconv_2_1/bias*
dtype0*
_output_shapes
: 
Д
?AquamanNet/netconv_2_1/bias/Initializer/truncated_normal/stddevConst*
valueB
 *Eё>*.
_class$
" loc:@AquamanNet/netconv_2_1/bias*
dtype0*
_output_shapes
: 

HAquamanNet/netconv_2_1/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/netconv_2_1/bias/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
: *

seed *
T0*.
_class$
" loc:@AquamanNet/netconv_2_1/bias*
seed2 
Ѓ
<AquamanNet/netconv_2_1/bias/Initializer/truncated_normal/mulMulHAquamanNet/netconv_2_1/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/netconv_2_1/bias/Initializer/truncated_normal/stddev*
_output_shapes
: *
T0*.
_class$
" loc:@AquamanNet/netconv_2_1/bias

8AquamanNet/netconv_2_1/bias/Initializer/truncated_normalAdd<AquamanNet/netconv_2_1/bias/Initializer/truncated_normal/mul=AquamanNet/netconv_2_1/bias/Initializer/truncated_normal/mean*
_output_shapes
: *
T0*.
_class$
" loc:@AquamanNet/netconv_2_1/bias
З
AquamanNet/netconv_2_1/bias
VariableV2*
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@AquamanNet/netconv_2_1/bias*
	container *
shape: 

"AquamanNet/netconv_2_1/bias/AssignAssignAquamanNet/netconv_2_1/bias8AquamanNet/netconv_2_1/bias/Initializer/truncated_normal*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@AquamanNet/netconv_2_1/bias*
validate_shape(

 AquamanNet/netconv_2_1/bias/readIdentityAquamanNet/netconv_2_1/bias*
T0*.
_class$
" loc:@AquamanNet/netconv_2_1/bias*
_output_shapes
: 

<AquamanNet/netconv_2_1/bias/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=AquamanNet/netconv_2_1/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/netconv_2_1/bias/read*
T0*
_output_shapes
: 
л
6AquamanNet/netconv_2_1/bias/Regularizer/l2_regularizerMul<AquamanNet/netconv_2_1/bias/Regularizer/l2_regularizer/scale=AquamanNet/netconv_2_1/bias/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
y
$AquamanNet/netconv_2_1/dilation_rateConst*
dtype0*
_output_shapes
:*!
valueB"         

AquamanNet/netconv_2_1/Conv3DConv3D!AquamanNet/netMaxPool_1/MaxPool3D"AquamanNet/netconv_2_1/kernel/read*
paddingVALID*3
_output_shapes!
:џџџџџџџџџGG *
	dilations	
*
T0*
data_formatNDHWC*
strides	

П
AquamanNet/netconv_2_1/BiasAddBiasAddAquamanNet/netconv_2_1/Conv3D AquamanNet/netconv_2_1/bias/read*
T0*
data_formatNHWC*3
_output_shapes!
:џџџџџџџџџGG 

AquamanNet/netconv_2_1/EluEluAquamanNet/netconv_2_1/BiasAdd*
T0*3
_output_shapes!
:џџџџџџџџџGG 
Я
@AquamanNet/netconv_2_2/kernel/Initializer/truncated_normal/shapeConst*)
value B"                 *0
_class&
$"loc:@AquamanNet/netconv_2_2/kernel*
dtype0*
_output_shapes
:
Ж
?AquamanNet/netconv_2_2/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *0
_class&
$"loc:@AquamanNet/netconv_2_2/kernel
И
AAquamanNet/netconv_2_2/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *=*0
_class&
$"loc:@AquamanNet/netconv_2_2/kernel*
dtype0*
_output_shapes
: 
Ќ
JAquamanNet/netconv_2_2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/netconv_2_2/kernel/Initializer/truncated_normal/shape*
dtype0**
_output_shapes
:  *

seed *
T0*0
_class&
$"loc:@AquamanNet/netconv_2_2/kernel*
seed2 
Л
>AquamanNet/netconv_2_2/kernel/Initializer/truncated_normal/mulMulJAquamanNet/netconv_2_2/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/netconv_2_2/kernel/Initializer/truncated_normal/stddev**
_output_shapes
:  *
T0*0
_class&
$"loc:@AquamanNet/netconv_2_2/kernel
Љ
:AquamanNet/netconv_2_2/kernel/Initializer/truncated_normalAdd>AquamanNet/netconv_2_2/kernel/Initializer/truncated_normal/mul?AquamanNet/netconv_2_2/kernel/Initializer/truncated_normal/mean**
_output_shapes
:  *
T0*0
_class&
$"loc:@AquamanNet/netconv_2_2/kernel
л
AquamanNet/netconv_2_2/kernel
VariableV2*
shared_name *0
_class&
$"loc:@AquamanNet/netconv_2_2/kernel*
	container *
shape:  *
dtype0**
_output_shapes
:  

$AquamanNet/netconv_2_2/kernel/AssignAssignAquamanNet/netconv_2_2/kernel:AquamanNet/netconv_2_2/kernel/Initializer/truncated_normal*
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_2_2/kernel*
validate_shape(**
_output_shapes
:  
Д
"AquamanNet/netconv_2_2/kernel/readIdentityAquamanNet/netconv_2_2/kernel**
_output_shapes
:  *
T0*0
_class&
$"loc:@AquamanNet/netconv_2_2/kernel

>AquamanNet/netconv_2_2/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

?AquamanNet/netconv_2_2/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/netconv_2_2/kernel/read*
T0*
_output_shapes
: 
с
8AquamanNet/netconv_2_2/kernel/Regularizer/l2_regularizerMul>AquamanNet/netconv_2_2/kernel/Regularizer/l2_regularizer/scale?AquamanNet/netconv_2_2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
И
>AquamanNet/netconv_2_2/bias/Initializer/truncated_normal/shapeConst*
valueB: *.
_class$
" loc:@AquamanNet/netconv_2_2/bias*
dtype0*
_output_shapes
:
В
=AquamanNet/netconv_2_2/bias/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@AquamanNet/netconv_2_2/bias*
dtype0
Д
?AquamanNet/netconv_2_2/bias/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *Eё>*.
_class$
" loc:@AquamanNet/netconv_2_2/bias

HAquamanNet/netconv_2_2/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/netconv_2_2/bias/Initializer/truncated_normal/shape*

seed *
T0*.
_class$
" loc:@AquamanNet/netconv_2_2/bias*
seed2 *
dtype0*
_output_shapes
: 
Ѓ
<AquamanNet/netconv_2_2/bias/Initializer/truncated_normal/mulMulHAquamanNet/netconv_2_2/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/netconv_2_2/bias/Initializer/truncated_normal/stddev*
_output_shapes
: *
T0*.
_class$
" loc:@AquamanNet/netconv_2_2/bias

8AquamanNet/netconv_2_2/bias/Initializer/truncated_normalAdd<AquamanNet/netconv_2_2/bias/Initializer/truncated_normal/mul=AquamanNet/netconv_2_2/bias/Initializer/truncated_normal/mean*
_output_shapes
: *
T0*.
_class$
" loc:@AquamanNet/netconv_2_2/bias
З
AquamanNet/netconv_2_2/bias
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@AquamanNet/netconv_2_2/bias

"AquamanNet/netconv_2_2/bias/AssignAssignAquamanNet/netconv_2_2/bias8AquamanNet/netconv_2_2/bias/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@AquamanNet/netconv_2_2/bias*
validate_shape(*
_output_shapes
: 

 AquamanNet/netconv_2_2/bias/readIdentityAquamanNet/netconv_2_2/bias*
_output_shapes
: *
T0*.
_class$
" loc:@AquamanNet/netconv_2_2/bias

<AquamanNet/netconv_2_2/bias/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=AquamanNet/netconv_2_2/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/netconv_2_2/bias/read*
T0*
_output_shapes
: 
л
6AquamanNet/netconv_2_2/bias/Regularizer/l2_regularizerMul<AquamanNet/netconv_2_2/bias/Regularizer/l2_regularizer/scale=AquamanNet/netconv_2_2/bias/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
y
$AquamanNet/netconv_2_2/dilation_rateConst*!
valueB"         *
dtype0*
_output_shapes
:
џ
AquamanNet/netconv_2_2/Conv3DConv3DAquamanNet/netconv_2_1/Elu"AquamanNet/netconv_2_2/kernel/read*3
_output_shapes!
:џџџџџџџџџEE *
	dilations	
*
T0*
data_formatNDHWC*
strides	
*
paddingVALID
П
AquamanNet/netconv_2_2/BiasAddBiasAddAquamanNet/netconv_2_2/Conv3D AquamanNet/netconv_2_2/bias/read*3
_output_shapes!
:џџџџџџџџџEE *
T0*
data_formatNHWC

AquamanNet/netconv_2_2/EluEluAquamanNet/netconv_2_2/BiasAdd*
T0*3
_output_shapes!
:џџџџџџџџџEE 
н
!AquamanNet/netMaxPool_2/MaxPool3D	MaxPool3DAquamanNet/netconv_2_2/Elu*3
_output_shapes!
:џџџџџџџџџ## *
T0*
data_formatNDHWC*
strides	
*
ksize	
*
paddingSAME

AquamanNet/net_squeeze_0Squeeze!AquamanNet/netMaxPool_2/MaxPool3D*
T0*/
_output_shapes
:џџџџџџџџџ## *
squeeze_dims

Ы
@AquamanNet/netconv_3_1/kernel/Initializer/truncated_normal/shapeConst*%
valueB"              *0
_class&
$"loc:@AquamanNet/netconv_3_1/kernel*
dtype0*
_output_shapes
:
Ж
?AquamanNet/netconv_3_1/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *0
_class&
$"loc:@AquamanNet/netconv_3_1/kernel*
dtype0
И
AAquamanNet/netconv_3_1/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *Т=*0
_class&
$"loc:@AquamanNet/netconv_3_1/kernel*
dtype0*
_output_shapes
: 
Ј
JAquamanNet/netconv_3_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/netconv_3_1/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@AquamanNet/netconv_3_1/kernel*
seed2 *
dtype0*&
_output_shapes
:  *

seed 
З
>AquamanNet/netconv_3_1/kernel/Initializer/truncated_normal/mulMulJAquamanNet/netconv_3_1/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/netconv_3_1/kernel/Initializer/truncated_normal/stddev*0
_class&
$"loc:@AquamanNet/netconv_3_1/kernel*&
_output_shapes
:  *
T0
Ѕ
:AquamanNet/netconv_3_1/kernel/Initializer/truncated_normalAdd>AquamanNet/netconv_3_1/kernel/Initializer/truncated_normal/mul?AquamanNet/netconv_3_1/kernel/Initializer/truncated_normal/mean*&
_output_shapes
:  *
T0*0
_class&
$"loc:@AquamanNet/netconv_3_1/kernel
г
AquamanNet/netconv_3_1/kernel
VariableV2*
dtype0*&
_output_shapes
:  *
shared_name *0
_class&
$"loc:@AquamanNet/netconv_3_1/kernel*
	container *
shape:  

$AquamanNet/netconv_3_1/kernel/AssignAssignAquamanNet/netconv_3_1/kernel:AquamanNet/netconv_3_1/kernel/Initializer/truncated_normal*&
_output_shapes
:  *
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_3_1/kernel*
validate_shape(
А
"AquamanNet/netconv_3_1/kernel/readIdentityAquamanNet/netconv_3_1/kernel*0
_class&
$"loc:@AquamanNet/netconv_3_1/kernel*&
_output_shapes
:  *
T0

>AquamanNet/netconv_3_1/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

?AquamanNet/netconv_3_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/netconv_3_1/kernel/read*
T0*
_output_shapes
: 
с
8AquamanNet/netconv_3_1/kernel/Regularizer/l2_regularizerMul>AquamanNet/netconv_3_1/kernel/Regularizer/l2_regularizer/scale?AquamanNet/netconv_3_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
И
>AquamanNet/netconv_3_1/bias/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB: *.
_class$
" loc:@AquamanNet/netconv_3_1/bias*
dtype0
В
=AquamanNet/netconv_3_1/bias/Initializer/truncated_normal/meanConst*
valueB
 *    *.
_class$
" loc:@AquamanNet/netconv_3_1/bias*
dtype0*
_output_shapes
: 
Д
?AquamanNet/netconv_3_1/bias/Initializer/truncated_normal/stddevConst*
valueB
 *Eё>*.
_class$
" loc:@AquamanNet/netconv_3_1/bias*
dtype0*
_output_shapes
: 

HAquamanNet/netconv_3_1/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/netconv_3_1/bias/Initializer/truncated_normal/shape*
T0*.
_class$
" loc:@AquamanNet/netconv_3_1/bias*
seed2 *
dtype0*
_output_shapes
: *

seed 
Ѓ
<AquamanNet/netconv_3_1/bias/Initializer/truncated_normal/mulMulHAquamanNet/netconv_3_1/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/netconv_3_1/bias/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@AquamanNet/netconv_3_1/bias*
_output_shapes
: 

8AquamanNet/netconv_3_1/bias/Initializer/truncated_normalAdd<AquamanNet/netconv_3_1/bias/Initializer/truncated_normal/mul=AquamanNet/netconv_3_1/bias/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@AquamanNet/netconv_3_1/bias*
_output_shapes
: 
З
AquamanNet/netconv_3_1/bias
VariableV2*
shared_name *.
_class$
" loc:@AquamanNet/netconv_3_1/bias*
	container *
shape: *
dtype0*
_output_shapes
: 

"AquamanNet/netconv_3_1/bias/AssignAssignAquamanNet/netconv_3_1/bias8AquamanNet/netconv_3_1/bias/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@AquamanNet/netconv_3_1/bias*
validate_shape(*
_output_shapes
: 

 AquamanNet/netconv_3_1/bias/readIdentityAquamanNet/netconv_3_1/bias*.
_class$
" loc:@AquamanNet/netconv_3_1/bias*
_output_shapes
: *
T0

<AquamanNet/netconv_3_1/bias/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=AquamanNet/netconv_3_1/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/netconv_3_1/bias/read*
_output_shapes
: *
T0
л
6AquamanNet/netconv_3_1/bias/Regularizer/l2_regularizerMul<AquamanNet/netconv_3_1/bias/Regularizer/l2_regularizer/scale=AquamanNet/netconv_3_1/bias/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
u
$AquamanNet/netconv_3_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

AquamanNet/netconv_3_1/Conv2DConv2DAquamanNet/net_squeeze_0"AquamanNet/netconv_3_1/kernel/read*/
_output_shapes
:џџџџџџџџџ!! *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
Л
AquamanNet/netconv_3_1/BiasAddBiasAddAquamanNet/netconv_3_1/Conv2D AquamanNet/netconv_3_1/bias/read*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ!! *
T0
{
AquamanNet/netconv_3_1/EluEluAquamanNet/netconv_3_1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ!! 
Ы
@AquamanNet/netconv_3_2/kernel/Initializer/truncated_normal/shapeConst*%
valueB"              *0
_class&
$"loc:@AquamanNet/netconv_3_2/kernel*
dtype0*
_output_shapes
:
Ж
?AquamanNet/netconv_3_2/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@AquamanNet/netconv_3_2/kernel*
dtype0*
_output_shapes
: 
И
AAquamanNet/netconv_3_2/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *Т=*0
_class&
$"loc:@AquamanNet/netconv_3_2/kernel*
dtype0
Ј
JAquamanNet/netconv_3_2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/netconv_3_2/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@AquamanNet/netconv_3_2/kernel*
seed2 *
dtype0*&
_output_shapes
:  *

seed 
З
>AquamanNet/netconv_3_2/kernel/Initializer/truncated_normal/mulMulJAquamanNet/netconv_3_2/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/netconv_3_2/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@AquamanNet/netconv_3_2/kernel*&
_output_shapes
:  
Ѕ
:AquamanNet/netconv_3_2/kernel/Initializer/truncated_normalAdd>AquamanNet/netconv_3_2/kernel/Initializer/truncated_normal/mul?AquamanNet/netconv_3_2/kernel/Initializer/truncated_normal/mean*&
_output_shapes
:  *
T0*0
_class&
$"loc:@AquamanNet/netconv_3_2/kernel
г
AquamanNet/netconv_3_2/kernel
VariableV2*
dtype0*&
_output_shapes
:  *
shared_name *0
_class&
$"loc:@AquamanNet/netconv_3_2/kernel*
	container *
shape:  

$AquamanNet/netconv_3_2/kernel/AssignAssignAquamanNet/netconv_3_2/kernel:AquamanNet/netconv_3_2/kernel/Initializer/truncated_normal*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_3_2/kernel
А
"AquamanNet/netconv_3_2/kernel/readIdentityAquamanNet/netconv_3_2/kernel*
T0*0
_class&
$"loc:@AquamanNet/netconv_3_2/kernel*&
_output_shapes
:  

>AquamanNet/netconv_3_2/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

?AquamanNet/netconv_3_2/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/netconv_3_2/kernel/read*
T0*
_output_shapes
: 
с
8AquamanNet/netconv_3_2/kernel/Regularizer/l2_regularizerMul>AquamanNet/netconv_3_2/kernel/Regularizer/l2_regularizer/scale?AquamanNet/netconv_3_2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
И
>AquamanNet/netconv_3_2/bias/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB: *.
_class$
" loc:@AquamanNet/netconv_3_2/bias
В
=AquamanNet/netconv_3_2/bias/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@AquamanNet/netconv_3_2/bias*
dtype0
Д
?AquamanNet/netconv_3_2/bias/Initializer/truncated_normal/stddevConst*
valueB
 *Eё>*.
_class$
" loc:@AquamanNet/netconv_3_2/bias*
dtype0*
_output_shapes
: 

HAquamanNet/netconv_3_2/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/netconv_3_2/bias/Initializer/truncated_normal/shape*
seed2 *
dtype0*
_output_shapes
: *

seed *
T0*.
_class$
" loc:@AquamanNet/netconv_3_2/bias
Ѓ
<AquamanNet/netconv_3_2/bias/Initializer/truncated_normal/mulMulHAquamanNet/netconv_3_2/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/netconv_3_2/bias/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@AquamanNet/netconv_3_2/bias*
_output_shapes
: 

8AquamanNet/netconv_3_2/bias/Initializer/truncated_normalAdd<AquamanNet/netconv_3_2/bias/Initializer/truncated_normal/mul=AquamanNet/netconv_3_2/bias/Initializer/truncated_normal/mean*.
_class$
" loc:@AquamanNet/netconv_3_2/bias*
_output_shapes
: *
T0
З
AquamanNet/netconv_3_2/bias
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@AquamanNet/netconv_3_2/bias

"AquamanNet/netconv_3_2/bias/AssignAssignAquamanNet/netconv_3_2/bias8AquamanNet/netconv_3_2/bias/Initializer/truncated_normal*.
_class$
" loc:@AquamanNet/netconv_3_2/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

 AquamanNet/netconv_3_2/bias/readIdentityAquamanNet/netconv_3_2/bias*
T0*.
_class$
" loc:@AquamanNet/netconv_3_2/bias*
_output_shapes
: 

<AquamanNet/netconv_3_2/bias/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o:

=AquamanNet/netconv_3_2/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/netconv_3_2/bias/read*
_output_shapes
: *
T0
л
6AquamanNet/netconv_3_2/bias/Regularizer/l2_regularizerMul<AquamanNet/netconv_3_2/bias/Regularizer/l2_regularizer/scale=AquamanNet/netconv_3_2/bias/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
u
$AquamanNet/netconv_3_2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

AquamanNet/netconv_3_2/Conv2DConv2DAquamanNet/netconv_3_1/Elu"AquamanNet/netconv_3_2/kernel/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ *
	dilations
*
T0
Л
AquamanNet/netconv_3_2/BiasAddBiasAddAquamanNet/netconv_3_2/Conv2D AquamanNet/netconv_3_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ 
{
AquamanNet/netconv_3_2/EluEluAquamanNet/netconv_3_2/BiasAdd*/
_output_shapes
:џџџџџџџџџ *
T0
в
AquamanNet/netMaxPool_3/MaxPoolMaxPoolAquamanNet/netconv_3_2/Elu*/
_output_shapes
:џџџџџџџџџ *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
Ы
@AquamanNet/netconv_4_1/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*%
valueB"              *0
_class&
$"loc:@AquamanNet/netconv_4_1/kernel*
dtype0
Ж
?AquamanNet/netconv_4_1/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@AquamanNet/netconv_4_1/kernel*
dtype0*
_output_shapes
: 
И
AAquamanNet/netconv_4_1/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *Т=*0
_class&
$"loc:@AquamanNet/netconv_4_1/kernel*
dtype0*
_output_shapes
: 
Ј
JAquamanNet/netconv_4_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/netconv_4_1/kernel/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:  *

seed *
T0*0
_class&
$"loc:@AquamanNet/netconv_4_1/kernel*
seed2 
З
>AquamanNet/netconv_4_1/kernel/Initializer/truncated_normal/mulMulJAquamanNet/netconv_4_1/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/netconv_4_1/kernel/Initializer/truncated_normal/stddev*&
_output_shapes
:  *
T0*0
_class&
$"loc:@AquamanNet/netconv_4_1/kernel
Ѕ
:AquamanNet/netconv_4_1/kernel/Initializer/truncated_normalAdd>AquamanNet/netconv_4_1/kernel/Initializer/truncated_normal/mul?AquamanNet/netconv_4_1/kernel/Initializer/truncated_normal/mean*0
_class&
$"loc:@AquamanNet/netconv_4_1/kernel*&
_output_shapes
:  *
T0
г
AquamanNet/netconv_4_1/kernel
VariableV2*0
_class&
$"loc:@AquamanNet/netconv_4_1/kernel*
	container *
shape:  *
dtype0*&
_output_shapes
:  *
shared_name 

$AquamanNet/netconv_4_1/kernel/AssignAssignAquamanNet/netconv_4_1/kernel:AquamanNet/netconv_4_1/kernel/Initializer/truncated_normal*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_4_1/kernel
А
"AquamanNet/netconv_4_1/kernel/readIdentityAquamanNet/netconv_4_1/kernel*
T0*0
_class&
$"loc:@AquamanNet/netconv_4_1/kernel*&
_output_shapes
:  

>AquamanNet/netconv_4_1/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

?AquamanNet/netconv_4_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/netconv_4_1/kernel/read*
_output_shapes
: *
T0
с
8AquamanNet/netconv_4_1/kernel/Regularizer/l2_regularizerMul>AquamanNet/netconv_4_1/kernel/Regularizer/l2_regularizer/scale?AquamanNet/netconv_4_1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
И
>AquamanNet/netconv_4_1/bias/Initializer/truncated_normal/shapeConst*
valueB: *.
_class$
" loc:@AquamanNet/netconv_4_1/bias*
dtype0*
_output_shapes
:
В
=AquamanNet/netconv_4_1/bias/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *.
_class$
" loc:@AquamanNet/netconv_4_1/bias*
dtype0
Д
?AquamanNet/netconv_4_1/bias/Initializer/truncated_normal/stddevConst*
valueB
 *Eё>*.
_class$
" loc:@AquamanNet/netconv_4_1/bias*
dtype0*
_output_shapes
: 

HAquamanNet/netconv_4_1/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/netconv_4_1/bias/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
: *

seed *
T0*.
_class$
" loc:@AquamanNet/netconv_4_1/bias*
seed2 
Ѓ
<AquamanNet/netconv_4_1/bias/Initializer/truncated_normal/mulMulHAquamanNet/netconv_4_1/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/netconv_4_1/bias/Initializer/truncated_normal/stddev*.
_class$
" loc:@AquamanNet/netconv_4_1/bias*
_output_shapes
: *
T0

8AquamanNet/netconv_4_1/bias/Initializer/truncated_normalAdd<AquamanNet/netconv_4_1/bias/Initializer/truncated_normal/mul=AquamanNet/netconv_4_1/bias/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@AquamanNet/netconv_4_1/bias*
_output_shapes
: 
З
AquamanNet/netconv_4_1/bias
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@AquamanNet/netconv_4_1/bias*
	container 

"AquamanNet/netconv_4_1/bias/AssignAssignAquamanNet/netconv_4_1/bias8AquamanNet/netconv_4_1/bias/Initializer/truncated_normal*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@AquamanNet/netconv_4_1/bias*
validate_shape(

 AquamanNet/netconv_4_1/bias/readIdentityAquamanNet/netconv_4_1/bias*
T0*.
_class$
" loc:@AquamanNet/netconv_4_1/bias*
_output_shapes
: 

<AquamanNet/netconv_4_1/bias/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=AquamanNet/netconv_4_1/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/netconv_4_1/bias/read*
T0*
_output_shapes
: 
л
6AquamanNet/netconv_4_1/bias/Regularizer/l2_regularizerMul<AquamanNet/netconv_4_1/bias/Regularizer/l2_regularizer/scale=AquamanNet/netconv_4_1/bias/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
u
$AquamanNet/netconv_4_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

AquamanNet/netconv_4_1/Conv2DConv2DAquamanNet/netMaxPool_3/MaxPool"AquamanNet/netconv_4_1/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ 
Л
AquamanNet/netconv_4_1/BiasAddBiasAddAquamanNet/netconv_4_1/Conv2D AquamanNet/netconv_4_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ 
{
AquamanNet/netconv_4_1/EluEluAquamanNet/netconv_4_1/BiasAdd*
T0*/
_output_shapes
:џџџџџџџџџ 
Ы
@AquamanNet/netconv_4_2/kernel/Initializer/truncated_normal/shapeConst*%
valueB"              *0
_class&
$"loc:@AquamanNet/netconv_4_2/kernel*
dtype0*
_output_shapes
:
Ж
?AquamanNet/netconv_4_2/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@AquamanNet/netconv_4_2/kernel*
dtype0*
_output_shapes
: 
И
AAquamanNet/netconv_4_2/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *Т=*0
_class&
$"loc:@AquamanNet/netconv_4_2/kernel*
dtype0*
_output_shapes
: 
Ј
JAquamanNet/netconv_4_2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/netconv_4_2/kernel/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:  *

seed *
T0*0
_class&
$"loc:@AquamanNet/netconv_4_2/kernel*
seed2 
З
>AquamanNet/netconv_4_2/kernel/Initializer/truncated_normal/mulMulJAquamanNet/netconv_4_2/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/netconv_4_2/kernel/Initializer/truncated_normal/stddev*0
_class&
$"loc:@AquamanNet/netconv_4_2/kernel*&
_output_shapes
:  *
T0
Ѕ
:AquamanNet/netconv_4_2/kernel/Initializer/truncated_normalAdd>AquamanNet/netconv_4_2/kernel/Initializer/truncated_normal/mul?AquamanNet/netconv_4_2/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@AquamanNet/netconv_4_2/kernel*&
_output_shapes
:  
г
AquamanNet/netconv_4_2/kernel
VariableV2*
	container *
shape:  *
dtype0*&
_output_shapes
:  *
shared_name *0
_class&
$"loc:@AquamanNet/netconv_4_2/kernel

$AquamanNet/netconv_4_2/kernel/AssignAssignAquamanNet/netconv_4_2/kernel:AquamanNet/netconv_4_2/kernel/Initializer/truncated_normal*
T0*0
_class&
$"loc:@AquamanNet/netconv_4_2/kernel*
validate_shape(*&
_output_shapes
:  *
use_locking(
А
"AquamanNet/netconv_4_2/kernel/readIdentityAquamanNet/netconv_4_2/kernel*
T0*0
_class&
$"loc:@AquamanNet/netconv_4_2/kernel*&
_output_shapes
:  

>AquamanNet/netconv_4_2/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

?AquamanNet/netconv_4_2/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/netconv_4_2/kernel/read*
T0*
_output_shapes
: 
с
8AquamanNet/netconv_4_2/kernel/Regularizer/l2_regularizerMul>AquamanNet/netconv_4_2/kernel/Regularizer/l2_regularizer/scale?AquamanNet/netconv_4_2/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
И
>AquamanNet/netconv_4_2/bias/Initializer/truncated_normal/shapeConst*
valueB: *.
_class$
" loc:@AquamanNet/netconv_4_2/bias*
dtype0*
_output_shapes
:
В
=AquamanNet/netconv_4_2/bias/Initializer/truncated_normal/meanConst*
valueB
 *    *.
_class$
" loc:@AquamanNet/netconv_4_2/bias*
dtype0*
_output_shapes
: 
Д
?AquamanNet/netconv_4_2/bias/Initializer/truncated_normal/stddevConst*
valueB
 *Eё>*.
_class$
" loc:@AquamanNet/netconv_4_2/bias*
dtype0*
_output_shapes
: 

HAquamanNet/netconv_4_2/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/netconv_4_2/bias/Initializer/truncated_normal/shape*
seed2 *
dtype0*
_output_shapes
: *

seed *
T0*.
_class$
" loc:@AquamanNet/netconv_4_2/bias
Ѓ
<AquamanNet/netconv_4_2/bias/Initializer/truncated_normal/mulMulHAquamanNet/netconv_4_2/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/netconv_4_2/bias/Initializer/truncated_normal/stddev*
_output_shapes
: *
T0*.
_class$
" loc:@AquamanNet/netconv_4_2/bias

8AquamanNet/netconv_4_2/bias/Initializer/truncated_normalAdd<AquamanNet/netconv_4_2/bias/Initializer/truncated_normal/mul=AquamanNet/netconv_4_2/bias/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@AquamanNet/netconv_4_2/bias*
_output_shapes
: 
З
AquamanNet/netconv_4_2/bias
VariableV2*
shared_name *.
_class$
" loc:@AquamanNet/netconv_4_2/bias*
	container *
shape: *
dtype0*
_output_shapes
: 

"AquamanNet/netconv_4_2/bias/AssignAssignAquamanNet/netconv_4_2/bias8AquamanNet/netconv_4_2/bias/Initializer/truncated_normal*.
_class$
" loc:@AquamanNet/netconv_4_2/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

 AquamanNet/netconv_4_2/bias/readIdentityAquamanNet/netconv_4_2/bias*
_output_shapes
: *
T0*.
_class$
" loc:@AquamanNet/netconv_4_2/bias

<AquamanNet/netconv_4_2/bias/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=AquamanNet/netconv_4_2/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/netconv_4_2/bias/read*
T0*
_output_shapes
: 
л
6AquamanNet/netconv_4_2/bias/Regularizer/l2_regularizerMul<AquamanNet/netconv_4_2/bias/Regularizer/l2_regularizer/scale=AquamanNet/netconv_4_2/bias/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
u
$AquamanNet/netconv_4_2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

AquamanNet/netconv_4_2/Conv2DConv2DAquamanNet/netconv_4_1/Elu"AquamanNet/netconv_4_2/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:џџџџџџџџџ *
	dilations

Л
AquamanNet/netconv_4_2/BiasAddBiasAddAquamanNet/netconv_4_2/Conv2D AquamanNet/netconv_4_2/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ 
{
AquamanNet/netconv_4_2/EluEluAquamanNet/netconv_4_2/BiasAdd*/
_output_shapes
:џџџџџџџџџ *
T0
в
AquamanNet/netMaxPool_4/MaxPoolMaxPoolAquamanNet/netconv_4_2/Elu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ 
w
AquamanNet/flatten/ShapeShapeAquamanNet/netMaxPool_4/MaxPool*
_output_shapes
:*
T0*
out_type0
p
&AquamanNet/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
r
(AquamanNet/flatten/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
r
(AquamanNet/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
и
 AquamanNet/flatten/strided_sliceStridedSliceAquamanNet/flatten/Shape&AquamanNet/flatten/strided_slice/stack(AquamanNet/flatten/strided_slice/stack_1(AquamanNet/flatten/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
m
"AquamanNet/flatten/Reshape/shape/1Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ј
 AquamanNet/flatten/Reshape/shapePack AquamanNet/flatten/strided_slice"AquamanNet/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Љ
AquamanNet/flatten/ReshapeReshapeAquamanNet/netMaxPool_4/MaxPool AquamanNet/flatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ	
У
@AquamanNet/net_dense_1/kernel/Initializer/truncated_normal/shapeConst*
valueB"     *0
_class&
$"loc:@AquamanNet/net_dense_1/kernel*
dtype0*
_output_shapes
:
Ж
?AquamanNet/net_dense_1/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@AquamanNet/net_dense_1/kernel*
dtype0*
_output_shapes
: 
И
AAquamanNet/net_dense_1/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *B=*0
_class&
$"loc:@AquamanNet/net_dense_1/kernel*
dtype0
Ђ
JAquamanNet/net_dense_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/net_dense_1/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@AquamanNet/net_dense_1/kernel*
seed2 *
dtype0* 
_output_shapes
:
	*

seed 
Б
>AquamanNet/net_dense_1/kernel/Initializer/truncated_normal/mulMulJAquamanNet/net_dense_1/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/net_dense_1/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@AquamanNet/net_dense_1/kernel* 
_output_shapes
:
	

:AquamanNet/net_dense_1/kernel/Initializer/truncated_normalAdd>AquamanNet/net_dense_1/kernel/Initializer/truncated_normal/mul?AquamanNet/net_dense_1/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@AquamanNet/net_dense_1/kernel* 
_output_shapes
:
	
Ч
AquamanNet/net_dense_1/kernel
VariableV2*
dtype0* 
_output_shapes
:
	*
shared_name *0
_class&
$"loc:@AquamanNet/net_dense_1/kernel*
	container *
shape:
	

$AquamanNet/net_dense_1/kernel/AssignAssignAquamanNet/net_dense_1/kernel:AquamanNet/net_dense_1/kernel/Initializer/truncated_normal* 
_output_shapes
:
	*
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/net_dense_1/kernel*
validate_shape(
Њ
"AquamanNet/net_dense_1/kernel/readIdentityAquamanNet/net_dense_1/kernel*0
_class&
$"loc:@AquamanNet/net_dense_1/kernel* 
_output_shapes
:
	*
T0

>AquamanNet/net_dense_1/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

?AquamanNet/net_dense_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/net_dense_1/kernel/read*
_output_shapes
: *
T0
с
8AquamanNet/net_dense_1/kernel/Regularizer/l2_regularizerMul>AquamanNet/net_dense_1/kernel/Regularizer/l2_regularizer/scale?AquamanNet/net_dense_1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
Й
>AquamanNet/net_dense_1/bias/Initializer/truncated_normal/shapeConst*
valueB:*.
_class$
" loc:@AquamanNet/net_dense_1/bias*
dtype0*
_output_shapes
:
В
=AquamanNet/net_dense_1/bias/Initializer/truncated_normal/meanConst*
valueB
 *    *.
_class$
" loc:@AquamanNet/net_dense_1/bias*
dtype0*
_output_shapes
: 
Д
?AquamanNet/net_dense_1/bias/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *аdЮ=*.
_class$
" loc:@AquamanNet/net_dense_1/bias*
dtype0

HAquamanNet/net_dense_1/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/net_dense_1/bias/Initializer/truncated_normal/shape*

seed *
T0*.
_class$
" loc:@AquamanNet/net_dense_1/bias*
seed2 *
dtype0*
_output_shapes	
:
Є
<AquamanNet/net_dense_1/bias/Initializer/truncated_normal/mulMulHAquamanNet/net_dense_1/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/net_dense_1/bias/Initializer/truncated_normal/stddev*
_output_shapes	
:*
T0*.
_class$
" loc:@AquamanNet/net_dense_1/bias

8AquamanNet/net_dense_1/bias/Initializer/truncated_normalAdd<AquamanNet/net_dense_1/bias/Initializer/truncated_normal/mul=AquamanNet/net_dense_1/bias/Initializer/truncated_normal/mean*
_output_shapes	
:*
T0*.
_class$
" loc:@AquamanNet/net_dense_1/bias
Й
AquamanNet/net_dense_1/bias
VariableV2*
shared_name *.
_class$
" loc:@AquamanNet/net_dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:

"AquamanNet/net_dense_1/bias/AssignAssignAquamanNet/net_dense_1/bias8AquamanNet/net_dense_1/bias/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@AquamanNet/net_dense_1/bias*
validate_shape(*
_output_shapes	
:

 AquamanNet/net_dense_1/bias/readIdentityAquamanNet/net_dense_1/bias*
T0*.
_class$
" loc:@AquamanNet/net_dense_1/bias*
_output_shapes	
:

<AquamanNet/net_dense_1/bias/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=AquamanNet/net_dense_1/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/net_dense_1/bias/read*
T0*
_output_shapes
: 
л
6AquamanNet/net_dense_1/bias/Regularizer/l2_regularizerMul<AquamanNet/net_dense_1/bias/Regularizer/l2_regularizer/scale=AquamanNet/net_dense_1/bias/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
Р
AquamanNet/net_dense_1/MatMulMatMulAquamanNet/flatten/Reshape"AquamanNet/net_dense_1/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Д
AquamanNet/net_dense_1/BiasAddBiasAddAquamanNet/net_dense_1/MatMul AquamanNet/net_dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
t
AquamanNet/net_dense_1/EluEluAquamanNet/net_dense_1/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
|
!AquamanNet/net_dropout_1/IdentityIdentityAquamanNet/net_dense_1/Elu*
T0*(
_output_shapes
:џџџџџџџџџ
У
@AquamanNet/net_dense_2/kernel/Initializer/truncated_normal/shapeConst*
valueB"   @   *0
_class&
$"loc:@AquamanNet/net_dense_2/kernel*
dtype0*
_output_shapes
:
Ж
?AquamanNet/net_dense_2/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@AquamanNet/net_dense_2/kernel*
dtype0*
_output_shapes
: 
И
AAquamanNet/net_dense_2/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *аdЮ=*0
_class&
$"loc:@AquamanNet/net_dense_2/kernel*
dtype0*
_output_shapes
: 
Ё
JAquamanNet/net_dense_2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/net_dense_2/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@AquamanNet/net_dense_2/kernel*
seed2 *
dtype0*
_output_shapes
:	@*

seed 
А
>AquamanNet/net_dense_2/kernel/Initializer/truncated_normal/mulMulJAquamanNet/net_dense_2/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/net_dense_2/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@AquamanNet/net_dense_2/kernel*
_output_shapes
:	@

:AquamanNet/net_dense_2/kernel/Initializer/truncated_normalAdd>AquamanNet/net_dense_2/kernel/Initializer/truncated_normal/mul?AquamanNet/net_dense_2/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@AquamanNet/net_dense_2/kernel*
_output_shapes
:	@
Х
AquamanNet/net_dense_2/kernel
VariableV2*0
_class&
$"loc:@AquamanNet/net_dense_2/kernel*
	container *
shape:	@*
dtype0*
_output_shapes
:	@*
shared_name 

$AquamanNet/net_dense_2/kernel/AssignAssignAquamanNet/net_dense_2/kernel:AquamanNet/net_dense_2/kernel/Initializer/truncated_normal*
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/net_dense_2/kernel*
validate_shape(*
_output_shapes
:	@
Љ
"AquamanNet/net_dense_2/kernel/readIdentityAquamanNet/net_dense_2/kernel*
T0*0
_class&
$"loc:@AquamanNet/net_dense_2/kernel*
_output_shapes
:	@

>AquamanNet/net_dense_2/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
valueB
 *o:*
dtype0

?AquamanNet/net_dense_2/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/net_dense_2/kernel/read*
T0*
_output_shapes
: 
с
8AquamanNet/net_dense_2/kernel/Regularizer/l2_regularizerMul>AquamanNet/net_dense_2/kernel/Regularizer/l2_regularizer/scale?AquamanNet/net_dense_2/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
И
>AquamanNet/net_dense_2/bias/Initializer/truncated_normal/shapeConst*
valueB:@*.
_class$
" loc:@AquamanNet/net_dense_2/bias*
dtype0*
_output_shapes
:
В
=AquamanNet/net_dense_2/bias/Initializer/truncated_normal/meanConst*
valueB
 *    *.
_class$
" loc:@AquamanNet/net_dense_2/bias*
dtype0*
_output_shapes
: 
Д
?AquamanNet/net_dense_2/bias/Initializer/truncated_normal/stddevConst*
valueB
 *аdN>*.
_class$
" loc:@AquamanNet/net_dense_2/bias*
dtype0*
_output_shapes
: 

HAquamanNet/net_dense_2/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/net_dense_2/bias/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
:@*

seed *
T0*.
_class$
" loc:@AquamanNet/net_dense_2/bias*
seed2 
Ѓ
<AquamanNet/net_dense_2/bias/Initializer/truncated_normal/mulMulHAquamanNet/net_dense_2/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/net_dense_2/bias/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@AquamanNet/net_dense_2/bias*
_output_shapes
:@

8AquamanNet/net_dense_2/bias/Initializer/truncated_normalAdd<AquamanNet/net_dense_2/bias/Initializer/truncated_normal/mul=AquamanNet/net_dense_2/bias/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@AquamanNet/net_dense_2/bias*
_output_shapes
:@
З
AquamanNet/net_dense_2/bias
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *.
_class$
" loc:@AquamanNet/net_dense_2/bias*
	container *
shape:@

"AquamanNet/net_dense_2/bias/AssignAssignAquamanNet/net_dense_2/bias8AquamanNet/net_dense_2/bias/Initializer/truncated_normal*.
_class$
" loc:@AquamanNet/net_dense_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0

 AquamanNet/net_dense_2/bias/readIdentityAquamanNet/net_dense_2/bias*
T0*.
_class$
" loc:@AquamanNet/net_dense_2/bias*
_output_shapes
:@

<AquamanNet/net_dense_2/bias/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=AquamanNet/net_dense_2/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/net_dense_2/bias/read*
T0*
_output_shapes
: 
л
6AquamanNet/net_dense_2/bias/Regularizer/l2_regularizerMul<AquamanNet/net_dense_2/bias/Regularizer/l2_regularizer/scale=AquamanNet/net_dense_2/bias/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
Ц
AquamanNet/net_dense_2/MatMulMatMul!AquamanNet/net_dropout_1/Identity"AquamanNet/net_dense_2/kernel/read*'
_output_shapes
:џџџџџџџџџ@*
transpose_a( *
transpose_b( *
T0
Г
AquamanNet/net_dense_2/BiasAddBiasAddAquamanNet/net_dense_2/MatMul AquamanNet/net_dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ@
s
AquamanNet/net_dense_2/EluEluAquamanNet/net_dense_2/BiasAdd*'
_output_shapes
:џџџџџџџџџ@*
T0
{
!AquamanNet/net_dropout_2/IdentityIdentityAquamanNet/net_dense_2/Elu*'
_output_shapes
:џџџџџџџџџ@*
T0
У
@AquamanNet/net_dense_3/kernel/Initializer/truncated_normal/shapeConst*
valueB"@      *0
_class&
$"loc:@AquamanNet/net_dense_3/kernel*
dtype0*
_output_shapes
:
Ж
?AquamanNet/net_dense_3/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@AquamanNet/net_dense_3/kernel*
dtype0*
_output_shapes
: 
И
AAquamanNet/net_dense_3/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *аdN>*0
_class&
$"loc:@AquamanNet/net_dense_3/kernel*
dtype0*
_output_shapes
: 
 
JAquamanNet/net_dense_3/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@AquamanNet/net_dense_3/kernel/Initializer/truncated_normal/shape*
_output_shapes

:@*

seed *
T0*0
_class&
$"loc:@AquamanNet/net_dense_3/kernel*
seed2 *
dtype0
Џ
>AquamanNet/net_dense_3/kernel/Initializer/truncated_normal/mulMulJAquamanNet/net_dense_3/kernel/Initializer/truncated_normal/TruncatedNormalAAquamanNet/net_dense_3/kernel/Initializer/truncated_normal/stddev*0
_class&
$"loc:@AquamanNet/net_dense_3/kernel*
_output_shapes

:@*
T0

:AquamanNet/net_dense_3/kernel/Initializer/truncated_normalAdd>AquamanNet/net_dense_3/kernel/Initializer/truncated_normal/mul?AquamanNet/net_dense_3/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@AquamanNet/net_dense_3/kernel*
_output_shapes

:@
У
AquamanNet/net_dense_3/kernel
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *0
_class&
$"loc:@AquamanNet/net_dense_3/kernel*
	container 

$AquamanNet/net_dense_3/kernel/AssignAssignAquamanNet/net_dense_3/kernel:AquamanNet/net_dense_3/kernel/Initializer/truncated_normal*
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/net_dense_3/kernel*
validate_shape(*
_output_shapes

:@
Ј
"AquamanNet/net_dense_3/kernel/readIdentityAquamanNet/net_dense_3/kernel*
T0*0
_class&
$"loc:@AquamanNet/net_dense_3/kernel*
_output_shapes

:@

>AquamanNet/net_dense_3/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

?AquamanNet/net_dense_3/kernel/Regularizer/l2_regularizer/L2LossL2Loss"AquamanNet/net_dense_3/kernel/read*
T0*
_output_shapes
: 
с
8AquamanNet/net_dense_3/kernel/Regularizer/l2_regularizerMul>AquamanNet/net_dense_3/kernel/Regularizer/l2_regularizer/scale?AquamanNet/net_dense_3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
И
>AquamanNet/net_dense_3/bias/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB:*.
_class$
" loc:@AquamanNet/net_dense_3/bias*
dtype0
В
=AquamanNet/net_dense_3/bias/Initializer/truncated_normal/meanConst*
valueB
 *    *.
_class$
" loc:@AquamanNet/net_dense_3/bias*
dtype0*
_output_shapes
: 
Д
?AquamanNet/net_dense_3/bias/Initializer/truncated_normal/stddevConst*
valueB
 *Eё?*.
_class$
" loc:@AquamanNet/net_dense_3/bias*
dtype0*
_output_shapes
: 

HAquamanNet/net_dense_3/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>AquamanNet/net_dense_3/bias/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
:*

seed *
T0*.
_class$
" loc:@AquamanNet/net_dense_3/bias*
seed2 
Ѓ
<AquamanNet/net_dense_3/bias/Initializer/truncated_normal/mulMulHAquamanNet/net_dense_3/bias/Initializer/truncated_normal/TruncatedNormal?AquamanNet/net_dense_3/bias/Initializer/truncated_normal/stddev*.
_class$
" loc:@AquamanNet/net_dense_3/bias*
_output_shapes
:*
T0

8AquamanNet/net_dense_3/bias/Initializer/truncated_normalAdd<AquamanNet/net_dense_3/bias/Initializer/truncated_normal/mul=AquamanNet/net_dense_3/bias/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@AquamanNet/net_dense_3/bias*
_output_shapes
:
З
AquamanNet/net_dense_3/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@AquamanNet/net_dense_3/bias*
	container *
shape:

"AquamanNet/net_dense_3/bias/AssignAssignAquamanNet/net_dense_3/bias8AquamanNet/net_dense_3/bias/Initializer/truncated_normal*.
_class$
" loc:@AquamanNet/net_dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

 AquamanNet/net_dense_3/bias/readIdentityAquamanNet/net_dense_3/bias*
_output_shapes
:*
T0*.
_class$
" loc:@AquamanNet/net_dense_3/bias

<AquamanNet/net_dense_3/bias/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 

=AquamanNet/net_dense_3/bias/Regularizer/l2_regularizer/L2LossL2Loss AquamanNet/net_dense_3/bias/read*
T0*
_output_shapes
: 
л
6AquamanNet/net_dense_3/bias/Regularizer/l2_regularizerMul<AquamanNet/net_dense_3/bias/Regularizer/l2_regularizer/scale=AquamanNet/net_dense_3/bias/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
Ц
AquamanNet/net_dense_3/MatMulMatMul!AquamanNet/net_dropout_2/Identity"AquamanNet/net_dense_3/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Г
AquamanNet/net_dense_3/BiasAddBiasAddAquamanNet/net_dense_3/MatMul AquamanNet/net_dense_3/bias/read*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ*
T0
s
AquamanNet/net_dense_3/EluEluAquamanNet/net_dense_3/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
W
predictions/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

predictionsArgMaxAquamanNet/net_dense_3/Elupredictions/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_331ee1264d4e4872892fbfe2e292da67/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 

save/SaveV2/tensor_namesConst"/device:CPU:0*Ў
valueЄBЁBAquamanNet/net_dense_1/biasBAquamanNet/net_dense_1/kernelBAquamanNet/net_dense_2/biasBAquamanNet/net_dense_2/kernelBAquamanNet/net_dense_3/biasBAquamanNet/net_dense_3/kernelBAquamanNet/netconv_0_0/biasBAquamanNet/netconv_0_0/kernelBAquamanNet/netconv_0_1/biasBAquamanNet/netconv_0_1/kernelBAquamanNet/netconv_1_1/biasBAquamanNet/netconv_1_1/kernelBAquamanNet/netconv_1_2/biasBAquamanNet/netconv_1_2/kernelBAquamanNet/netconv_2_1/biasBAquamanNet/netconv_2_1/kernelBAquamanNet/netconv_2_2/biasBAquamanNet/netconv_2_2/kernelBAquamanNet/netconv_3_1/biasBAquamanNet/netconv_3_1/kernelBAquamanNet/netconv_3_2/biasBAquamanNet/netconv_3_2/kernelBAquamanNet/netconv_4_1/biasBAquamanNet/netconv_4_1/kernelBAquamanNet/netconv_4_2/biasBAquamanNet/netconv_4_2/kernelBglobal_step*
dtype0*
_output_shapes
:
Ј
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ж
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesAquamanNet/net_dense_1/biasAquamanNet/net_dense_1/kernelAquamanNet/net_dense_2/biasAquamanNet/net_dense_2/kernelAquamanNet/net_dense_3/biasAquamanNet/net_dense_3/kernelAquamanNet/netconv_0_0/biasAquamanNet/netconv_0_0/kernelAquamanNet/netconv_0_1/biasAquamanNet/netconv_0_1/kernelAquamanNet/netconv_1_1/biasAquamanNet/netconv_1_1/kernelAquamanNet/netconv_1_2/biasAquamanNet/netconv_1_2/kernelAquamanNet/netconv_2_1/biasAquamanNet/netconv_2_1/kernelAquamanNet/netconv_2_2/biasAquamanNet/netconv_2_2/kernelAquamanNet/netconv_3_1/biasAquamanNet/netconv_3_1/kernelAquamanNet/netconv_3_2/biasAquamanNet/netconv_3_2/kernelAquamanNet/netconv_4_1/biasAquamanNet/netconv_4_1/kernelAquamanNet/netconv_4_2/biasAquamanNet/netconv_4_2/kernelglobal_step"/device:CPU:0*)
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ќ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0

save/RestoreV2/tensor_namesConst"/device:CPU:0*Ў
valueЄBЁBAquamanNet/net_dense_1/biasBAquamanNet/net_dense_1/kernelBAquamanNet/net_dense_2/biasBAquamanNet/net_dense_2/kernelBAquamanNet/net_dense_3/biasBAquamanNet/net_dense_3/kernelBAquamanNet/netconv_0_0/biasBAquamanNet/netconv_0_0/kernelBAquamanNet/netconv_0_1/biasBAquamanNet/netconv_0_1/kernelBAquamanNet/netconv_1_1/biasBAquamanNet/netconv_1_1/kernelBAquamanNet/netconv_1_2/biasBAquamanNet/netconv_1_2/kernelBAquamanNet/netconv_2_1/biasBAquamanNet/netconv_2_1/kernelBAquamanNet/netconv_2_2/biasBAquamanNet/netconv_2_2/kernelBAquamanNet/netconv_3_1/biasBAquamanNet/netconv_3_1/kernelBAquamanNet/netconv_3_2/biasBAquamanNet/netconv_3_2/kernelBAquamanNet/netconv_4_1/biasBAquamanNet/netconv_4_1/kernelBAquamanNet/netconv_4_2/biasBAquamanNet/netconv_4_2/kernelBglobal_step*
dtype0*
_output_shapes
:
Ћ
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ђ
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	
С
save/AssignAssignAquamanNet/net_dense_1/biassave/RestoreV2*.
_class$
" loc:@AquamanNet/net_dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ю
save/Assign_1AssignAquamanNet/net_dense_1/kernelsave/RestoreV2:1* 
_output_shapes
:
	*
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/net_dense_1/kernel*
validate_shape(
Ф
save/Assign_2AssignAquamanNet/net_dense_2/biassave/RestoreV2:2*
T0*.
_class$
" loc:@AquamanNet/net_dense_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
Э
save/Assign_3AssignAquamanNet/net_dense_2/kernelsave/RestoreV2:3*
_output_shapes
:	@*
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/net_dense_2/kernel*
validate_shape(
Ф
save/Assign_4AssignAquamanNet/net_dense_3/biassave/RestoreV2:4*
use_locking(*
T0*.
_class$
" loc:@AquamanNet/net_dense_3/bias*
validate_shape(*
_output_shapes
:
Ь
save/Assign_5AssignAquamanNet/net_dense_3/kernelsave/RestoreV2:5*
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/net_dense_3/kernel*
validate_shape(*
_output_shapes

:@
Ф
save/Assign_6AssignAquamanNet/netconv_0_0/biassave/RestoreV2:6*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@AquamanNet/netconv_0_0/bias*
validate_shape(
и
save/Assign_7AssignAquamanNet/netconv_0_0/kernelsave/RestoreV2:7*
T0*0
_class&
$"loc:@AquamanNet/netconv_0_0/kernel*
validate_shape(**
_output_shapes
: *
use_locking(
Ф
save/Assign_8AssignAquamanNet/netconv_0_1/biassave/RestoreV2:8*
T0*.
_class$
" loc:@AquamanNet/netconv_0_1/bias*
validate_shape(*
_output_shapes
: *
use_locking(
и
save/Assign_9AssignAquamanNet/netconv_0_1/kernelsave/RestoreV2:9*
validate_shape(**
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_0_1/kernel
Ц
save/Assign_10AssignAquamanNet/netconv_1_1/biassave/RestoreV2:10*
use_locking(*
T0*.
_class$
" loc:@AquamanNet/netconv_1_1/bias*
validate_shape(*
_output_shapes
: 
к
save/Assign_11AssignAquamanNet/netconv_1_1/kernelsave/RestoreV2:11*0
_class&
$"loc:@AquamanNet/netconv_1_1/kernel*
validate_shape(**
_output_shapes
:  *
use_locking(*
T0
Ц
save/Assign_12AssignAquamanNet/netconv_1_2/biassave/RestoreV2:12*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@AquamanNet/netconv_1_2/bias*
validate_shape(
к
save/Assign_13AssignAquamanNet/netconv_1_2/kernelsave/RestoreV2:13*
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_1_2/kernel*
validate_shape(**
_output_shapes
:  
Ц
save/Assign_14AssignAquamanNet/netconv_2_1/biassave/RestoreV2:14*.
_class$
" loc:@AquamanNet/netconv_2_1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
к
save/Assign_15AssignAquamanNet/netconv_2_1/kernelsave/RestoreV2:15*0
_class&
$"loc:@AquamanNet/netconv_2_1/kernel*
validate_shape(**
_output_shapes
:  *
use_locking(*
T0
Ц
save/Assign_16AssignAquamanNet/netconv_2_2/biassave/RestoreV2:16*
use_locking(*
T0*.
_class$
" loc:@AquamanNet/netconv_2_2/bias*
validate_shape(*
_output_shapes
: 
к
save/Assign_17AssignAquamanNet/netconv_2_2/kernelsave/RestoreV2:17**
_output_shapes
:  *
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_2_2/kernel*
validate_shape(
Ц
save/Assign_18AssignAquamanNet/netconv_3_1/biassave/RestoreV2:18*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@AquamanNet/netconv_3_1/bias*
validate_shape(
ж
save/Assign_19AssignAquamanNet/netconv_3_1/kernelsave/RestoreV2:19*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_3_1/kernel
Ц
save/Assign_20AssignAquamanNet/netconv_3_2/biassave/RestoreV2:20*
use_locking(*
T0*.
_class$
" loc:@AquamanNet/netconv_3_2/bias*
validate_shape(*
_output_shapes
: 
ж
save/Assign_21AssignAquamanNet/netconv_3_2/kernelsave/RestoreV2:21*0
_class&
$"loc:@AquamanNet/netconv_3_2/kernel*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0
Ц
save/Assign_22AssignAquamanNet/netconv_4_1/biassave/RestoreV2:22*
use_locking(*
T0*.
_class$
" loc:@AquamanNet/netconv_4_1/bias*
validate_shape(*
_output_shapes
: 
ж
save/Assign_23AssignAquamanNet/netconv_4_1/kernelsave/RestoreV2:23*0
_class&
$"loc:@AquamanNet/netconv_4_1/kernel*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0
Ц
save/Assign_24AssignAquamanNet/netconv_4_2/biassave/RestoreV2:24*.
_class$
" loc:@AquamanNet/netconv_4_2/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
ж
save/Assign_25AssignAquamanNet/netconv_4_2/kernelsave/RestoreV2:25*
use_locking(*
T0*0
_class&
$"loc:@AquamanNet/netconv_4_2/kernel*
validate_shape(*&
_output_shapes
:  
Ђ
save/Assign_26Assignglobal_stepsave/RestoreV2:26*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
й
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"#
trainable_variablesћ"ј"
­
AquamanNet/netconv_0_0/kernel:0$AquamanNet/netconv_0_0/kernel/Assign$AquamanNet/netconv_0_0/kernel/read:02<AquamanNet/netconv_0_0/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_0_0/bias:0"AquamanNet/netconv_0_0/bias/Assign"AquamanNet/netconv_0_0/bias/read:02:AquamanNet/netconv_0_0/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_0_1/kernel:0$AquamanNet/netconv_0_1/kernel/Assign$AquamanNet/netconv_0_1/kernel/read:02<AquamanNet/netconv_0_1/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_0_1/bias:0"AquamanNet/netconv_0_1/bias/Assign"AquamanNet/netconv_0_1/bias/read:02:AquamanNet/netconv_0_1/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_1_1/kernel:0$AquamanNet/netconv_1_1/kernel/Assign$AquamanNet/netconv_1_1/kernel/read:02<AquamanNet/netconv_1_1/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_1_1/bias:0"AquamanNet/netconv_1_1/bias/Assign"AquamanNet/netconv_1_1/bias/read:02:AquamanNet/netconv_1_1/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_1_2/kernel:0$AquamanNet/netconv_1_2/kernel/Assign$AquamanNet/netconv_1_2/kernel/read:02<AquamanNet/netconv_1_2/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_1_2/bias:0"AquamanNet/netconv_1_2/bias/Assign"AquamanNet/netconv_1_2/bias/read:02:AquamanNet/netconv_1_2/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_2_1/kernel:0$AquamanNet/netconv_2_1/kernel/Assign$AquamanNet/netconv_2_1/kernel/read:02<AquamanNet/netconv_2_1/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_2_1/bias:0"AquamanNet/netconv_2_1/bias/Assign"AquamanNet/netconv_2_1/bias/read:02:AquamanNet/netconv_2_1/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_2_2/kernel:0$AquamanNet/netconv_2_2/kernel/Assign$AquamanNet/netconv_2_2/kernel/read:02<AquamanNet/netconv_2_2/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_2_2/bias:0"AquamanNet/netconv_2_2/bias/Assign"AquamanNet/netconv_2_2/bias/read:02:AquamanNet/netconv_2_2/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_3_1/kernel:0$AquamanNet/netconv_3_1/kernel/Assign$AquamanNet/netconv_3_1/kernel/read:02<AquamanNet/netconv_3_1/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_3_1/bias:0"AquamanNet/netconv_3_1/bias/Assign"AquamanNet/netconv_3_1/bias/read:02:AquamanNet/netconv_3_1/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_3_2/kernel:0$AquamanNet/netconv_3_2/kernel/Assign$AquamanNet/netconv_3_2/kernel/read:02<AquamanNet/netconv_3_2/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_3_2/bias:0"AquamanNet/netconv_3_2/bias/Assign"AquamanNet/netconv_3_2/bias/read:02:AquamanNet/netconv_3_2/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_4_1/kernel:0$AquamanNet/netconv_4_1/kernel/Assign$AquamanNet/netconv_4_1/kernel/read:02<AquamanNet/netconv_4_1/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_4_1/bias:0"AquamanNet/netconv_4_1/bias/Assign"AquamanNet/netconv_4_1/bias/read:02:AquamanNet/netconv_4_1/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_4_2/kernel:0$AquamanNet/netconv_4_2/kernel/Assign$AquamanNet/netconv_4_2/kernel/read:02<AquamanNet/netconv_4_2/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_4_2/bias:0"AquamanNet/netconv_4_2/bias/Assign"AquamanNet/netconv_4_2/bias/read:02:AquamanNet/netconv_4_2/bias/Initializer/truncated_normal:08
­
AquamanNet/net_dense_1/kernel:0$AquamanNet/net_dense_1/kernel/Assign$AquamanNet/net_dense_1/kernel/read:02<AquamanNet/net_dense_1/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/net_dense_1/bias:0"AquamanNet/net_dense_1/bias/Assign"AquamanNet/net_dense_1/bias/read:02:AquamanNet/net_dense_1/bias/Initializer/truncated_normal:08
­
AquamanNet/net_dense_2/kernel:0$AquamanNet/net_dense_2/kernel/Assign$AquamanNet/net_dense_2/kernel/read:02<AquamanNet/net_dense_2/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/net_dense_2/bias:0"AquamanNet/net_dense_2/bias/Assign"AquamanNet/net_dense_2/bias/read:02:AquamanNet/net_dense_2/bias/Initializer/truncated_normal:08
­
AquamanNet/net_dense_3/kernel:0$AquamanNet/net_dense_3/kernel/Assign$AquamanNet/net_dense_3/kernel/read:02<AquamanNet/net_dense_3/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/net_dense_3/bias:0"AquamanNet/net_dense_3/bias/Assign"AquamanNet/net_dense_3/bias/read:02:AquamanNet/net_dense_3/bias/Initializer/truncated_normal:08"гћ
while_contextРћМћ
Й
map/while/while_context
*map/while/LoopCond:02map/while/Merge:0:map/while/Identity:0Bmap/while/Exit:0Bmap/while/Exit_1:0Bmap/while/Exit_2:0JЮ
map/TensorArray:0
@map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map/TensorArray_1:0
map/strided_slice:0
map/while/DecodeJpeg:0
map/while/Enter:0
map/while/Enter_1:0
map/while/Enter_2:0
map/while/Exit:0
map/while/Exit_1:0
map/while/Exit_2:0
map/while/Identity:0
map/while/Identity_1:0
map/while/Identity_2:0
map/while/Less/Enter:0
map/while/Less:0
map/while/Less_1:0
map/while/LogicalAnd:0
map/while/LoopCond:0
map/while/Merge:0
map/while/Merge:1
map/while/Merge_1:0
map/while/Merge_1:1
map/while/Merge_2:0
map/while/Merge_2:1
map/while/NextIteration:0
map/while/NextIteration_1:0
map/while/NextIteration_2:0
!map/while/Reshape_Preproc/shape:0
map/while/Reshape_Preproc:0
map/while/Switch:0
map/while/Switch:1
map/while/Switch_1:0
map/while/Switch_1:1
map/while/Switch_2:0
map/while/Switch_2:1
#map/while/TensorArrayReadV3/Enter:0
%map/while/TensorArrayReadV3/Enter_1:0
map/while/TensorArrayReadV3:0
5map/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
/map/while/TensorArrayWrite/TensorArrayWriteV3:0
map/while/add/y:0
map/while/add:0
map/while/add_1/y:0
map/while/add_1:0
map/while/div/y:0
map/while/div:0
!map/while/resize/ExpandDims/dim:0
map/while/resize/ExpandDims:0
!map/while/resize/ResizeBilinear:0
map/while/resize/Squeeze:0
map/while/resize/size:0L
map/TensorArray_1:05map/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0i
@map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0%map/while/TensorArrayReadV3/Enter_1:08
map/TensorArray:0#map/while/TensorArrayReadV3/Enter:0-
map/strided_slice:0map/while/Less/Enter:0Rmap/while/Enter:0Rmap/while/Enter_1:0Rmap/while/Enter_2:0Zmap/strided_slice:0
Ч
map_1/while/while_context
*map_1/while/LoopCond:02map_1/while/Merge:0:map_1/while/Identity:0Bmap_1/while/Exit:0Bmap_1/while/Exit_1:0Bmap_1/while/Exit_2:0JЦ
map_1/TensorArray:0
Bmap_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_1/TensorArray_1:0
map_1/strided_slice:0
map_1/while/DecodeJpeg:0
map_1/while/Enter:0
map_1/while/Enter_1:0
map_1/while/Enter_2:0
map_1/while/Exit:0
map_1/while/Exit_1:0
map_1/while/Exit_2:0
map_1/while/Identity:0
map_1/while/Identity_1:0
map_1/while/Identity_2:0
map_1/while/Less/Enter:0
map_1/while/Less:0
map_1/while/Less_1:0
map_1/while/LogicalAnd:0
map_1/while/LoopCond:0
map_1/while/Merge:0
map_1/while/Merge:1
map_1/while/Merge_1:0
map_1/while/Merge_1:1
map_1/while/Merge_2:0
map_1/while/Merge_2:1
map_1/while/NextIteration:0
map_1/while/NextIteration_1:0
map_1/while/NextIteration_2:0
#map_1/while/Reshape_Preproc/shape:0
map_1/while/Reshape_Preproc:0
map_1/while/Switch:0
map_1/while/Switch:1
map_1/while/Switch_1:0
map_1/while/Switch_1:1
map_1/while/Switch_2:0
map_1/while/Switch_2:1
%map_1/while/TensorArrayReadV3/Enter:0
'map_1/while/TensorArrayReadV3/Enter_1:0
map_1/while/TensorArrayReadV3:0
7map_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
1map_1/while/TensorArrayWrite/TensorArrayWriteV3:0
map_1/while/add/y:0
map_1/while/add:0
map_1/while/add_1/y:0
map_1/while/add_1:0
map_1/while/div/y:0
map_1/while/div:0
#map_1/while/resize/ExpandDims/dim:0
map_1/while/resize/ExpandDims:0
#map_1/while/resize/ResizeBilinear:0
map_1/while/resize/Squeeze:0
map_1/while/resize/size:0m
Bmap_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0'map_1/while/TensorArrayReadV3/Enter_1:0P
map_1/TensorArray_1:07map_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:01
map_1/strided_slice:0map_1/while/Less/Enter:0<
map_1/TensorArray:0%map_1/while/TensorArrayReadV3/Enter:0Rmap_1/while/Enter:0Rmap_1/while/Enter_1:0Rmap_1/while/Enter_2:0Zmap_1/strided_slice:0
Ч
map_2/while/while_context
*map_2/while/LoopCond:02map_2/while/Merge:0:map_2/while/Identity:0Bmap_2/while/Exit:0Bmap_2/while/Exit_1:0Bmap_2/while/Exit_2:0JЦ
map_2/TensorArray:0
Bmap_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_2/TensorArray_1:0
map_2/strided_slice:0
map_2/while/DecodeJpeg:0
map_2/while/Enter:0
map_2/while/Enter_1:0
map_2/while/Enter_2:0
map_2/while/Exit:0
map_2/while/Exit_1:0
map_2/while/Exit_2:0
map_2/while/Identity:0
map_2/while/Identity_1:0
map_2/while/Identity_2:0
map_2/while/Less/Enter:0
map_2/while/Less:0
map_2/while/Less_1:0
map_2/while/LogicalAnd:0
map_2/while/LoopCond:0
map_2/while/Merge:0
map_2/while/Merge:1
map_2/while/Merge_1:0
map_2/while/Merge_1:1
map_2/while/Merge_2:0
map_2/while/Merge_2:1
map_2/while/NextIteration:0
map_2/while/NextIteration_1:0
map_2/while/NextIteration_2:0
#map_2/while/Reshape_Preproc/shape:0
map_2/while/Reshape_Preproc:0
map_2/while/Switch:0
map_2/while/Switch:1
map_2/while/Switch_1:0
map_2/while/Switch_1:1
map_2/while/Switch_2:0
map_2/while/Switch_2:1
%map_2/while/TensorArrayReadV3/Enter:0
'map_2/while/TensorArrayReadV3/Enter_1:0
map_2/while/TensorArrayReadV3:0
7map_2/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
1map_2/while/TensorArrayWrite/TensorArrayWriteV3:0
map_2/while/add/y:0
map_2/while/add:0
map_2/while/add_1/y:0
map_2/while/add_1:0
map_2/while/div/y:0
map_2/while/div:0
#map_2/while/resize/ExpandDims/dim:0
map_2/while/resize/ExpandDims:0
#map_2/while/resize/ResizeBilinear:0
map_2/while/resize/Squeeze:0
map_2/while/resize/size:0P
map_2/TensorArray_1:07map_2/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0<
map_2/TensorArray:0%map_2/while/TensorArrayReadV3/Enter:0m
Bmap_2/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0'map_2/while/TensorArrayReadV3/Enter_1:01
map_2/strided_slice:0map_2/while/Less/Enter:0Rmap_2/while/Enter:0Rmap_2/while/Enter_1:0Rmap_2/while/Enter_2:0Zmap_2/strided_slice:0
Ч
map_3/while/while_context
*map_3/while/LoopCond:02map_3/while/Merge:0:map_3/while/Identity:0Bmap_3/while/Exit:0Bmap_3/while/Exit_1:0Bmap_3/while/Exit_2:0JЦ
map_3/TensorArray:0
Bmap_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_3/TensorArray_1:0
map_3/strided_slice:0
map_3/while/DecodeJpeg:0
map_3/while/Enter:0
map_3/while/Enter_1:0
map_3/while/Enter_2:0
map_3/while/Exit:0
map_3/while/Exit_1:0
map_3/while/Exit_2:0
map_3/while/Identity:0
map_3/while/Identity_1:0
map_3/while/Identity_2:0
map_3/while/Less/Enter:0
map_3/while/Less:0
map_3/while/Less_1:0
map_3/while/LogicalAnd:0
map_3/while/LoopCond:0
map_3/while/Merge:0
map_3/while/Merge:1
map_3/while/Merge_1:0
map_3/while/Merge_1:1
map_3/while/Merge_2:0
map_3/while/Merge_2:1
map_3/while/NextIteration:0
map_3/while/NextIteration_1:0
map_3/while/NextIteration_2:0
#map_3/while/Reshape_Preproc/shape:0
map_3/while/Reshape_Preproc:0
map_3/while/Switch:0
map_3/while/Switch:1
map_3/while/Switch_1:0
map_3/while/Switch_1:1
map_3/while/Switch_2:0
map_3/while/Switch_2:1
%map_3/while/TensorArrayReadV3/Enter:0
'map_3/while/TensorArrayReadV3/Enter_1:0
map_3/while/TensorArrayReadV3:0
7map_3/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
1map_3/while/TensorArrayWrite/TensorArrayWriteV3:0
map_3/while/add/y:0
map_3/while/add:0
map_3/while/add_1/y:0
map_3/while/add_1:0
map_3/while/div/y:0
map_3/while/div:0
#map_3/while/resize/ExpandDims/dim:0
map_3/while/resize/ExpandDims:0
#map_3/while/resize/ResizeBilinear:0
map_3/while/resize/Squeeze:0
map_3/while/resize/size:0P
map_3/TensorArray_1:07map_3/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0m
Bmap_3/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0'map_3/while/TensorArrayReadV3/Enter_1:0<
map_3/TensorArray:0%map_3/while/TensorArrayReadV3/Enter:01
map_3/strided_slice:0map_3/while/Less/Enter:0Rmap_3/while/Enter:0Rmap_3/while/Enter_1:0Rmap_3/while/Enter_2:0Zmap_3/strided_slice:0
Ч
map_4/while/while_context
*map_4/while/LoopCond:02map_4/while/Merge:0:map_4/while/Identity:0Bmap_4/while/Exit:0Bmap_4/while/Exit_1:0Bmap_4/while/Exit_2:0JЦ
map_4/TensorArray:0
Bmap_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_4/TensorArray_1:0
map_4/strided_slice:0
map_4/while/DecodeJpeg:0
map_4/while/Enter:0
map_4/while/Enter_1:0
map_4/while/Enter_2:0
map_4/while/Exit:0
map_4/while/Exit_1:0
map_4/while/Exit_2:0
map_4/while/Identity:0
map_4/while/Identity_1:0
map_4/while/Identity_2:0
map_4/while/Less/Enter:0
map_4/while/Less:0
map_4/while/Less_1:0
map_4/while/LogicalAnd:0
map_4/while/LoopCond:0
map_4/while/Merge:0
map_4/while/Merge:1
map_4/while/Merge_1:0
map_4/while/Merge_1:1
map_4/while/Merge_2:0
map_4/while/Merge_2:1
map_4/while/NextIteration:0
map_4/while/NextIteration_1:0
map_4/while/NextIteration_2:0
#map_4/while/Reshape_Preproc/shape:0
map_4/while/Reshape_Preproc:0
map_4/while/Switch:0
map_4/while/Switch:1
map_4/while/Switch_1:0
map_4/while/Switch_1:1
map_4/while/Switch_2:0
map_4/while/Switch_2:1
%map_4/while/TensorArrayReadV3/Enter:0
'map_4/while/TensorArrayReadV3/Enter_1:0
map_4/while/TensorArrayReadV3:0
7map_4/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
1map_4/while/TensorArrayWrite/TensorArrayWriteV3:0
map_4/while/add/y:0
map_4/while/add:0
map_4/while/add_1/y:0
map_4/while/add_1:0
map_4/while/div/y:0
map_4/while/div:0
#map_4/while/resize/ExpandDims/dim:0
map_4/while/resize/ExpandDims:0
#map_4/while/resize/ResizeBilinear:0
map_4/while/resize/Squeeze:0
map_4/while/resize/size:0P
map_4/TensorArray_1:07map_4/while/TensorArrayWrite/TensorArrayWriteV3/Enter:01
map_4/strided_slice:0map_4/while/Less/Enter:0m
Bmap_4/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0'map_4/while/TensorArrayReadV3/Enter_1:0<
map_4/TensorArray:0%map_4/while/TensorArrayReadV3/Enter:0Rmap_4/while/Enter:0Rmap_4/while/Enter_1:0Rmap_4/while/Enter_2:0Zmap_4/strided_slice:0
Ч
map_5/while/while_context
*map_5/while/LoopCond:02map_5/while/Merge:0:map_5/while/Identity:0Bmap_5/while/Exit:0Bmap_5/while/Exit_1:0Bmap_5/while/Exit_2:0JЦ
map_5/TensorArray:0
Bmap_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_5/TensorArray_1:0
map_5/strided_slice:0
map_5/while/DecodeJpeg:0
map_5/while/Enter:0
map_5/while/Enter_1:0
map_5/while/Enter_2:0
map_5/while/Exit:0
map_5/while/Exit_1:0
map_5/while/Exit_2:0
map_5/while/Identity:0
map_5/while/Identity_1:0
map_5/while/Identity_2:0
map_5/while/Less/Enter:0
map_5/while/Less:0
map_5/while/Less_1:0
map_5/while/LogicalAnd:0
map_5/while/LoopCond:0
map_5/while/Merge:0
map_5/while/Merge:1
map_5/while/Merge_1:0
map_5/while/Merge_1:1
map_5/while/Merge_2:0
map_5/while/Merge_2:1
map_5/while/NextIteration:0
map_5/while/NextIteration_1:0
map_5/while/NextIteration_2:0
#map_5/while/Reshape_Preproc/shape:0
map_5/while/Reshape_Preproc:0
map_5/while/Switch:0
map_5/while/Switch:1
map_5/while/Switch_1:0
map_5/while/Switch_1:1
map_5/while/Switch_2:0
map_5/while/Switch_2:1
%map_5/while/TensorArrayReadV3/Enter:0
'map_5/while/TensorArrayReadV3/Enter_1:0
map_5/while/TensorArrayReadV3:0
7map_5/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
1map_5/while/TensorArrayWrite/TensorArrayWriteV3:0
map_5/while/add/y:0
map_5/while/add:0
map_5/while/add_1/y:0
map_5/while/add_1:0
map_5/while/div/y:0
map_5/while/div:0
#map_5/while/resize/ExpandDims/dim:0
map_5/while/resize/ExpandDims:0
#map_5/while/resize/ResizeBilinear:0
map_5/while/resize/Squeeze:0
map_5/while/resize/size:0m
Bmap_5/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0'map_5/while/TensorArrayReadV3/Enter_1:0P
map_5/TensorArray_1:07map_5/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0<
map_5/TensorArray:0%map_5/while/TensorArrayReadV3/Enter:01
map_5/strided_slice:0map_5/while/Less/Enter:0Rmap_5/while/Enter:0Rmap_5/while/Enter_1:0Rmap_5/while/Enter_2:0Zmap_5/strided_slice:0
Ч
map_6/while/while_context
*map_6/while/LoopCond:02map_6/while/Merge:0:map_6/while/Identity:0Bmap_6/while/Exit:0Bmap_6/while/Exit_1:0Bmap_6/while/Exit_2:0JЦ
map_6/TensorArray:0
Bmap_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_6/TensorArray_1:0
map_6/strided_slice:0
map_6/while/DecodeJpeg:0
map_6/while/Enter:0
map_6/while/Enter_1:0
map_6/while/Enter_2:0
map_6/while/Exit:0
map_6/while/Exit_1:0
map_6/while/Exit_2:0
map_6/while/Identity:0
map_6/while/Identity_1:0
map_6/while/Identity_2:0
map_6/while/Less/Enter:0
map_6/while/Less:0
map_6/while/Less_1:0
map_6/while/LogicalAnd:0
map_6/while/LoopCond:0
map_6/while/Merge:0
map_6/while/Merge:1
map_6/while/Merge_1:0
map_6/while/Merge_1:1
map_6/while/Merge_2:0
map_6/while/Merge_2:1
map_6/while/NextIteration:0
map_6/while/NextIteration_1:0
map_6/while/NextIteration_2:0
#map_6/while/Reshape_Preproc/shape:0
map_6/while/Reshape_Preproc:0
map_6/while/Switch:0
map_6/while/Switch:1
map_6/while/Switch_1:0
map_6/while/Switch_1:1
map_6/while/Switch_2:0
map_6/while/Switch_2:1
%map_6/while/TensorArrayReadV3/Enter:0
'map_6/while/TensorArrayReadV3/Enter_1:0
map_6/while/TensorArrayReadV3:0
7map_6/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
1map_6/while/TensorArrayWrite/TensorArrayWriteV3:0
map_6/while/add/y:0
map_6/while/add:0
map_6/while/add_1/y:0
map_6/while/add_1:0
map_6/while/div/y:0
map_6/while/div:0
#map_6/while/resize/ExpandDims/dim:0
map_6/while/resize/ExpandDims:0
#map_6/while/resize/ResizeBilinear:0
map_6/while/resize/Squeeze:0
map_6/while/resize/size:0<
map_6/TensorArray:0%map_6/while/TensorArrayReadV3/Enter:0m
Bmap_6/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0'map_6/while/TensorArrayReadV3/Enter_1:01
map_6/strided_slice:0map_6/while/Less/Enter:0P
map_6/TensorArray_1:07map_6/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Rmap_6/while/Enter:0Rmap_6/while/Enter_1:0Rmap_6/while/Enter_2:0Zmap_6/strided_slice:0
Ч
map_7/while/while_context
*map_7/while/LoopCond:02map_7/while/Merge:0:map_7/while/Identity:0Bmap_7/while/Exit:0Bmap_7/while/Exit_1:0Bmap_7/while/Exit_2:0JЦ
map_7/TensorArray:0
Bmap_7/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_7/TensorArray_1:0
map_7/strided_slice:0
map_7/while/DecodeJpeg:0
map_7/while/Enter:0
map_7/while/Enter_1:0
map_7/while/Enter_2:0
map_7/while/Exit:0
map_7/while/Exit_1:0
map_7/while/Exit_2:0
map_7/while/Identity:0
map_7/while/Identity_1:0
map_7/while/Identity_2:0
map_7/while/Less/Enter:0
map_7/while/Less:0
map_7/while/Less_1:0
map_7/while/LogicalAnd:0
map_7/while/LoopCond:0
map_7/while/Merge:0
map_7/while/Merge:1
map_7/while/Merge_1:0
map_7/while/Merge_1:1
map_7/while/Merge_2:0
map_7/while/Merge_2:1
map_7/while/NextIteration:0
map_7/while/NextIteration_1:0
map_7/while/NextIteration_2:0
#map_7/while/Reshape_Preproc/shape:0
map_7/while/Reshape_Preproc:0
map_7/while/Switch:0
map_7/while/Switch:1
map_7/while/Switch_1:0
map_7/while/Switch_1:1
map_7/while/Switch_2:0
map_7/while/Switch_2:1
%map_7/while/TensorArrayReadV3/Enter:0
'map_7/while/TensorArrayReadV3/Enter_1:0
map_7/while/TensorArrayReadV3:0
7map_7/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
1map_7/while/TensorArrayWrite/TensorArrayWriteV3:0
map_7/while/add/y:0
map_7/while/add:0
map_7/while/add_1/y:0
map_7/while/add_1:0
map_7/while/div/y:0
map_7/while/div:0
#map_7/while/resize/ExpandDims/dim:0
map_7/while/resize/ExpandDims:0
#map_7/while/resize/ResizeBilinear:0
map_7/while/resize/Squeeze:0
map_7/while/resize/size:0m
Bmap_7/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0'map_7/while/TensorArrayReadV3/Enter_1:0P
map_7/TensorArray_1:07map_7/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0<
map_7/TensorArray:0%map_7/while/TensorArrayReadV3/Enter:01
map_7/strided_slice:0map_7/while/Less/Enter:0Rmap_7/while/Enter:0Rmap_7/while/Enter_1:0Rmap_7/while/Enter_2:0Zmap_7/strided_slice:0
Ч
map_8/while/while_context
*map_8/while/LoopCond:02map_8/while/Merge:0:map_8/while/Identity:0Bmap_8/while/Exit:0Bmap_8/while/Exit_1:0Bmap_8/while/Exit_2:0JЦ
map_8/TensorArray:0
Bmap_8/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_8/TensorArray_1:0
map_8/strided_slice:0
map_8/while/DecodeJpeg:0
map_8/while/Enter:0
map_8/while/Enter_1:0
map_8/while/Enter_2:0
map_8/while/Exit:0
map_8/while/Exit_1:0
map_8/while/Exit_2:0
map_8/while/Identity:0
map_8/while/Identity_1:0
map_8/while/Identity_2:0
map_8/while/Less/Enter:0
map_8/while/Less:0
map_8/while/Less_1:0
map_8/while/LogicalAnd:0
map_8/while/LoopCond:0
map_8/while/Merge:0
map_8/while/Merge:1
map_8/while/Merge_1:0
map_8/while/Merge_1:1
map_8/while/Merge_2:0
map_8/while/Merge_2:1
map_8/while/NextIteration:0
map_8/while/NextIteration_1:0
map_8/while/NextIteration_2:0
#map_8/while/Reshape_Preproc/shape:0
map_8/while/Reshape_Preproc:0
map_8/while/Switch:0
map_8/while/Switch:1
map_8/while/Switch_1:0
map_8/while/Switch_1:1
map_8/while/Switch_2:0
map_8/while/Switch_2:1
%map_8/while/TensorArrayReadV3/Enter:0
'map_8/while/TensorArrayReadV3/Enter_1:0
map_8/while/TensorArrayReadV3:0
7map_8/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
1map_8/while/TensorArrayWrite/TensorArrayWriteV3:0
map_8/while/add/y:0
map_8/while/add:0
map_8/while/add_1/y:0
map_8/while/add_1:0
map_8/while/div/y:0
map_8/while/div:0
#map_8/while/resize/ExpandDims/dim:0
map_8/while/resize/ExpandDims:0
#map_8/while/resize/ResizeBilinear:0
map_8/while/resize/Squeeze:0
map_8/while/resize/size:0P
map_8/TensorArray_1:07map_8/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0<
map_8/TensorArray:0%map_8/while/TensorArrayReadV3/Enter:01
map_8/strided_slice:0map_8/while/Less/Enter:0m
Bmap_8/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0'map_8/while/TensorArrayReadV3/Enter_1:0Rmap_8/while/Enter:0Rmap_8/while/Enter_1:0Rmap_8/while/Enter_2:0Zmap_8/strided_slice:0
Ч
map_9/while/while_context
*map_9/while/LoopCond:02map_9/while/Merge:0:map_9/while/Identity:0Bmap_9/while/Exit:0Bmap_9/while/Exit_1:0Bmap_9/while/Exit_2:0JЦ
map_9/TensorArray:0
Bmap_9/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_9/TensorArray_1:0
map_9/strided_slice:0
map_9/while/DecodeJpeg:0
map_9/while/Enter:0
map_9/while/Enter_1:0
map_9/while/Enter_2:0
map_9/while/Exit:0
map_9/while/Exit_1:0
map_9/while/Exit_2:0
map_9/while/Identity:0
map_9/while/Identity_1:0
map_9/while/Identity_2:0
map_9/while/Less/Enter:0
map_9/while/Less:0
map_9/while/Less_1:0
map_9/while/LogicalAnd:0
map_9/while/LoopCond:0
map_9/while/Merge:0
map_9/while/Merge:1
map_9/while/Merge_1:0
map_9/while/Merge_1:1
map_9/while/Merge_2:0
map_9/while/Merge_2:1
map_9/while/NextIteration:0
map_9/while/NextIteration_1:0
map_9/while/NextIteration_2:0
#map_9/while/Reshape_Preproc/shape:0
map_9/while/Reshape_Preproc:0
map_9/while/Switch:0
map_9/while/Switch:1
map_9/while/Switch_1:0
map_9/while/Switch_1:1
map_9/while/Switch_2:0
map_9/while/Switch_2:1
%map_9/while/TensorArrayReadV3/Enter:0
'map_9/while/TensorArrayReadV3/Enter_1:0
map_9/while/TensorArrayReadV3:0
7map_9/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
1map_9/while/TensorArrayWrite/TensorArrayWriteV3:0
map_9/while/add/y:0
map_9/while/add:0
map_9/while/add_1/y:0
map_9/while/add_1:0
map_9/while/div/y:0
map_9/while/div:0
#map_9/while/resize/ExpandDims/dim:0
map_9/while/resize/ExpandDims:0
#map_9/while/resize/ResizeBilinear:0
map_9/while/resize/Squeeze:0
map_9/while/resize/size:0m
Bmap_9/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0'map_9/while/TensorArrayReadV3/Enter_1:0P
map_9/TensorArray_1:07map_9/while/TensorArrayWrite/TensorArrayWriteV3/Enter:01
map_9/strided_slice:0map_9/while/Less/Enter:0<
map_9/TensorArray:0%map_9/while/TensorArrayReadV3/Enter:0Rmap_9/while/Enter:0Rmap_9/while/Enter_1:0Rmap_9/while/Enter_2:0Zmap_9/strided_slice:0

map_10/while/while_context
*map_10/while/LoopCond:02map_10/while/Merge:0:map_10/while/Identity:0Bmap_10/while/Exit:0Bmap_10/while/Exit_1:0Bmap_10/while/Exit_2:0J
map_10/TensorArray:0
Cmap_10/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_10/TensorArray_1:0
map_10/strided_slice:0
map_10/while/DecodeJpeg:0
map_10/while/Enter:0
map_10/while/Enter_1:0
map_10/while/Enter_2:0
map_10/while/Exit:0
map_10/while/Exit_1:0
map_10/while/Exit_2:0
map_10/while/Identity:0
map_10/while/Identity_1:0
map_10/while/Identity_2:0
map_10/while/Less/Enter:0
map_10/while/Less:0
map_10/while/Less_1:0
map_10/while/LogicalAnd:0
map_10/while/LoopCond:0
map_10/while/Merge:0
map_10/while/Merge:1
map_10/while/Merge_1:0
map_10/while/Merge_1:1
map_10/while/Merge_2:0
map_10/while/Merge_2:1
map_10/while/NextIteration:0
map_10/while/NextIteration_1:0
map_10/while/NextIteration_2:0
$map_10/while/Reshape_Preproc/shape:0
map_10/while/Reshape_Preproc:0
map_10/while/Switch:0
map_10/while/Switch:1
map_10/while/Switch_1:0
map_10/while/Switch_1:1
map_10/while/Switch_2:0
map_10/while/Switch_2:1
&map_10/while/TensorArrayReadV3/Enter:0
(map_10/while/TensorArrayReadV3/Enter_1:0
 map_10/while/TensorArrayReadV3:0
8map_10/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
2map_10/while/TensorArrayWrite/TensorArrayWriteV3:0
map_10/while/add/y:0
map_10/while/add:0
map_10/while/add_1/y:0
map_10/while/add_1:0
map_10/while/div/y:0
map_10/while/div:0
$map_10/while/resize/ExpandDims/dim:0
 map_10/while/resize/ExpandDims:0
$map_10/while/resize/ResizeBilinear:0
map_10/while/resize/Squeeze:0
map_10/while/resize/size:0o
Cmap_10/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0(map_10/while/TensorArrayReadV3/Enter_1:0R
map_10/TensorArray_1:08map_10/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0>
map_10/TensorArray:0&map_10/while/TensorArrayReadV3/Enter:03
map_10/strided_slice:0map_10/while/Less/Enter:0Rmap_10/while/Enter:0Rmap_10/while/Enter_1:0Rmap_10/while/Enter_2:0Zmap_10/strided_slice:0

map_11/while/while_context
*map_11/while/LoopCond:02map_11/while/Merge:0:map_11/while/Identity:0Bmap_11/while/Exit:0Bmap_11/while/Exit_1:0Bmap_11/while/Exit_2:0J
map_11/TensorArray:0
Cmap_11/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_11/TensorArray_1:0
map_11/strided_slice:0
map_11/while/DecodeJpeg:0
map_11/while/Enter:0
map_11/while/Enter_1:0
map_11/while/Enter_2:0
map_11/while/Exit:0
map_11/while/Exit_1:0
map_11/while/Exit_2:0
map_11/while/Identity:0
map_11/while/Identity_1:0
map_11/while/Identity_2:0
map_11/while/Less/Enter:0
map_11/while/Less:0
map_11/while/Less_1:0
map_11/while/LogicalAnd:0
map_11/while/LoopCond:0
map_11/while/Merge:0
map_11/while/Merge:1
map_11/while/Merge_1:0
map_11/while/Merge_1:1
map_11/while/Merge_2:0
map_11/while/Merge_2:1
map_11/while/NextIteration:0
map_11/while/NextIteration_1:0
map_11/while/NextIteration_2:0
$map_11/while/Reshape_Preproc/shape:0
map_11/while/Reshape_Preproc:0
map_11/while/Switch:0
map_11/while/Switch:1
map_11/while/Switch_1:0
map_11/while/Switch_1:1
map_11/while/Switch_2:0
map_11/while/Switch_2:1
&map_11/while/TensorArrayReadV3/Enter:0
(map_11/while/TensorArrayReadV3/Enter_1:0
 map_11/while/TensorArrayReadV3:0
8map_11/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
2map_11/while/TensorArrayWrite/TensorArrayWriteV3:0
map_11/while/add/y:0
map_11/while/add:0
map_11/while/add_1/y:0
map_11/while/add_1:0
map_11/while/div/y:0
map_11/while/div:0
$map_11/while/resize/ExpandDims/dim:0
 map_11/while/resize/ExpandDims:0
$map_11/while/resize/ResizeBilinear:0
map_11/while/resize/Squeeze:0
map_11/while/resize/size:0>
map_11/TensorArray:0&map_11/while/TensorArrayReadV3/Enter:0o
Cmap_11/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0(map_11/while/TensorArrayReadV3/Enter_1:03
map_11/strided_slice:0map_11/while/Less/Enter:0R
map_11/TensorArray_1:08map_11/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Rmap_11/while/Enter:0Rmap_11/while/Enter_1:0Rmap_11/while/Enter_2:0Zmap_11/strided_slice:0

map_12/while/while_context
*map_12/while/LoopCond:02map_12/while/Merge:0:map_12/while/Identity:0Bmap_12/while/Exit:0Bmap_12/while/Exit_1:0Bmap_12/while/Exit_2:0J
map_12/TensorArray:0
Cmap_12/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_12/TensorArray_1:0
map_12/strided_slice:0
map_12/while/DecodeJpeg:0
map_12/while/Enter:0
map_12/while/Enter_1:0
map_12/while/Enter_2:0
map_12/while/Exit:0
map_12/while/Exit_1:0
map_12/while/Exit_2:0
map_12/while/Identity:0
map_12/while/Identity_1:0
map_12/while/Identity_2:0
map_12/while/Less/Enter:0
map_12/while/Less:0
map_12/while/Less_1:0
map_12/while/LogicalAnd:0
map_12/while/LoopCond:0
map_12/while/Merge:0
map_12/while/Merge:1
map_12/while/Merge_1:0
map_12/while/Merge_1:1
map_12/while/Merge_2:0
map_12/while/Merge_2:1
map_12/while/NextIteration:0
map_12/while/NextIteration_1:0
map_12/while/NextIteration_2:0
$map_12/while/Reshape_Preproc/shape:0
map_12/while/Reshape_Preproc:0
map_12/while/Switch:0
map_12/while/Switch:1
map_12/while/Switch_1:0
map_12/while/Switch_1:1
map_12/while/Switch_2:0
map_12/while/Switch_2:1
&map_12/while/TensorArrayReadV3/Enter:0
(map_12/while/TensorArrayReadV3/Enter_1:0
 map_12/while/TensorArrayReadV3:0
8map_12/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
2map_12/while/TensorArrayWrite/TensorArrayWriteV3:0
map_12/while/add/y:0
map_12/while/add:0
map_12/while/add_1/y:0
map_12/while/add_1:0
map_12/while/div/y:0
map_12/while/div:0
$map_12/while/resize/ExpandDims/dim:0
 map_12/while/resize/ExpandDims:0
$map_12/while/resize/ResizeBilinear:0
map_12/while/resize/Squeeze:0
map_12/while/resize/size:0o
Cmap_12/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0(map_12/while/TensorArrayReadV3/Enter_1:0R
map_12/TensorArray_1:08map_12/while/TensorArrayWrite/TensorArrayWriteV3/Enter:03
map_12/strided_slice:0map_12/while/Less/Enter:0>
map_12/TensorArray:0&map_12/while/TensorArrayReadV3/Enter:0Rmap_12/while/Enter:0Rmap_12/while/Enter_1:0Rmap_12/while/Enter_2:0Zmap_12/strided_slice:0

map_13/while/while_context
*map_13/while/LoopCond:02map_13/while/Merge:0:map_13/while/Identity:0Bmap_13/while/Exit:0Bmap_13/while/Exit_1:0Bmap_13/while/Exit_2:0J
map_13/TensorArray:0
Cmap_13/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_13/TensorArray_1:0
map_13/strided_slice:0
map_13/while/DecodeJpeg:0
map_13/while/Enter:0
map_13/while/Enter_1:0
map_13/while/Enter_2:0
map_13/while/Exit:0
map_13/while/Exit_1:0
map_13/while/Exit_2:0
map_13/while/Identity:0
map_13/while/Identity_1:0
map_13/while/Identity_2:0
map_13/while/Less/Enter:0
map_13/while/Less:0
map_13/while/Less_1:0
map_13/while/LogicalAnd:0
map_13/while/LoopCond:0
map_13/while/Merge:0
map_13/while/Merge:1
map_13/while/Merge_1:0
map_13/while/Merge_1:1
map_13/while/Merge_2:0
map_13/while/Merge_2:1
map_13/while/NextIteration:0
map_13/while/NextIteration_1:0
map_13/while/NextIteration_2:0
$map_13/while/Reshape_Preproc/shape:0
map_13/while/Reshape_Preproc:0
map_13/while/Switch:0
map_13/while/Switch:1
map_13/while/Switch_1:0
map_13/while/Switch_1:1
map_13/while/Switch_2:0
map_13/while/Switch_2:1
&map_13/while/TensorArrayReadV3/Enter:0
(map_13/while/TensorArrayReadV3/Enter_1:0
 map_13/while/TensorArrayReadV3:0
8map_13/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
2map_13/while/TensorArrayWrite/TensorArrayWriteV3:0
map_13/while/add/y:0
map_13/while/add:0
map_13/while/add_1/y:0
map_13/while/add_1:0
map_13/while/div/y:0
map_13/while/div:0
$map_13/while/resize/ExpandDims/dim:0
 map_13/while/resize/ExpandDims:0
$map_13/while/resize/ResizeBilinear:0
map_13/while/resize/Squeeze:0
map_13/while/resize/size:0R
map_13/TensorArray_1:08map_13/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0>
map_13/TensorArray:0&map_13/while/TensorArrayReadV3/Enter:0o
Cmap_13/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0(map_13/while/TensorArrayReadV3/Enter_1:03
map_13/strided_slice:0map_13/while/Less/Enter:0Rmap_13/while/Enter:0Rmap_13/while/Enter_1:0Rmap_13/while/Enter_2:0Zmap_13/strided_slice:0

map_14/while/while_context
*map_14/while/LoopCond:02map_14/while/Merge:0:map_14/while/Identity:0Bmap_14/while/Exit:0Bmap_14/while/Exit_1:0Bmap_14/while/Exit_2:0J
map_14/TensorArray:0
Cmap_14/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_14/TensorArray_1:0
map_14/strided_slice:0
map_14/while/DecodeJpeg:0
map_14/while/Enter:0
map_14/while/Enter_1:0
map_14/while/Enter_2:0
map_14/while/Exit:0
map_14/while/Exit_1:0
map_14/while/Exit_2:0
map_14/while/Identity:0
map_14/while/Identity_1:0
map_14/while/Identity_2:0
map_14/while/Less/Enter:0
map_14/while/Less:0
map_14/while/Less_1:0
map_14/while/LogicalAnd:0
map_14/while/LoopCond:0
map_14/while/Merge:0
map_14/while/Merge:1
map_14/while/Merge_1:0
map_14/while/Merge_1:1
map_14/while/Merge_2:0
map_14/while/Merge_2:1
map_14/while/NextIteration:0
map_14/while/NextIteration_1:0
map_14/while/NextIteration_2:0
$map_14/while/Reshape_Preproc/shape:0
map_14/while/Reshape_Preproc:0
map_14/while/Switch:0
map_14/while/Switch:1
map_14/while/Switch_1:0
map_14/while/Switch_1:1
map_14/while/Switch_2:0
map_14/while/Switch_2:1
&map_14/while/TensorArrayReadV3/Enter:0
(map_14/while/TensorArrayReadV3/Enter_1:0
 map_14/while/TensorArrayReadV3:0
8map_14/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
2map_14/while/TensorArrayWrite/TensorArrayWriteV3:0
map_14/while/add/y:0
map_14/while/add:0
map_14/while/add_1/y:0
map_14/while/add_1:0
map_14/while/div/y:0
map_14/while/div:0
$map_14/while/resize/ExpandDims/dim:0
 map_14/while/resize/ExpandDims:0
$map_14/while/resize/ResizeBilinear:0
map_14/while/resize/Squeeze:0
map_14/while/resize/size:0o
Cmap_14/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0(map_14/while/TensorArrayReadV3/Enter_1:0R
map_14/TensorArray_1:08map_14/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0>
map_14/TensorArray:0&map_14/while/TensorArrayReadV3/Enter:03
map_14/strided_slice:0map_14/while/Less/Enter:0Rmap_14/while/Enter:0Rmap_14/while/Enter_1:0Rmap_14/while/Enter_2:0Zmap_14/strided_slice:0

map_15/while/while_context
*map_15/while/LoopCond:02map_15/while/Merge:0:map_15/while/Identity:0Bmap_15/while/Exit:0Bmap_15/while/Exit_1:0Bmap_15/while/Exit_2:0J
map_15/TensorArray:0
Cmap_15/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map_15/TensorArray_1:0
map_15/strided_slice:0
map_15/while/DecodeJpeg:0
map_15/while/Enter:0
map_15/while/Enter_1:0
map_15/while/Enter_2:0
map_15/while/Exit:0
map_15/while/Exit_1:0
map_15/while/Exit_2:0
map_15/while/Identity:0
map_15/while/Identity_1:0
map_15/while/Identity_2:0
map_15/while/Less/Enter:0
map_15/while/Less:0
map_15/while/Less_1:0
map_15/while/LogicalAnd:0
map_15/while/LoopCond:0
map_15/while/Merge:0
map_15/while/Merge:1
map_15/while/Merge_1:0
map_15/while/Merge_1:1
map_15/while/Merge_2:0
map_15/while/Merge_2:1
map_15/while/NextIteration:0
map_15/while/NextIteration_1:0
map_15/while/NextIteration_2:0
$map_15/while/Reshape_Preproc/shape:0
map_15/while/Reshape_Preproc:0
map_15/while/Switch:0
map_15/while/Switch:1
map_15/while/Switch_1:0
map_15/while/Switch_1:1
map_15/while/Switch_2:0
map_15/while/Switch_2:1
&map_15/while/TensorArrayReadV3/Enter:0
(map_15/while/TensorArrayReadV3/Enter_1:0
 map_15/while/TensorArrayReadV3:0
8map_15/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
2map_15/while/TensorArrayWrite/TensorArrayWriteV3:0
map_15/while/add/y:0
map_15/while/add:0
map_15/while/add_1/y:0
map_15/while/add_1:0
map_15/while/div/y:0
map_15/while/div:0
$map_15/while/resize/ExpandDims/dim:0
 map_15/while/resize/ExpandDims:0
$map_15/while/resize/ResizeBilinear:0
map_15/while/resize/Squeeze:0
map_15/while/resize/size:0R
map_15/TensorArray_1:08map_15/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0>
map_15/TensorArray:0&map_15/while/TensorArrayReadV3/Enter:0o
Cmap_15/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0(map_15/while/TensorArrayReadV3/Enter_1:03
map_15/strided_slice:0map_15/while/Less/Enter:0Rmap_15/while/Enter:0Rmap_15/while/Enter_1:0Rmap_15/while/Enter_2:0Zmap_15/strided_slice:0"у#
	variablesе#в#
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
­
AquamanNet/netconv_0_0/kernel:0$AquamanNet/netconv_0_0/kernel/Assign$AquamanNet/netconv_0_0/kernel/read:02<AquamanNet/netconv_0_0/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_0_0/bias:0"AquamanNet/netconv_0_0/bias/Assign"AquamanNet/netconv_0_0/bias/read:02:AquamanNet/netconv_0_0/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_0_1/kernel:0$AquamanNet/netconv_0_1/kernel/Assign$AquamanNet/netconv_0_1/kernel/read:02<AquamanNet/netconv_0_1/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_0_1/bias:0"AquamanNet/netconv_0_1/bias/Assign"AquamanNet/netconv_0_1/bias/read:02:AquamanNet/netconv_0_1/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_1_1/kernel:0$AquamanNet/netconv_1_1/kernel/Assign$AquamanNet/netconv_1_1/kernel/read:02<AquamanNet/netconv_1_1/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_1_1/bias:0"AquamanNet/netconv_1_1/bias/Assign"AquamanNet/netconv_1_1/bias/read:02:AquamanNet/netconv_1_1/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_1_2/kernel:0$AquamanNet/netconv_1_2/kernel/Assign$AquamanNet/netconv_1_2/kernel/read:02<AquamanNet/netconv_1_2/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_1_2/bias:0"AquamanNet/netconv_1_2/bias/Assign"AquamanNet/netconv_1_2/bias/read:02:AquamanNet/netconv_1_2/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_2_1/kernel:0$AquamanNet/netconv_2_1/kernel/Assign$AquamanNet/netconv_2_1/kernel/read:02<AquamanNet/netconv_2_1/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_2_1/bias:0"AquamanNet/netconv_2_1/bias/Assign"AquamanNet/netconv_2_1/bias/read:02:AquamanNet/netconv_2_1/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_2_2/kernel:0$AquamanNet/netconv_2_2/kernel/Assign$AquamanNet/netconv_2_2/kernel/read:02<AquamanNet/netconv_2_2/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_2_2/bias:0"AquamanNet/netconv_2_2/bias/Assign"AquamanNet/netconv_2_2/bias/read:02:AquamanNet/netconv_2_2/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_3_1/kernel:0$AquamanNet/netconv_3_1/kernel/Assign$AquamanNet/netconv_3_1/kernel/read:02<AquamanNet/netconv_3_1/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_3_1/bias:0"AquamanNet/netconv_3_1/bias/Assign"AquamanNet/netconv_3_1/bias/read:02:AquamanNet/netconv_3_1/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_3_2/kernel:0$AquamanNet/netconv_3_2/kernel/Assign$AquamanNet/netconv_3_2/kernel/read:02<AquamanNet/netconv_3_2/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_3_2/bias:0"AquamanNet/netconv_3_2/bias/Assign"AquamanNet/netconv_3_2/bias/read:02:AquamanNet/netconv_3_2/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_4_1/kernel:0$AquamanNet/netconv_4_1/kernel/Assign$AquamanNet/netconv_4_1/kernel/read:02<AquamanNet/netconv_4_1/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_4_1/bias:0"AquamanNet/netconv_4_1/bias/Assign"AquamanNet/netconv_4_1/bias/read:02:AquamanNet/netconv_4_1/bias/Initializer/truncated_normal:08
­
AquamanNet/netconv_4_2/kernel:0$AquamanNet/netconv_4_2/kernel/Assign$AquamanNet/netconv_4_2/kernel/read:02<AquamanNet/netconv_4_2/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/netconv_4_2/bias:0"AquamanNet/netconv_4_2/bias/Assign"AquamanNet/netconv_4_2/bias/read:02:AquamanNet/netconv_4_2/bias/Initializer/truncated_normal:08
­
AquamanNet/net_dense_1/kernel:0$AquamanNet/net_dense_1/kernel/Assign$AquamanNet/net_dense_1/kernel/read:02<AquamanNet/net_dense_1/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/net_dense_1/bias:0"AquamanNet/net_dense_1/bias/Assign"AquamanNet/net_dense_1/bias/read:02:AquamanNet/net_dense_1/bias/Initializer/truncated_normal:08
­
AquamanNet/net_dense_2/kernel:0$AquamanNet/net_dense_2/kernel/Assign$AquamanNet/net_dense_2/kernel/read:02<AquamanNet/net_dense_2/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/net_dense_2/bias:0"AquamanNet/net_dense_2/bias/Assign"AquamanNet/net_dense_2/bias/read:02:AquamanNet/net_dense_2/bias/Initializer/truncated_normal:08
­
AquamanNet/net_dense_3/kernel:0$AquamanNet/net_dense_3/kernel/Assign$AquamanNet/net_dense_3/kernel/read:02<AquamanNet/net_dense_3/kernel/Initializer/truncated_normal:08
Ѕ
AquamanNet/net_dense_3/bias:0"AquamanNet/net_dense_3/bias/Assign"AquamanNet/net_dense_3/bias/read:02:AquamanNet/net_dense_3/bias/Initializer/truncated_normal:08"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"%
saved_model_main_op


group_deps"
regularization_losses
ў
:AquamanNet/netconv_0_0/kernel/Regularizer/l2_regularizer:0
8AquamanNet/netconv_0_0/bias/Regularizer/l2_regularizer:0
:AquamanNet/netconv_0_1/kernel/Regularizer/l2_regularizer:0
8AquamanNet/netconv_0_1/bias/Regularizer/l2_regularizer:0
:AquamanNet/netconv_1_1/kernel/Regularizer/l2_regularizer:0
8AquamanNet/netconv_1_1/bias/Regularizer/l2_regularizer:0
:AquamanNet/netconv_1_2/kernel/Regularizer/l2_regularizer:0
8AquamanNet/netconv_1_2/bias/Regularizer/l2_regularizer:0
:AquamanNet/netconv_2_1/kernel/Regularizer/l2_regularizer:0
8AquamanNet/netconv_2_1/bias/Regularizer/l2_regularizer:0
:AquamanNet/netconv_2_2/kernel/Regularizer/l2_regularizer:0
8AquamanNet/netconv_2_2/bias/Regularizer/l2_regularizer:0
:AquamanNet/netconv_3_1/kernel/Regularizer/l2_regularizer:0
8AquamanNet/netconv_3_1/bias/Regularizer/l2_regularizer:0
:AquamanNet/netconv_3_2/kernel/Regularizer/l2_regularizer:0
8AquamanNet/netconv_3_2/bias/Regularizer/l2_regularizer:0
:AquamanNet/netconv_4_1/kernel/Regularizer/l2_regularizer:0
8AquamanNet/netconv_4_1/bias/Regularizer/l2_regularizer:0
:AquamanNet/netconv_4_2/kernel/Regularizer/l2_regularizer:0
8AquamanNet/netconv_4_2/bias/Regularizer/l2_regularizer:0
:AquamanNet/net_dense_1/kernel/Regularizer/l2_regularizer:0
8AquamanNet/net_dense_1/bias/Regularizer/l2_regularizer:0
:AquamanNet/net_dense_2/kernel/Regularizer/l2_regularizer:0
8AquamanNet/net_dense_2/bias/Regularizer/l2_regularizer:0
:AquamanNet/net_dense_3/kernel/Regularizer/l2_regularizer:0
8AquamanNet/net_dense_3/bias/Regularizer/l2_regularizer:0*щ
serving_defaultе
/
frame_10#
Placeholder_10:0џџџџџџџџџ
/
frame_11#
Placeholder_11:0џџџџџџџџџ
/
frame_12#
Placeholder_12:0џџџџџџџџџ
/
frame_13#
Placeholder_13:0џџџџџџџџџ
+
frame_0 
Placeholder:0џџџџџџџџџ
/
frame_14#
Placeholder_14:0џџџџџџџџџ
-
frame_1"
Placeholder_1:0џџџџџџџџџ
/
frame_15#
Placeholder_15:0џџџџџџџџџ
-
frame_2"
Placeholder_2:0џџџџџџџџџ
-
frame_3"
Placeholder_3:0џџџџџџџџџ
-
frame_4"
Placeholder_4:0џџџџџџџџџ
-
frame_5"
Placeholder_5:0џџџџџџџџџ
-
frame_6"
Placeholder_6:0џџџџџџџџџ
-
frame_7"
Placeholder_7:0џџџџџџџџџ
-
frame_8"
Placeholder_8:0џџџџџџџџџ
-
frame_9"
Placeholder_9:0џџџџџџџџџ=
output3
AquamanNet/net_dense_3/Elu:0џџџџџџџџџtensorflow/serving/predict*р
logitsе
-
frame_8"
Placeholder_8:0џџџџџџџџџ
-
frame_9"
Placeholder_9:0џџџџџџџџџ
/
frame_10#
Placeholder_10:0џџџџџџџџџ
/
frame_11#
Placeholder_11:0џџџџџџџџџ
/
frame_12#
Placeholder_12:0џџџџџџџџџ
/
frame_13#
Placeholder_13:0џџџџџџџџџ
+
frame_0 
Placeholder:0џџџџџџџџџ
/
frame_14#
Placeholder_14:0џџџџџџџџџ
-
frame_1"
Placeholder_1:0џџџџџџџџџ
/
frame_15#
Placeholder_15:0џџџџџџџџџ
-
frame_2"
Placeholder_2:0џџџџџџџџџ
-
frame_3"
Placeholder_3:0џџџџџџџџџ
-
frame_4"
Placeholder_4:0џџџџџџџџџ
-
frame_5"
Placeholder_5:0џџџџџџџџџ
-
frame_6"
Placeholder_6:0џџџџџџџџџ
-
frame_7"
Placeholder_7:0џџџџџџџџџ=
output3
AquamanNet/net_dense_3/Elu:0џџџџџџџџџtensorflow/serving/predict