# IR Text Grammar (EBNF)

Terminals are quoted strings; non-terminals are bare identifiers.
ISO/IEC 14977 EBNF: `[ X ]` = optional (zero or one); `{ X }` = zero or more repetition.

## Top-level

```ebnf
program                 ::= 'program' attr_list '{' { function } '}';
function                ::= 'function' ident
                           'incast' '(' [ var_def_list ] ')'
                           'outcast' '(' [ type_list ] ')'
                           attr_list body;
```

## Attributes

```ebnf
attr                    ::= attrname '(' attr_value ')';
attr_value              ::= expr | ident | attr_value_list;
attr_value_list         ::= '[' [ attr_value { ',' attr_value } ] ']';
attr_list               ::= { attr }
```

## Types

```ebnf
type                    ::= 'unknown' | dtype | 'memref_type' | 'token' | 'none'
                         |  'ptr' '<' dtype '>'
                         |  'tuple' '<' [ type_list ] '>'
                         |  'tensor' '<' shape ',' dtype ',' tensor_view '>'
                         |  'tile' '<' shape ',' dtype ',' tile_view ',' hw_info '>'
                         |  v0_logical_tensor;
type_list               ::= type { ',' type };
shape                   ::= expr { 'x' expr };
tensor_view             ::= 'tensor_view' '<' [ shape ',' shape ',' ident [ ',' expr ] ] '>';
tile_view               ::= 'tile_view' '<' [ shape ',' shape ',' ( expr | 'null' ) ] '>';
hw_info                 ::= 'hw_info' '<' [ ident ',' ident ',' int ',' ident    ',' ident ] '>';
v0_logical_tensor       ::= 'v0_logical_tensor';
```

## Expressions

```ebnf
expr                    ::= var_ref | int | float | bool
                         |  '(' expr bop expr ')'
                         |  '(' uop [ dtype ] expr ')'
                         |  ident '(' [ call_arg_list ] ')'
                         |  'tuple' '(' [ expr_list ] ')'
                         |  'getitem' '(' expr ',' expr ')'
                         |  'memref' '(' ident ',' expr ',' int ')'
                         |  v0_scalar_expr;
var_ref                 ::= varname;
var_def                 ::= type varname attr_list;
var_def_list            ::= var_def { ',' var_def };
call_arg                ::= expr | ident '=' attr_value;
call_arg_list           ::= [ call_arg { ',' call_arg } ];
expr_list               ::= expr { ',' expr };
v0_scalar_expr          ::= v0_scalar_op '(' [ v0_scalar_expr_arg_list ] ')'
v0_scalar_expr_arg      ::= v0_scalar_expr | ident | int;
v0_scalar_expr_arg_list ::= v0_scalar_expr_arg { ',' v0_scalar_expr_arg };
iter_args               ::= 'iter' '{' { var_def '=' expr ';' } '}';
```

## Statements

```ebnf
stmt                    ::= seq_stmts | yield_stmt | break_stmt | continue_stmt | return_stmt
                         |  section_stmt | eval_stmt | assign_stmt | if_stmt | for_stmt
                         |  while_stmt | scalar_op_stmt | tensor_op_stmt;
seq_stmts               ::= '{' { stmt } '}';
yield_stmt              ::= 'yield' [ expr_list ] ';';
break_stmt              ::= 'break' [ expr_list ] ';';
continue_stmt           ::= 'continue' [ expr_list ] ';';
return_stmt             ::= 'return' [ expr_list ] ';';
section_stmt            ::= 'section' ident body;
eval_stmt               ::= expr ';';  // deprecated: exists in current code, to be removed
assign_stmt             ::= [ var_def_list '=' ] expr ';';
if_stmt                 ::= [ var_def_list '=' ] 'if' expr 'then' body 'else' body;
for_stmt                ::= [ var_def_list '=' ] 'for' var_ref 'inrange' expr ',' expr ',' expr
                           iter_args attr_list body;
while_stmt              ::= [ var_def_list '=' ] 'while' expr iter_args body;
scalar_op_stmt          ::= [ var_def_list '=' ] ident '(' [ expr_list ] ')' ';';
tensor_op_stmt          ::= [ var_def_list '=' ] [ v0_tensor_opmagic ] ident '(' [ expr_list ] ')'
                           'token' '(' [ var_ref_list ] ')' attr_list ';';
body                    ::= stmt;
var_ref_list            ::= var_ref { ',' var_ref };
```

## Lexical tokens

```ebnf
ident                   ::= [A-Za-z_][A-Za-z0-9_]*;
varname                 ::= '%' ( [A-Za-z_][A-Za-z0-9_.]* | [0-9]+ ) [ '@' ( [A-Za-z_][A-Za-z0-9_.]* | [0-9]+ ) ];
attrname                ::= '#' [A-Za-z_][A-Za-z0-9_.]*;
v0_tensor_opmagic       ::= '!' [0-9]+;
int                     ::= [-+]?[0-9]+;
float                   ::= [-+]?\d+\.\d+([eE][-+]?\d+)? | [-+]?\d+[eE][-+]?\d+;
bool                    ::= 'true' | 'false';
dtype                   ::= 'int8_t' | 'int16_t' | 'int32_t' | 'int64_t'
                         |  'uint8_t' | 'uint16_t' | 'uint32_t' | 'uint64_t'
                         |  'half' | 'float' | 'double' | 'bfloat16_t'
                         |  'fp8_e4m3fn' | 'fp8_e5m2' | 'hf4' | 'hf8' | 'bool';
bop                     ::= 'add' | 'sub' | 'mul' | 'div' | 'mod' | 'fdiv'
                         |  'min' | 'max' | 'pow'
                         |  'eq' | 'ne' | 'lt' | 'le' | 'gt' | 'ge'
                         |  'land' | 'lor' | 'lxor'
                         |  'and' | 'or' | 'xor' | 'shl' | 'shr';
uop                     ::= 'abs' | 'neg' | 'not' | 'inv' | 'cast';
v0_scalar_op            ::= 'v0pos' | 'v0neg' | 'v0not'
                         |  'v0add' | 'v0sub' | 'v0mul' | 'v0div' | 'v0mod'
                         |  'v0eq' | 'v0ne' | 'v0lt' | 'v0le' | 'v0gt' | 'v0ge'
                         |  'v0call' | 'v0min' | 'v0max' | 'v0and' | 'v0or';
```
