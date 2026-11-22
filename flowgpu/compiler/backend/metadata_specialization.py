"""Partial evaluation of host metadata, before exact-context guards are added.

Only tensor metadata is substituted. Device allocation, tensor data and all
kernel numerical bodies stay live. An unprovable guard remains in the source.
"""
import ast

from flowgpu.compiler.optimization.strategy import launch_value


def _contiguous(item):
    stride=1
    for size,step in reversed(list(zip(item['shape'],item['stride']))):
        if size != 1 and step != stride:return False
        stride*=size
    return True


def specialize_host(host, context):
    inputs={a.arg: item for a, item in zip(host.args.args, context)
            if item.get('kind') != 'scalar'}
    constants={}
    # This is partial evaluation, not general Python control-flow analysis.
    # Values written again (including inside a branch/loop) cannot keep an
    # earlier binding, and mutable literals are never compile-time values.
    writes={}
    for node in ast.walk(host):
        if isinstance(node,ast.Name) and isinstance(node.ctx,(ast.Store,ast.Del)):
            writes[node.id]=writes.get(node.id,0)+1
    inputs={name:item for name,item in inputs.items() if name not in writes}

    def immutable(value):
        return type(value) in {type(None),bool,int,float,str,bytes} or (
            type(value) is tuple and all(immutable(v) for v in value))

    class Metadata(ast.NodeTransformer):
        def visit_Name(self,node):
            if isinstance(node.ctx,ast.Load) and node.id in constants:
                return ast.copy_location(ast.Constant(constants[node.id]),node)
            return node

        def visit_Attribute(self,node):
            if isinstance(node.value,ast.Name) and node.value.id in inputs:
                item=inputs[node.value.id]
                values={'ndim':len(item['shape']),'shape':tuple(item['shape']),'is_cuda':True}
                if node.attr in values:return ast.copy_location(ast.Constant(values[node.attr]),node)
                if node.attr=='dtype':
                    if item['dtype'] not in {'torch.float16','torch.bfloat16','torch.float32','torch.int32','torch.int64','torch.bool'}:
                        raise ValueError('unsupported prepared dtype')
                    return ast.parse(item['dtype'],mode='eval').body
                if node.attr=='layout':return ast.parse('torch.strided',mode='eval').body
            return self.generic_visit(node)

        def visit_Call(self,node):
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                    and node.func.value.id in inputs and not node.keywords):
                item=inputs[node.func.value.id]
                if node.func.attr=='stride':
                    stride=tuple(item['stride'])
                    if not node.args:return ast.Constant(stride)
                    if len(node.args)==1 and isinstance(node.args[0],ast.Constant):
                        return ast.Constant(stride[node.args[0].value])
                if node.func.attr=='is_contiguous' and not node.args:return ast.Constant(_contiguous(item))
            node=self.generic_visit(node)
            name=ast.unparse(node.func)
            if name=='tuple' and len(node.args)==1 and isinstance(node.args[0],ast.Constant):
                return node.args[0]
            if name in {'triton.cdiv','triton.next_power_of_2','min','max'}:
                try:return ast.Constant(launch_value(ast.unparse(node),{}))
                except ValueError:pass
            return node

        def visit_Subscript(self,node):
            node=self.generic_visit(node)
            if (isinstance(node.ctx,ast.Load) and isinstance(node.value,ast.Constant)
                    and isinstance(node.slice,ast.Constant)):
                try:return ast.Constant(node.value.value[node.slice.value])
                except (IndexError,KeyError,TypeError):pass
            return node

        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            try:return ast.Constant(launch_value(ast.unparse(node),{}))
            except ValueError:return node

        def visit_Compare(self,node):
            node=self.generic_visit(node)
            if len(node.ops)!=1:return node
            left,right=node.left,node.comparators[0]
            if isinstance(node.ops[0],(ast.Eq,ast.NotEq)):
                # Only metadata attributes, not arbitrary tensor expressions.
                if isinstance(left,ast.Attribute) and isinstance(right,ast.Attribute):
                    if ast.dump(left)==ast.dump(right):return ast.Constant(isinstance(node.ops[0],ast.Eq))
                    if (left.attr==right.attr=='device' and isinstance(left.value,ast.Name)
                            and isinstance(right.value,ast.Name) and left.value.id in inputs and right.value.id in inputs):
                        same=inputs[left.value.id]['device']==inputs[right.value.id]['device']
                        return ast.Constant(same if isinstance(node.ops[0],ast.Eq) else not same)
            try:a,b=ast.literal_eval(left),ast.literal_eval(right)
            except (ValueError,TypeError):return node
            ops={ast.Eq:lambda:a==b,ast.NotEq:lambda:a!=b,ast.Lt:lambda:a<b,
                 ast.LtE:lambda:a<=b,ast.Gt:lambda:a>b,ast.GtE:lambda:a>=b}
            if type(node.ops[0]) in ops:
                try:return ast.Constant(ops[type(node.ops[0])]())
                except (TypeError,ValueError):pass
            return node

        def visit_UnaryOp(self,node):
            node=self.generic_visit(node)
            if isinstance(node.op,ast.Not) and isinstance(node.operand,ast.Constant):
                return ast.Constant(not node.operand.value)
            return node

    transform=Metadata()
    body=[]
    for statement in host.body:
        statement=transform.visit(statement)
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
            name=statement.targets[0].id
            try:value=ast.literal_eval(statement.value)
            except (ValueError,TypeError):constants.pop(name,None)
            else:
                if writes.get(name)==1 and immutable(value):constants[name]=value
                else:constants.pop(name,None)
        if isinstance(statement,ast.If) and isinstance(statement.test,ast.Constant) and statement.test.value is False:
            body.extend(statement.orelse)
        else:body.append(statement)
    host.body=body
    return host
