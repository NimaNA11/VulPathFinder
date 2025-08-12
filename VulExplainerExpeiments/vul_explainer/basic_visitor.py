from CppCodeAnalyzer.mainTool.ast.astNode import ASTNode, ASTNodeVisitor
from CppCodeAnalyzer.mainTool.ast.expressions.expression import Identifier, ClassStaticIdentifier
from CppCodeAnalyzer.mainTool.ast.expressions.binaryExpressions import RelationalExpression, EqualityExpression

from CppCodeAnalyzer.mainTool.ast.expressions.postfixExpressions import CallExpression
from CppCodeAnalyzer.mainTool.ast.expressions.primaryExpressions import PrimaryExpression

from typing import List, Dict, Set

# extract variable names from an expression (An AST Node)
def extract_vars(item: ASTNode) -> Set[str]:
    vars: Set[str] = set()
    if isinstance(item, Identifier) and not isinstance(item, ClassStaticIdentifier):
        vars.add(item.getEscapedCodeStr())
    else:
        for i in range(item.getChildCount()):
            sub_vars = extract_vars(item.getChild(i))
            vars.update(sub_vars)
    return vars

# evaluate whether a statement (ASTNode) could be a sink point
class SinkVisitor(ASTNodeVisitor):
    def __init__(self):
        self.reset()
        self.conds: List[ASTNode] = list() # Stores potential constraint conditions, such as constant expressions that do not belong to this category

    def reset(self):
        self.isSink: bool = False  # Is the CFG node (statement) a PSP
        self.isCond: bool = False # Is the CFG a condition expression?
        self.potential: bool = False  # Need to evaluate the dependence relations to further judge whether it could be a sink
        self.potential_var: Identifier = None # The variable that is related to PSP
        self.check_upper: bool = False  # Whether to check the upper bound
        self.check_lower: bool = False  # Whether to check the lower bound
        self.key_vars: Set[str] = set()
        self.conds: List[ASTNode] = list()  # Stores potential constraint conditions, such as constant expressions that do not belong to this category
    # Extract key variable names

    def extract_vars(self, item: ASTNode) -> Set[str]:
        vars: Set[str] = set()
        if isinstance(item, Identifier) and not isinstance(item, ClassStaticIdentifier):
            vars.add(item.getEscapedCodeStr())
        else:
            for i in range(item.getChildCount()):
                sub_vars = self.extract_vars(item.getChild(i))
                vars.update(sub_vars)
        return vars


# Used to search for conditional expressions
class RelationExprVisitor(ASTNodeVisitor):
    def __init__(self):
        self.containsRelationExpr: bool = False

    def visit(self, item: ASTNode):
        if isinstance(item, RelationalExpression) or isinstance(item, EqualityExpression):
            self.containsRelationExpr = True
        super().visit(item)

# Evaluate whether an expression (ASTNode) is a variable or a constant expression
# If it is a constant, return false. Otherwise, return true
def checkConstantExpression(item: ASTNode) -> bool:
    # This expression is a constant
    if isinstance(item, PrimaryExpression):
        return False
    # Check whether the parameter is constant
    elif isinstance(item, CallExpression):
        # The first child node is the parameter list
        return checkConstantExpression(item.getChild(1))
    # Usage of variable names
    elif isinstance(item, Identifier):
        return True
    flag = False
    for i in range(item.getChildCount()):
        flag |= checkConstantExpression(item.getChild(i))
    return flag

# Check whether the expression is a conditional expression
def checkisREexpr(item: ASTNode) -> bool:
    visitor: RelationExprVisitor = RelationExprVisitor()
    item.accept(visitor)
    return visitor.containsRelationExpr

# Check whether the variable has corresponding constraint conditions
def checkDependence(sink_idx: int, sink_var: Identifier, check_upper: bool, check_lower: bool,
                    cdg_precs: Dict[int, List[int]], stmts: List[ASTNode]) -> bool:
    queue: List[int] = [prec_idx for prec_idx in cdg_precs.get(sink_idx, [])]
    visited: Set[int] = set()
    while len(queue) > 0:
        cond_idx: int = queue.pop(0)
        visited.add(cond_idx)
        cond: ASTNode = stmts[cond_idx]
        if checkCondition(sink_var, cond, check_upper, check_lower):
            return True
        for prec_idx in cdg_precs.get(cond_idx, []):
            if prec_idx not in visited:
                queue.append(prec_idx)
    return False


def checkCondition(sink_var: Identifier, condition: ASTNode,
                   check_upper: bool, check_lower: bool) -> bool:
    if isinstance(condition, RelationalExpression):
        # f(var) < xxx
        if sink_var.getEscapedCodeStr() in condition.getChild(0).getEscapedCodeStr() \
            and condition.operator == "<" and check_upper:
            return True
        if sink_var.getEscapedCodeStr() in condition.getChild(0).getEscapedCodeStr() \
            and condition.operator == ">" and check_lower:
            return True
        return False

    flag: bool = False
    for i in range(condition.getChildCount()):
        flag |= checkCondition(sink_var, condition.getChild(i), check_upper, check_lower)
    return flag
