from vul_explainer.basic_visitor import SinkVisitor, checkConstantExpression, checkisREexpr

from CppCodeAnalyzer.mainTool.ast.astNode import ASTNode
from CppCodeAnalyzer.mainTool.ast.expressions.expression import ArrayIndexing, UnaryOp, Identifier
from CppCodeAnalyzer.mainTool.ast.expressions.binaryExpressions import BinaryExpression, AssignmentExpr
from CppCodeAnalyzer.mainTool.ast.expressions.postfixExpressions import IncDecOp, NewExpression, MemberAccess, CallExpression
from CppCodeAnalyzer.mainTool.ast.expressions.primaryExpressions import IntegerExpression
from CppCodeAnalyzer.mainTool.ast.expressions.expressionHolders import Condition

from typing import List, Dict, Set

class BufferOverflowVisitor(SinkVisitor):
    # Buffer overflow vulnerabilities mainly occur in string copying, array/pointer access, including ptrMemberAccess, CWE119, CWE125, CWE787
    def __init__(self):
        super().__init__()
        self.cpy_funcs: List[str] = ['memcpy', 'memmove', 'memcmp', 'strncpy', 'wcsncpy', 'strcpy',
                                     'wcscpy', 'strncat', 'strcat', 'wcscat', 'snprintf', 'wcsncat']

    def visit(self, item: ASTNode):
        # print("ASTNode")
        # print(item)
        if isinstance(item, CallExpression):
            if item.getChild(0).getEscapedCodeStr().lower() in self.cpy_funcs \
                    or item.getChild(0).getEscapedCodeStr() in self.cpy_funcs:
                self.isSink = True
                # All variables in the parameter list need to be visited
                self.key_vars.update(self.extract_vars(item.getChild(1)))
                return
        # Array Usage
        elif isinstance(item, ArrayIndexing):
            # Check if its indices are all constants
            if checkConstantExpression(item.getChild(1)):
                self.key_vars.update(self.extract_vars(item.getChild(0)))
                self.key_vars.update(self.extract_vars(item.getChild(1)))
                self.isSink = True
            return
        # Pointer Usage
        elif isinstance(item, UnaryOp):
            if item.operator == '*':
                self.key_vars.update(self.extract_vars(item))
                self.isSink = True
                return
        super().visit(item)


class IncorrectCalculationVisitor(SinkVisitor):
    # Integer overflow vulnerabilities mainly occur in arithmetic operations, CWE190 CWE191 CWE369
    def __init__(self):
        super().__init__()

    def visit(self, item: ASTNode):
        if isinstance(item, BinaryExpression):
            if item.operator in {'+', '-', '*', '<<', '>>', '/', '%'}:
                self.isSink = True
                self.key_vars.update(self.extract_vars(item))
                return
        # Increment assignment
        elif isinstance(item, AssignmentExpr):
            if item.operator in {"+=", "-=", "*=", ">>=", "<<=", '/=', '%='}:
                self.isSink = True
                self.key_vars.update(self.extract_vars(item))
                return
        # x++ / x-- / ++x / --x
        elif isinstance(item, IncDecOp):
            self.potential = True
            self.key_vars.update(self.extract_vars(item))
            for i in range(item.getChildCount()):
                # Check whether there are path constraints on ++ operation
                if isinstance(item.getChild(i), Identifier):
                    self.potential_var = item.getChild(i)
                    if "++" in item.getEscapedCodeStr():
                        self.check_upper = True
                    elif "--" in item.getEscapedCodeStr():
                        self.check_lower = True
                    break
            return
        self.visitChildren(item)


# Backward
class BackwardLeakVisitor(SinkVisitor):
    def __init__(self):
        super().__init__()
        self.resource_funcs: Set[str] = {"fopen", "open", "_wfopen", "_wopen", "_open", "opendir",
                           "_wfreopen_s", "freopen64", "open64", "fopen64", "fopen_s", "freopen", "freopen_s", "CreateDirectory",
                           "CreateFileA", "CreateFileW", "CreateFile", "CreateFileTransacted", "createFileTransactedA",
                            'alloca', 'malloc', 'realloc', 'oballoc', 'mem_realloc', 'calloc', '_alloca', 'strdup', "sleep", "fwrite", "write",
                            'HeapAlloc', 'nhalloc', 'valloc', 'xalloc', 'xrealloc'}
        self.member_funcs: Set[str] = {"open"}

    def visit(self, item: ASTNode):
        if isinstance(item, CallExpression):
            if item.getChild(0).getEscapedCodeStr().lower() in self.resource_funcs \
                    or item.getChild(0).getEscapedCodeStr() in self.resource_funcs:
                self.isSink = True
                self.key_vars.update(self.extract_vars(item.getChild(1)))
                return
            else:
                if isinstance(item.getChild(0).getChild(0), MemberAccess):
                    memAccess: MemberAccess = item.getChild(0).getChild(0)
                    # Get member function name
                    funcName: Identifier = memAccess.getChild(1)
                    if funcName.getEscapedCodeStr() in self.member_funcs:
                        self.isSink = True
                        self.key_vars.update(self.extract_vars(item.getChild(1)))
                        return
        elif isinstance(item, NewExpression):
            self.isSink = True
            self.key_vars.update(self.extract_vars(item))
            return

        self.visitChildren(item)


class PathTraversalVisitor(SinkVisitor):
    def __init__(self):
        super().__init__()
        self.file_funcs = {"fopen", "open", "_wfopen", "_wopen", "_open", "opendir",
                           "_wfreopen_s", "freopen64", "open64", "fopen64", "fopen_s", "freopen", "freopen_s", "CreateDirectory",
                           "CreateFileA", "CreateFileW", "CreateFile", "CreateFileTransacted", "createFileTransactedA"}
        self.member_funcs = {"open"}


    def visit(self, item: ASTNode):
        if isinstance(item, CallExpression):
            # Direct function call
            if item.getChild(0).getEscapedCodeStr().lower() in self.file_funcs \
                    or item.getChild(0).getEscapedCodeStr() in self.file_funcs:
                self.isSink = True
                self.key_vars.update(self.extract_vars(item.getChild(1)))
                return
            else:
                if isinstance(item.getChild(0).getChild(0), MemberAccess):
                    memAccess: MemberAccess = item.getChild(0).getChild(0)
                    # Get member function name
                    funcName: Identifier = memAccess.getChild(1)
                    if funcName.getEscapedCodeStr() in self.member_funcs:
                        self.isSink = True
                        self.key_vars.update(self.extract_vars(item.getChild(1)))
                        return

        self.visitChildren(item)


class CommandInjectionVisitor(SinkVisitor):
    def __init__(self):
        super().__init__()
        self.system_api = {'_spawnl', 'system', 'execlp', 'execve', 'execle', 'execv', 'execl', 'popen',
                  '_spawnvp', '_spawnlp', 'execvp', '_wspawnvp', '_spawnv',
                  '_wspawnl', '_wspawnv', '_wspawnlp'}

    def visit(self, item: ASTNode):
        if isinstance(item, CallExpression):
            # Direct function call
            if item.getChild(0).getEscapedCodeStr().lower() in self.system_api:
                self.isSink = True
                self.key_vars.update(self.extract_vars(item.getChild(1)))
                return

        self.visitChildren(item)


class UncontrolledFormatVisitor(SinkVisitor):
    # CWE134
    def __init__(self):
        super(UncontrolledFormatVisitor, self).__init__()
        self.apis: Dict[str, int] =  {
            'vfprintf': 1, 'fwprintf': 1, 'sprintf': 1, 'wprintf': 0, 'fprintf': 1, 'vfwprintf': 1, 'vsnprintf': 2, 'vprintf': 0, 'vwprintf': 0, 'snprintf': 2, 'sprintf_s': 2,
            '_vsnwprintf': 2, 'printf': 0, 'vsprintf_s': 2, '_vsprintf_s_l': 2, 'vswprintf_s': 2, '_vswprintf_s_l': 2, 'vswprintf': 2, 'asprintf': 1, 'swprintf_s': 1,
            'wvsprintfA': 1, 'wvnsprintfA': 2, '_snprintf_s_l': 3, 'vsnprintf_s': 2, '_snwprintf': 2, '_sntprintf': 2, '_snprintf': 2, 'vsprintf': 1, '_snprintf_s': 3,
            '_vsnwprintf_s_l': 3, 'wsprintfW': 1, '_snwprintf_s_l': 3, '_cprintf_s': 0, '_cprintf_s_l': 0, '_cwprintf_s': 0, '_cwprintf_s_l': 0, 'wsprintf': 1, 'wnsprintfW': 2,
            '_vsnwprintf_s': 3, '_vsntprintf': 2, 'vasprintf': 1, '_vsnprintf_s': 3, '_vsnprintf_s_l': 3, 'wnsprintfA': 2, 'wvsprintfW': 1, '_vsnprintf': 2, 'swprintf': 2,
            'wvnsprintfW': 2, '_snwprintf_s': 2, '_swprintf_s_l': 2, '_sprintf_s_l': 2, 'wnsprintf': 2, 'wvsprintf': 1, '_stprintf': 1
        }

    def visit(self, item: ASTNode):
        if isinstance(item, CallExpression):
            # Direct function call
            api_name = item.getChild(0).getEscapedCodeStr().lower()
            if api_name in self.apis.keys():
                self.isSink = True
                self.key_vars.update(self.extract_vars(item.getChild(1).getChild(self.apis[api_name])))
                return
        self.visitChildren(item)
