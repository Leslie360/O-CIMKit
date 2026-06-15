import os
import ast
import json

def analyze_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
    except Exception as e:
        return None

    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    
    methods = []
    for cls in classes:
        methods.extend([n for n in cls.body if isinstance(n, ast.FunctionDef)])
        
    all_callables = functions + methods
    
    docstring_count = 0
    type_hint_count = 0
    
    for node in all_callables:
        if ast.get_docstring(node):
            docstring_count += 1
        
        has_annotation = False
        if node.returns is not None:
            has_annotation = True
        for arg in node.args.args:
            if arg.annotation is not None and arg.arg != 'self':
                has_annotation = True
        if has_annotation:
            type_hint_count += 1
            
    return {
        "callables": len(all_callables),
        "docstrings": docstring_count,
        "type_hints": type_hint_count,
        "classes": len(classes)
    }

def main():
    print("🚀 [Skill] CIM Platform Engineering Audit")
    print("Reference Standards: IBM AIHWKit, Intel Lava, PyTorch\n")
    
    targets = ["core", "cli", "profiles"]
    
    total_callables = 0
    total_docstrings = 0
    total_type_hints = 0
    
    files_to_check = []
    for d in targets:
        if not os.path.exists(d): continue
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.endswith('.py'):
                    files_to_check.append(os.path.join(root, file))
                    
    for file in files_to_check:
        stats = analyze_file(file)
        if stats:
            total_callables += stats["callables"]
            total_docstrings += stats["docstrings"]
            total_type_hints += stats["type_hints"]
        
    print("📊 Quantitative Audit Results:")
    print("=" * 50)
    
    doc_cov = (total_docstrings / total_callables) * 100 if total_callables > 0 else 0
    type_cov = (total_type_hints / total_callables) * 100 if total_callables > 0 else 0
    
    print(f"Total API Endpoints (Functions/Methods): {total_callables}")
    print(f"Docstring Coverage: {doc_cov:.1f}% (Target: >80%)")
    print(f"Type Hint Coverage: {type_cov:.1f}% (Target: >60%)\n")
    
    print("🔍 Qualitative Architectural Assessment:")
    print("=" * 50)
    if doc_cov < 80:
        print("❌ [Docs] Lacking rigorous API documentation. Top-tier frameworks use Sphinx/Google docstrings.")
    else:
        print("✅ [Docs] Documentation coverage meets standards.")
        
    if type_cov < 50:
        print("❌ [Types] Missing Python Type Hints. PyTorch 2.0+ and IBM AIHWKit heavily enforce strict typing for IDE intellisense (mypy).")
    else:
        print("✅ [Types] Static typing coverage is adequate.")

if __name__ == "__main__":
    main()
