import importlib.util

def load_text(fpaths, by_lines=False):
    with open(fpaths, "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()


def func_name_from_code(code):
    code = code.split("def ")[1]
    name = code.split("(agent, env, env_info, trajectory):")[0]
    return name

def get_code_from_response(response):
    response = response.split("```python\n")[1]
    response = response.split("```")[0]
    return response

def postprocess(response):
    
    code = get_code_from_response(response)
    name = func_name_from_code(code)
    
    prefix = load_text("function/prefix.txt")
    
    code = prefix + code
    
    with open("function/func.py", "w") as text_file:
        text_file.write(code)
        
    return name

def import_class_from_file(file_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name)
    return function

