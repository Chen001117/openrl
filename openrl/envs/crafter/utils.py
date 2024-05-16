import importlib.util

def load_text(fpaths, by_lines=False):
    with open(fpaths, "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()

def func_name_from_code(code):
    code = code.split("def ")[1]
    name = code.split("(env, agent, observation):")[0]
    return name

def get_code_from_response(response):
    response = response.split("```python\n")[1]
    response = response.split("```")[0]
    return response

def postprocess(response, file_name):
    
    code = get_code_from_response(response)
    func_name = func_name_from_code(code)
    
    with open(file_name, "w") as text_file:
        text_file.write(code)
    
    spec = importlib.util.spec_from_file_location("module.name", file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, func_name)
    
    return function, code

