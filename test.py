import sys

import asyncio

import sglang as sgl

def trace_calls(frame, event, arg):
    if event == "call":
        code = frame.f_code
        func_name = code.co_name
        filename = code.co_filename
        lineno = frame.f_lineno
        print(f"CALL {func_name}  ({filename}:{lineno})")
    return trace_calls

sys.setprofile(trace_calls)

model_path = "meta-llama/Llama-3.2-1B"

llm = sgl.Engine(model_path=model_path)

prompts = ["Explain what RMSNorm does in one sentence.",]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

output = llm.generate(
    prompts, sampling_params
)

print("\n=== MODEL OUTPUT ===")
print(output)
