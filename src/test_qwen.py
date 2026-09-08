from qwen_model import model


response = model.invoke(
    "Explain what an MRI is in one sentence."
)

print(response.content)