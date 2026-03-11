import google.generativeai as genai

# Configure API Key
genai.configure(api_key="AIzaSyB2yBK0PIFPmou7Kwbj3AmJBP9TWrHb_Nw")

# List all available models
for model in genai.list_models():
    print(model.name, "->", model.supported_generation_methods,model.description)