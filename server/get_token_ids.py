import tiktoken

encodings = {
    "gpt-4": tiktoken.get_encoding("cl100k_base"),
    "gpt-4-32k": tiktoken.get_encoding("cl100k_base"),
    "gpt-3.5-turbo": tiktoken.get_encoding("cl100k_base"),
    "gpt-3.5-turbo-0301": tiktoken.get_encoding("cl100k_base"),
    "text-davinci-003": tiktoken.get_encoding("p50k_base"),
    "text-davinci-002": tiktoken.get_encoding("p50k_base"),
    "text-davinci-001": tiktoken.get_encoding("r50k_base"),
    "text-curie-001": tiktoken.get_encoding("r50k_base"),
    "text-babbage-001": tiktoken.get_encoding("r50k_base"),
    "text-ada-001": tiktoken.get_encoding("r50k_base"),
    "davinci": tiktoken.get_encoding("r50k_base"),
    "curie": tiktoken.get_encoding("r50k_base"),
    "babbage": tiktoken.get_encoding("r50k_base"),
    "ada": tiktoken.get_encoding("r50k_base"),
}

max_length = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "text-davinci-003": 4096,
    "text-davinci-002": 4096,
    "text-davinci-001": 2049,
    "text-curie-001": 2049,
    "text-babbage-001": 2049,
    "text-ada-001": 2049,
    "davinci": 2049,
    "curie": 2049,
    "babbage": 2049,
    "ada": 2049
}

def count_tokens(model_name, text):
    return len(encodings[model_name].encode(text))

def get_max_context_length(model_name):
    return max_length[model_name]

def get_token_ids_for_task_parsing(model_name):
    text = '''{"task": "text-classification",  "token-classification", "text2text-generation", "summarization", "translation",  "question-answering", "conversational", "text-generation", "sentence-similarity", "tabular-classification", "object-detection", "image-classification", "image-to-image", "image-to-text", "text-to-image", "visual-question-answering", "document-question-answering", "image-segmentation", "text-to-speech", "text-to-video", "automatic-speech-recognition", "audio-to-audio", "audio-classification", "canny-control", "hed-control", "mlsd-control", "normal-control", "openpose-control", "canny-text-to-image", "depth-text-to-image", "hed-text-to-image", "mlsd-text-to-image", "normal-text-to-image", "openpose-text-to-image", "seg-text-to-image", "args", "text", "path", "dep", "id", "<GENERATED>-"}'''
    res = encodings[model_name].encode(text)
    res = list(set(res))
    return res

def get_token_ids_for_choose_model(model_name):
    text = '''{"id": "reason"}'''
    res = encodings[model_name].encode(text)
    res = list(set(res))
    return res