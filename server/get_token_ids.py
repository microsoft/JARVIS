import tiktoken

encodings = {
    "gpt-3.5-turbo": tiktoken.get_encoding("cl100k_base"),
    "text-davinci-003": tiktoken.get_encoding("p50k_base"),
}

def count_tokens(model_name, text):
    return len(encodings[model_name].encode(text))


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